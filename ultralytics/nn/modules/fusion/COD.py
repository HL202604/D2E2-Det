import torch
import torch.nn as nn
import torch.nn.functional as F
class COD(nn.Module):

    def __init__(self, c_rgb, c_ir):
        super().__init__()

        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_ir = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.proj_rgb = nn.Identity() if c_rgb == c_rgb else nn.Conv2d(c_rgb, c_rgb, 1, bias=False)
        self.proj_ir = nn.Identity() if c_ir == c_rgb else nn.Conv2d(c_ir, c_rgb, 1, bias=False)
        self.proj_out_rgb = nn.Identity() if c_rgb == c_rgb else nn.Conv2d(c_rgb, c_rgb, 1, bias=False)
        self.proj_out_ir = nn.Identity() if c_rgb == c_ir else nn.Conv2d(c_rgb, c_ir, 1, bias=False)

        self.offset_conv = nn.Conv2d(c_rgb * 2, 2, kernel_size=3, padding=1)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        self.rgb_disentangler = CosineSimilarityDisentangler(c_rgb)
        self.ir_disentangler = CosineSimilarityDisentangler(c_rgb)

    def _spatial_align(self, src_feat, ref_feat):

        concat_feat = torch.cat([src_feat, ref_feat], dim=1)
        offset = self.offset_conv(concat_feat)

        B, _, H, W = src_feat.shape

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=src_feat.device, dtype=src_feat.dtype),
            torch.linspace(-1, 1, W, device=src_feat.device, dtype=src_feat.dtype),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

        offset = offset.to(src_feat.dtype)
        warped_grid = grid + offset.permute(0, 2, 3, 1)

        aligned_feat = F.grid_sample(src_feat, warped_grid, align_corners=True)
        return aligned_feat

    def _cross_attention(self, query, key, value):

        B, C, H, W = query.shape

        q = query.view(B, C, -1)
        k = key.view(B, C, -1).permute(0, 2, 1)
        v = value.view(B, C, -1)

        attn = torch.bmm(q, k) / (C ** 0.5)
        attn = self.softmax(attn)

        output = torch.bmm(attn, v)
        return output.view(B, C, H, W)

    def forward(self, rgb, ir):

        rgb_proj = self.proj_rgb(rgb)
        ir_proj = self.proj_ir(ir)


        rgb_specific, rgb_invariant = self.rgb_disentangler(rgb_proj)
        ir_specific, ir_invariant = self.ir_disentangler(ir_proj)


        rgb_invariant_aligned = self._spatial_align(rgb_invariant, ir_invariant)
        ir_invariant_aligned = self._spatial_align(ir_invariant, rgb_invariant)


        rgb_fused = self._cross_attention(rgb_invariant_aligned, ir_invariant_aligned, rgb_invariant_aligned)
        ir_fused = self._cross_attention(ir_invariant_aligned, rgb_invariant_aligned, ir_invariant_aligned)


        rgb_combined = rgb_specific + rgb_fused
        ir_combined = ir_specific + ir_fused


        rgb_out = rgb_proj + self.gamma_rgb * rgb_combined
        ir_out = ir_proj + self.gamma_ir * ir_combined

        rgb_out = self.proj_out_rgb(rgb_out)
        ir_out = self.proj_out_ir(ir_out)

        return rgb_out, ir_out


class CosineSimilarityDisentangler(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.pos_dim = channels // 2

        self.pos_proj = nn.Conv2d(self.pos_dim, channels, 1)
        self.cosine_sim = nn.CosineSimilarity(dim=1)
        self.gate_conv = nn.Conv2d(1, channels, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.specific_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.invariant_conv = nn.Conv2d(channels, channels, 3, padding=1)

    def _get_cosine_pos_enc(self, H, W, device):

        y_pos = torch.arange(H, device=device).float()
        x_pos = torch.arange(W, device=device).float()

        y_pos = (y_pos / max(H - 1, 1)) * 2 - 1
        x_pos = (x_pos / max(W - 1, 1)) * 2 - 1

        pos_enc = torch.zeros(1, self.pos_dim, H, W, device=device)

        for i in range(self.pos_dim // 2):
            freq = (i + 1) * torch.pi
            pos_enc[0, 2 * i, :, :] = torch.sin(y_pos.unsqueeze(1) * freq)
            if 2 * i + 1 < self.pos_dim:
                pos_enc[0, 2 * i + 1, :, :] = torch.cos(x_pos.unsqueeze(0) * freq)

        return pos_enc

    def forward(self, x):
        B, C, H, W = x.shape

        pos_enc = self._get_cosine_pos_enc(H, W, x.device).to(x.dtype)
        pos_feat = self.pos_proj(pos_enc).repeat(B, 1, 1, 1)

        similarity = self.cosine_sim(x, pos_feat).unsqueeze(1)
        gate_weights = self.sigmoid(self.gate_conv(similarity))

        specific_feat = self.specific_conv(x * gate_weights)
        invariant_feat = self.invariant_conv(x * (1 - gate_weights))

        return specific_feat, invariant_feat