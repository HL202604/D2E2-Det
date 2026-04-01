import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math
class RGBChannelAttention(nn.Module):

    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

        self.k_multiscale = MultiScaleStructureExtractor(channels)

        self.v_directional = DirectionalColorExtractor(channels)
        assert self.head_dim * num_heads == channels, f"channels {channels} must be divisible by num_heads {num_heads}"

        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):

        B, C, H, W = x.shape
        N = H * W

        Q = self.q_depthwise(x)  # [B, C, H, W]
        K = self.k_multiscale(x)  # [B, C, H, W]
        V = self.v_directional(x)  # [B, C, H, W]

        Q = Q.view(B, C, N)  # [B, C, N]
        K = K.view(B, C, N)  # [B, C, N]
        V = V.view(B, C, N)  # [B, C, N]

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, V)

        out = out.transpose(1, 2).view(B, C, H, W)

        return self.out_proj(out) + x


class IRChannelAttention(nn.Module):

    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_grouped = ThreeGroupConvQ(channels)

        self.k_thermal_structure = ThermalStructureExtractor(channels)

        self.v_identity = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 1)
        )
        assert self.head_dim * num_heads == channels, f"channels {channels} must be divisible by num_heads {num_heads}"

        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):

        B, C, H, W = x.shape
        N = H * W

        Q = self.q_grouped(x)
        K = self.k_thermal_structure(x)
        V = self.v_identity(x)

        Q = Q.view(B, C, N)
        K = K.view(B, C, N)
        V = V.view(B, C, N)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, V)

        out = out.transpose(1, 2).view(B, C, H, W)

        return self.out_proj(out) + x


class MultiScaleStructureExtractor(nn.Module):

    def __init__(self, channels):
        super().__init__()

        c1 = channels // 3
        c2 = channels // 3
        c3 = channels - c1 - c2

        self.conv3 = nn.Conv2d(channels, c1, 1, padding=0)
        self.conv5 = nn.Conv2d(channels, c2, 3, padding=1)
        self.conv7 = nn.Conv2d(channels, c3, 5, padding=2)

        self.fusion = nn.Conv2d(channels, channels, 1)

    def forward(self, x):

        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)

        multi_scale = torch.cat([f3, f5, f7], dim=1)
        return self.fusion(multi_scale)


class DirectionalColorExtractor(nn.Module):

    def __init__(self, channels):
        super().__init__()

        c1 = channels // 4
        c2 = channels // 4
        c3 = channels // 4
        c4 = channels - c1 - c2 - c3

        self.conv_h = nn.Conv2d(channels, c1, kernel_size=(1, 3), padding=(0, 1))
        self.conv_v = nn.Conv2d(channels, c2, kernel_size=(3, 1), padding=(1, 0))
        self.conv_d1 = nn.Conv2d(channels, c3, kernel_size=3, padding=1)
        self.conv_d2 = nn.Conv2d(channels, c4, kernel_size=3, padding=1)

        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):

        fh = self.conv_h(x)
        fv = self.conv_v(x)
        fd1 = self.conv_d1(x)
        fd2 = self.conv_d2(x)

        directional = torch.cat([fh, fv, fd1, fd2], dim=1)
        return self.fusion(directional)

class ThreeGroupConvQ(nn.Module):

    def __init__(self, channels):
        super().__init__()

        c1 = channels // 3
        c2 = channels // 3
        c3 = channels - c1 - c2

        self.split_sizes = [c1, c2, c3]

        self.branch1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, groups=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=1, groups=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=1, groups=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1, x2, x3 = torch.split(x, self.split_sizes, dim=1)
        y1 = self.branch1(x1)
        y2 = self.branch2(x2)
        y3 = self.branch3(x3)
        return torch.cat([y1, y2, y3], dim=1)


class ThermalStructureExtractor(nn.Module):

    def __init__(self, channels):
        super().__init__()

        c_half = channels // 2
        c_remain = channels - c_half

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

        self.high_freq_proj = nn.Conv2d(channels, c_half, 1)
        self.low_freq_proj = nn.Conv2d(channels, c_remain, 1)

        self.fusion = nn.Conv2d(channels, channels, 1)

    def forward(self, x):

        pooled3 = self.pool3(x)
        pooled5 = self.pool5(x)

        low_freq = (pooled3 + pooled5) / 2
        low_freq = self.low_freq_proj(low_freq)

        high_freq = x - (pooled3 + pooled5) / 2
        high_freq = self.high_freq_proj(high_freq)

        thermal_structure = torch.cat([high_freq, low_freq], dim=1)
        return self.fusion(thermal_structure)

class DCE(nn.Module):

    def __init__(self, num_heads=8):
        super().__init__()

        self.num_heads = num_heads

        self.illumination_net = IlluminationAwareWeight()

        self.rgb_channel_attn = None
        self.ir_channel_attn = None

        self._initialized = False

    def _initialize_attention(self, channels):

        if self._initialized:
            return

        num_heads = self.num_heads
        if channels % num_heads != 0:
            num_heads = self._find_divisor(channels)

        self.rgb_channel_attn = RGBChannelAttention(channels, num_heads)
        self.ir_channel_attn = IRChannelAttention(channels, num_heads)
        self._initialized = True

    def _find_divisor(self, channels):

        for num_heads in [8, 4, 2, 1]:
            if channels % num_heads == 0:
                return num_heads
        return 1

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        if len(x) != 3:
            raise ValueError(f"DCE expects 3 inputs, but got {len(x)}.")

        rgb_mid, rgb_fuse, ir_fuse = x

        B, C, H, W = rgb_fuse.shape
        self._initialize_attention(C)

        rgb_weight, ir_weight = self.illumination_net(rgb_mid)

        enhanced_rgb = self.rgb_channel_attn(rgb_fuse)
        enhanced_ir = self.ir_channel_attn(ir_fuse)

        enhanced_rgb = rgb_fuse + enhanced_rgb
        enhanced_ir = ir_fuse + enhanced_ir

        rgb_weight = rgb_weight.view(B, 1, 1, 1).expand(B, C, H, W)
        ir_weight = ir_weight.view(B, 1, 1, 1).expand(B, C, H, W)

        weighted_rgb = enhanced_rgb * rgb_weight
        weighted_ir = enhanced_ir * ir_weight

        return torch.cat([weighted_rgb, weighted_ir], dim=1)

class IlluminationAwareWeight(nn.Module):

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb_mid_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = rgb_mid_feat.shape

        gray_feat = (
            0.299 * rgb_mid_feat[:, 0:1]
            + 0.587 * rgb_mid_feat[:, 1:2]
            + 0.114 * rgb_mid_feat[:, 2:3]
        )

        global_avg = F.adaptive_avg_pool2d(gray_feat, 1)
        global_max = F.adaptive_max_pool2d(gray_feat, 1)
        illumination = (global_avg + global_max) / 2

        rgb_weight = self.sigmoid(illumination).squeeze(-1).squeeze(-1)
        ir_weight = 1 - rgb_weight

        return rgb_weight, ir_weight