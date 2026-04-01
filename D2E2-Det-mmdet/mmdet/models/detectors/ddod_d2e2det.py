import torch
import torch.nn as nn
from typing import Dict

from mmengine.config import ConfigDict
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, OptMultiConfig

from ..layers import COD, DCE
from .ddod import DDOD


@MODELS.register_module()
class DDODD2E2DET(DDOD):
    """DDOD with D2E2DET multimodal backbone interaction."""

    def __init__(self,
                 backbone: ConfigDict,
                 neck: ConfigDict,
                 bbox_head: ConfigDict,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # dual backbones
        self.backbone_vis = MODELS.build(backbone)
        self.backbone_lwir = MODELS.build(backbone)

        # shared neck after D2E2DET
        self.neck = MODELS.build(neck) if neck is not None else None

        # assume backbone out_indices=(1,2,3) => C3/C4/C5 = 512/1024/2048
        self.cod3 = COD(512, 512)
        self.cod4 = COD(1024, 1024)
        self.cod5 = COD(2048, 2048)

        self.dce3 = DCE(channels=512, num_heads=8)
        self.dce4 = DCE(channels=1024, num_heads=8)
        self.dce5 = DCE(channels=2048, num_heads=8)

    def extract_vis_backbone_feat(self, img):
        return self.backbone_vis(img)

    def extract_lwir_backbone_feat(self, img):
        return self.backbone_lwir(img)

    def extract_feat(self, batch_inputs):
        img_vis = batch_inputs['img_vis']
        img_lwir = batch_inputs['img_lwir']

        vis_feats = self.extract_vis_backbone_feat(img_vis)
        lwir_feats = self.extract_lwir_backbone_feat(img_lwir)

        # if backbone.out_indices = (1, 2, 3), outputs are C3/C4/C5
        c3_vis, c4_vis, c5_vis = vis_feats
        c3_ir, c4_ir, c5_ir = lwir_feats

        # COD
        c3_vis, c3_ir = self.cod3(c3_vis, c3_ir)
        c4_vis, c4_ir = self.cod4(c4_vis, c4_ir)
        c5_vis, c5_ir = self.cod5(c5_vis, c5_ir)

        # DCE
        p3 = self.dce3([img_vis, c3_vis, c3_ir])
        p4 = self.dce4([img_vis, c4_vis, c4_ir])
        p5 = self.dce5([img_vis, c5_vis, c5_ir])

        x = (p3, p4, p5)

        if self.with_neck:
            x = self.neck(x)

        return x

    def loss(self, batch_inputs: Dict,
             batch_data_samples: SampleList) -> dict:
        x = self.extract_feat(batch_inputs)
        return self.bbox_head.loss(x, batch_data_samples)

    def predict(self,
                batch_inputs: Dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self, batch_inputs: Dict,
                 batch_data_samples: SampleList = None):
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results