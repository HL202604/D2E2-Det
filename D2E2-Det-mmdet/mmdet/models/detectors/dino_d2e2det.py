# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from mmengine.config import ConfigDict

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, OptMultiConfig

from ..layers import COD, DCE
from .dino import DINO


@MODELS.register_module()
class DINOD2E2DET(DINO):
    """DINO with D2E2DET multimodal backbone interaction."""

    def __init__(self,
                 backbone: ConfigDict,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 num_queries: int = 900,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 with_box_refine: bool = True,
                 as_two_stage: bool = True,
                 num_feature_levels: int = 4,
                 positional_encoding: OptConfigType = None,
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 dn_cfg: OptConfigType = None,
                 **kwargs) -> None:

        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            num_queries=num_queries,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            with_box_refine=with_box_refine,
            as_two_stage=as_two_stage,
            num_feature_levels=num_feature_levels,
            positional_encoding=positional_encoding,
            encoder=encoder,
            decoder=decoder,
            dn_cfg=dn_cfg,
            **kwargs)

        # dual backbones
        self.backbone_vis = MODELS.build(backbone)
        self.backbone_lwir = MODELS.build(backbone)

        # shared neck after D2E2DET fusion
        self.neck = MODELS.build(neck) if neck is not None else None

        # assume backbone out_indices=(1,2,3) => C3/C4/C5 = 512/1024/2048
        self.cod3 = COD(512, 512)
        self.cod4 = COD(1024, 1024)
        self.cod5 = COD(2048, 2048)

        self.dce3 = DCE(channels=512, num_heads=8)
        self.dce4 = DCE(channels=1024, num_heads=8)
        self.dce5 = DCE(channels=2048, num_heads=8)

    def extract_visfeat(self, img):
        return self.backbone_vis(img)

    def extract_lwirfeat(self, img):
        return self.backbone_lwir(img)

    def extract_feat(self, batch_inputs):
        if not isinstance(batch_inputs, dict):
            raise ValueError(
                'For DINOD2E2DET, batch_inputs must be a dict containing '
                '"img_vis" and "img_lwir".')

        img_vis = batch_inputs['img_vis']
        img_lwir = batch_inputs['img_lwir']

        vis_feats = self.extract_visfeat(img_vis)
        lwir_feats = self.extract_lwirfeat(img_lwir)

        # backbone out_indices should be (1, 2, 3)
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
        head_inputs_dict = self.forward_transformer(x, batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        x = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(x, batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self,
                 batch_inputs: Dict,
                 batch_data_samples: SampleList = None):
        x = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(x, batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results