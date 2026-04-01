# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, OptMultiConfig
from mmengine.config import ConfigDict
from .dino import DINO


@MODELS.register_module()
class TwoStreamDINO(DINO):

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

        self.backbone_vis = MODELS.build(backbone)
        self.backbone_lwir = MODELS.build(backbone)


    def extract_vis_backbone_feat(self, img):
        x = self.backbone_vis(img)
        return x

    def extract_lwir_backbone_feat(self, img):
        x = self.backbone_lwir(img)
        return x

    def extract_feat(self, batch_inputs):
        if isinstance(batch_inputs, dict):
            img_vis = batch_inputs['img_vis']
            img_lwir = batch_inputs['img_lwir']
        else:
            raise ValueError("For two-stream detector, please provide both visible and infrared images in a dictionary")

        vis_x = self.extract_vis_backbone_feat(img_vis)
        lwir_x = self.extract_lwir_backbone_feat(img_lwir)

        x = []
        for i in range(len(vis_x)):
            x.append(0.5 * (vis_x[i] + lwir_x[i]))
        x = tuple(x)

        if self.with_neck:
            x = self.neck(x)

        return x

    def loss(self, batch_inputs: Dict,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples"""

        x = self.extract_feat(batch_inputs)


        head_inputs_dict = self.forward_transformer(x, batch_data_samples)


        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        return losses

    def predict(self,
                batch_inputs: Dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs"""

        x = self.extract_feat(batch_inputs)


        head_inputs_dict = self.forward_transformer(x, batch_data_samples)


        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self, batch_inputs: Dict,
                 batch_data_samples: SampleList = None) -> tuple:
        """Network forward process"""

        x = self.extract_feat(batch_inputs)


        head_inputs_dict = self.forward_transformer(x, batch_data_samples)


        results = self.bbox_head.forward(**head_inputs_dict)
        return results