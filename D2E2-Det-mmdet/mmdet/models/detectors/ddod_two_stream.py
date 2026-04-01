import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.config import ConfigDict
from mmdet.utils import OptConfigType, OptMultiConfig
from .ddod import DDOD
from mmdet.structures import SampleList
from typing import Tuple, Dict, List


@MODELS.register_module()
class TwoStreamDDOD(DDOD):

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

        self.backbone_vis = MODELS.build(backbone)
        self.backbone_lwir = MODELS.build(backbone)


    def extract_vis_backbone_feat(self, img):
        x = self.backbone_vis(img)
        return x

    def extract_lwir_backbone_feat(self, img):
        x = self.backbone_lwir(img)
        return x

    def extract_feat(self, batch_inputs):
        img_vis = batch_inputs['img_vis']
        img_lwir = batch_inputs['img_lwir']

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
        """Calculate losses from a batch of inputs and data samples."""
        x = self.extract_feat(batch_inputs)
        return self.bbox_head.loss(x, batch_data_samples)

    def predict(self,
                batch_inputs: Dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self, batch_inputs: Dict,
                 batch_data_samples: SampleList = None) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results