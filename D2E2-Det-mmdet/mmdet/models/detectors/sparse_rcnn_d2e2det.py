import copy
import torch
import torch.nn as nn
from typing import Dict

from mmengine.config import ConfigDict
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, OptMultiConfig

from ..layers import COD, DCE
from .sparse_rcnn import SparseRCNN


@MODELS.register_module()
class SparseRCNND2E2DET(SparseRCNN):
    """Sparse R-CNN with D2E2DET multimodal backbone interaction."""

    def __init__(self,
                 backbone: ConfigDict,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)


        self.backbone_vis = MODELS.build(backbone)
        self.backbone_lwir = MODELS.build(backbone)


        self.neck = MODELS.build(neck) if neck is not None else None

        # ResNet50: C3/C4/C5 = 512/1024/2048
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
        img_vis = batch_inputs['img_vis']
        img_lwir = batch_inputs['img_lwir']

        vis_feats = self.extract_visfeat(img_vis)
        lwir_feats = self.extract_lwirfeat(img_lwir)


        c2_vis, c3_vis, c4_vis, c5_vis = vis_feats
        c2_ir, c3_ir, c4_ir, c5_ir = lwir_feats


        c3_vis, c3_ir = self.cod3(c3_vis, c3_ir)
        c4_vis, c4_ir = self.cod4(c4_vis, c4_ir)
        c5_vis, c5_ir = self.cod5(c5_vis, c5_ir)


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

        losses = dict()

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)

            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(
                    data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)

            for key in list(rpn_losses.keys()):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)

            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list, batch_data_samples)
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        x = self.extract_feat(batch_inputs)

        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self,
                 batch_inputs: Dict,
                 batch_data_samples: SampleList = None) -> tuple:
        x = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_outs = self.roi_head.forward(x, rpn_results_list, batch_data_samples)
        return (roi_outs,)