import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.config import ConfigDict
from mmdet.utils import OptConfigType, OptMultiConfig
from .sparse_rcnn import SparseRCNN
from mmdet.structures import SampleList
from typing import Tuple, Dict, List
import copy


@MODELS.register_module()
class TwoStreamSparseRCNN(SparseRCNN):

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
        """Calculate losses from a batch of inputs and data samples"""
        x = self.extract_feat(batch_inputs)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs"""
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

    def _forward(self, batch_inputs: Dict,
                 batch_data_samples: SampleList = None) -> tuple:
        """Network forward process"""
        x = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_outs = self.roi_head.forward(x, rpn_results_list,
                                         batch_data_samples)
        return (roi_outs,)