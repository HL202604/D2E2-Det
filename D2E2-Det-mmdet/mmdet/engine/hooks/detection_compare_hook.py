
# detection_compare_hook.py
# import os
# import os.path as osp
# import warnings
# from typing import Dict, Optional, Sequence
#
# import mmcv
# import numpy as np
# from mmengine.hooks import Hook
# from mmengine.runner import Runner
# from mmengine.fileio import get
# from mmengine.utils import mkdir_or_exist  # 使用 mmengine 的 mkdir_or_exist
#
# from mmdet.registry import HOOKS
# from mmdet.structures import DetDataSample
# from mmengine.visualization import Visualizer
#
#
# @HOOKS.register_module()
# class DetectionCompareHook(Hook):
#     """Detection Comparison Hook. Used to visualize comparison between
#     prediction and ground truth for correct detections, false positives
#     and false negatives.
#     """
#
#     def __init__(self,
#                  draw: bool = True,
#                  interval: int = 50,
#                  score_thr: float = 0.3,
#                  iou_thr: float = 0.4,
#                  show: bool = False,
#                  wait_time: float = 0.,
#                  test_out_dir: Optional[str] = None,
#                  backend_args: dict = None,
#                  colors: dict = None):
#         self._visualizer: Visualizer = Visualizer.get_current_instance()
#         self.interval = interval
#         self.score_thr = score_thr
#         self.iou_thr = iou_thr
#         self.show = show
#         self.wait_time = wait_time
#         self.backend_args = backend_args
#         self.draw = draw
#         self.test_out_dir = test_out_dir
#         self._test_index = 0
#
#         # Set colors for different detection types
#         self.colors = colors or {
#             'true_positive': (0, 255, 0),  # Green - correct detections
#             'false_positive': (255, 0, 0),  # Red - false positives
#             'false_negative': (0, 0, 255),  # Blue - false negatives
#         }
#
#         if self.show:
#             self._visualizer._vis_backends = {}
#             warnings.warn('The show is True, it means that only '
#                           'the prediction results are visualized '
#                           'without storing data, so vis_backends '
#                           'needs to be excluded.')
#
#     def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
#         """Calculate IoU between two boxes in [x1, y1, x2, y2] format."""
#         x11, y11, x12, y12 = box1
#         x21, y21, x22, y22 = box2
#
#         # Calculate intersection area
#         xi1 = max(x11, x21)
#         yi1 = max(y11, y21)
#         xi2 = min(x12, x22)
#         yi2 = min(y12, y22)
#
#         inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
#
#         # Calculate union area
#         box1_area = (x12 - x11) * (y12 - y11)
#         box2_area = (x22 - x21) * (y22 - y21)
#         union_area = box1_area + box2_area - inter_area
#
#         return inter_area / union_area if union_area > 0 else 0
#
#     def _match_detections(self, pred_instances, gt_instances):
#         """Match predictions with ground truth boxes."""
#         if len(pred_instances) == 0:
#             return [], [], list(range(len(gt_instances)))
#
#         if len(gt_instances) == 0:
#             return [], list(range(len(pred_instances))), []
#
#         pred_boxes = pred_instances.bboxes.cpu().numpy()
#         pred_scores = pred_instances.scores.cpu().numpy()
#         pred_labels = pred_instances.labels.cpu().numpy()
#
#         gt_boxes = gt_instances.bboxes.cpu().numpy()
#         gt_labels = gt_instances.labels.cpu().numpy()
#
#         # Filter predictions by score threshold
#         keep = pred_scores >= self.score_thr
#         pred_boxes = pred_boxes[keep]
#         pred_scores = pred_scores[keep]
#         pred_labels = pred_labels[keep]
#
#         true_positives = []
#         false_positives = []
#         matched_gt_indices = set()
#
#         # Match predictions with ground truth
#         for i, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
#             matched = False
#             best_iou = 0
#             best_gt_idx = -1
#
#             for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
#                 if j in matched_gt_indices:
#                     continue
#
#                 iou = self._calculate_iou(pred_box, gt_box)
#                 if iou > best_iou and iou >= self.iou_thr and pred_label == gt_label:
#                     best_iou = iou
#                     best_gt_idx = j
#                     matched = True
#
#             if matched:
#                 true_positives.append({
#                     'bbox': pred_box,
#                     'label': pred_label,
#                     'score': pred_score,
#                     'matched_gt_idx': best_gt_idx
#                 })
#                 matched_gt_indices.add(best_gt_idx)
#             else:
#                 false_positives.append({
#                     'bbox': pred_box,
#                     'label': pred_label,
#                     'score': pred_score
#                 })
#
#         # Find false negatives (unmatched ground truth)
#         false_negatives = []
#         for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
#             if j not in matched_gt_indices:
#                 false_negatives.append({
#                     'bbox': gt_box,
#                     'label': gt_label
#                 })
#
#         return true_positives, false_positives, false_negatives
#
#     def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: Dict,
#                        outputs: Sequence[DetDataSample]) -> None:
#         """Run after every ``self.interval`` validation iterations."""
#         if not self.draw:
#             return
#
#         total_curr_iter = runner.iter + batch_idx
#
#         if total_curr_iter % self.interval == 0:
#             print(f"🚀 DetectionCompareHook.after_val_iter called! Batch: {batch_idx}")
#             self._visualize_comparison(outputs[0], runner, 'val')
#
#     def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: Dict,
#                         outputs: Sequence[DetDataSample]) -> None:
#         """Run after every testing iterations."""
#         print(f"🚀 DetectionCompareHook.after_test_iter called! Batch: {batch_idx}")
#         print(f"📊 Number of outputs: {len(outputs)}")
#
#         if not self.draw:
#             print("❌ draw is False, skipping...")
#             return
#         else:
#             print("✅ draw is True, proceeding...")
#
#         if self.test_out_dir is not None:
#             self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
#                                          self.test_out_dir)
#             mkdir_or_exist(self.test_out_dir)  # 使用 mmengine 的 mkdir_or_exist
#             print(f"📁 Output directory: {self.test_out_dir}")
#
#         for i, data_sample in enumerate(outputs):
#             self._test_index += 1
#             print(f"🖼️ Processing image {self._test_index}: {data_sample.img_path}")
#             if hasattr(data_sample, 'pred_instances') and data_sample.pred_instances is not None:
#                 print(f"📦 Pred instances: {len(data_sample.pred_instances)}")
#             else:
#                 print("📦 No pred instances")
#
#             if hasattr(data_sample, 'gt_instances') and data_sample.gt_instances is not None:
#                 print(f"🏷️ GT instances: {len(data_sample.gt_instances)}")
#             else:
#                 print("🏷️ No GT instances")
#
#             self._visualize_comparison(data_sample, runner, 'test')
#
#     def _visualize_comparison(self, data_sample: DetDataSample, runner: Runner, mode: str):
#         """Visualize comparison between predictions and ground truth."""
#         print(f"🎨 Starting visualization for {data_sample.img_path}")
#
#         img_path = data_sample.img_path
#         try:
#             img_bytes = get(img_path, backend_args=self.backend_args)
#             img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
#         except Exception as e:
#             print(f"❌ Failed to load image {img_path}: {e}")
#             return
#
#         # 详细检查数据样本的结构
#         print(f"📋 DataSample keys: {list(data_sample.__dict__.keys())}")
#
#         pred_instances = data_sample.pred_instances if hasattr(data_sample, 'pred_instances') else None
#         gt_instances = data_sample.gt_instances if hasattr(data_sample, 'gt_instances') else None
#
#         print(f"🔍 pred_instances exists: {pred_instances is not None}")
#         print(f"🔍 gt_instances exists: {gt_instances is not None}")
#
#         if pred_instances is not None:
#             print(f"📦 Pred instances attributes: {list(pred_instances.__dict__.keys())}")
#             if hasattr(pred_instances, 'bboxes') and pred_instances.bboxes is not None:
#                 print(f"📦 Pred bboxes shape: {pred_instances.bboxes.shape}")
#                 print(
#                     f"📦 Pred labels: {pred_instances.labels.cpu().numpy() if pred_instances.labels is not None else 'None'}")
#                 print(
#                     f"📦 Pred scores: {pred_instances.scores.cpu().numpy() if hasattr(pred_instances, 'scores') and pred_instances.scores is not None else 'None'}")
#             else:
#                 print("📦 No pred bboxes")
#
#         if gt_instances is not None:
#             print(f"🏷️ GT instances attributes: {list(gt_instances.__dict__.keys())}")
#             if hasattr(gt_instances, 'bboxes') and gt_instances.bboxes is not None:
#                 print(f"🏷️ GT bboxes shape: {gt_instances.bboxes.shape}")
#                 print(
#                     f"🏷️ GT labels: {gt_instances.labels.cpu().numpy() if gt_instances.labels is not None else 'None'}")
#             else:
#                 print("🏷️ No GT bboxes")
#
#         # 如果 GT 为空，尝试从其他位置获取
#         if gt_instances is None or (hasattr(gt_instances, 'bboxes') and gt_instances.bboxes is None):
#             print("🔄 Trying to find GT instances from other attributes...")
#             if hasattr(data_sample, 'gt_instances_ignore'):
#                 gt_instances = data_sample.gt_instances_ignore
#                 print(f"🔍 Found gt_instances_ignore: {gt_instances is not None}")
#             elif hasattr(data_sample, '_gt_instances'):
#                 gt_instances = data_sample._gt_instances
#                 print(f"🔍 Found _gt_instances: {gt_instances is not None}")
#
#         # 如果还是没有 GT，说明测试时确实没有加载真实标签
#         if gt_instances is None or (
#                 hasattr(gt_instances, 'bboxes') and (gt_instances.bboxes is None or len(gt_instances.bboxes) == 0)):
#             print("⚠️  Warning: No ground truth instances found!")
#             print("💡  This is normal for pure inference, but for comparison visualization,")
#             print("💡  you need to ensure test dataset includes annotations.")
#
#             # 只显示预测结果（全部显示为红色，因为没有GT对比）
#             visualizer = Visualizer()
#             visualizer.set_image(img)
#
#             if pred_instances is not None and hasattr(pred_instances,
#                                                       'bboxes') and pred_instances.bboxes is not None and len(
#                     pred_instances.bboxes) > 0:
#                 pred_boxes = pred_instances.bboxes.cpu().numpy()
#                 for bbox in pred_boxes:
#                     visualizer.draw_bboxes(
#                         bbox[None, :],
#                         edge_colors=(255, 0, 0),  # 全部显示为红色（只有预测）
#                         line_widths=3
#                     )
#                 print(f"📸 Only showing {len(pred_boxes)} predictions in RED")
#
#             drawn_img = visualizer.get_image()
#
#             # 保存图片
#             out_file = None
#             if self.test_out_dir is not None and mode == 'test':
#                 out_file = osp.basename(img_path)
#                 out_file = osp.join(self.test_out_dir, out_file)
#                 mkdir_or_exist(osp.dirname(out_file))
#                 mmcv.imwrite(drawn_img[..., ::-1], out_file)
#                 print(f"💾 Saved to: {out_file}")
#
#             return
#
#         # 正常情况：有 GT 和预测，进行匹配和对比可视化
#         print("✅ Both predictions and ground truth found, performing matching...")
#
#         true_positives, false_positives, false_negatives = self._match_detections(pred_instances, gt_instances)
#
#         print(f"📊 匹配结果 - TP: {len(true_positives)}, FP: {len(false_positives)}, FN: {len(false_negatives)}")
#
#         # 创建可视化器
#         visualizer = Visualizer()
#         visualizer.set_image(img)
#
#         # 绘制正确检测（绿色）
#         for tp in true_positives:
#             visualizer.draw_bboxes(
#                 tp['bbox'][None, :],
#                 edge_colors=self.colors['true_positive'],
#                 line_widths=3
#             )
#
#         # 绘制误检（红色）
#         for fp in false_positives:
#             visualizer.draw_bboxes(
#                 fp['bbox'][None, :],
#                 edge_colors=self.colors['false_positive'],
#                 line_widths=2
#             )
#
#         # 绘制漏检（蓝色）
#         for fn in false_negatives:
#             visualizer.draw_bboxes(
#                 fn['bbox'][None, :],
#                 edge_colors=self.colors['false_negative'],
#                 line_widths=2
#             )
#
#
#         # 获取绘制的图像
#         drawn_img = visualizer.get_image()
#
#         # 保存图片
#         out_file = None
#         if self.test_out_dir is not None and mode == 'test':
#             out_file = osp.basename(img_path)
#             out_file = osp.join(self.test_out_dir, out_file)
#             mkdir_or_exist(osp.dirname(out_file))
#             mmcv.imwrite(drawn_img[..., ::-1], out_file)
#             print(f"💾 Saved comparison to: {out_file}")
#
#         if self.show:
#             visualizer.show(drawn_img, wait_time=self.wait_time)




# detection_compare_hook.py
# import os
# import os.path as osp
# import warnings
# from typing import Dict, Optional, Sequence
#
# import mmcv
# import numpy as np
# import cv2
# from mmengine.hooks import Hook
# from mmengine.runner import Runner
# from mmengine.fileio import get
# from mmengine.utils import mkdir_or_exist
#
# from mmdet.registry import HOOKS
# from mmdet.structures import DetDataSample
# from mmengine.visualization import Visualizer
#
#
# @HOOKS.register_module()
# class DetectionCompareHook(Hook):
#     """Detection Comparison Hook. Used to visualize comparison between
#     prediction and ground truth for correct detections, false positives
#     and false negatives.
#
#     Modified to support multimodal (RGB + Thermal) visualization.
#     """
#
#     def __init__(self,
#                  draw: bool = True,
#                  interval: int = 50,
#                  score_thr: float = 0.3,
#                  iou_thr: float = 0.4,
#                  show: bool = False,
#                  wait_time: float = 0.,
#                  test_out_dir: Optional[str] = None,
#                  backend_args: dict = None,
#                  colors: dict = None,
#                  visualize_both_modalities: bool = True,  # 新增：是否可视化两种模态
#                  thermal_suffix: str = '_lwir'):  # 新增：红外图像后缀
#         self._visualizer: Visualizer = Visualizer.get_current_instance()
#         self.interval = interval
#         self.score_thr = score_thr
#         self.iou_thr = iou_thr
#         self.show = show
#         self.wait_time = wait_time
#         self.backend_args = backend_args
#         self.draw = draw
#         self.test_out_dir = test_out_dir
#         self._test_index = 0
#         self.visualize_both_modalities = visualize_both_modalities
#         self.thermal_suffix = thermal_suffix
#
#         # Set colors for different detection types
#         self.colors = colors or {
#             'true_positive': (0, 255, 0),  # Green - correct detections
#             'false_positive': (255, 0, 0),  # Red - false positives
#             'false_negative': (0, 0, 255),  # Blue - false negatives
#         }
#
#         if self.show:
#             self._visualizer._vis_backends = {}
#             warnings.warn('The show is True, it means that only '
#                           'the prediction results are visualized '
#                           'without storing data, so vis_backends '
#                           'needs to be excluded.')
#
#     def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
#         """Calculate IoU between two boxes in [x1, y1, x2, y2] format."""
#         x11, y11, x12, y12 = box1
#         x21, y21, x22, y22 = box2
#
#         # Calculate intersection area
#         xi1 = max(x11, x21)
#         yi1 = max(y11, y21)
#         xi2 = min(x12, x22)
#         yi2 = min(y12, y22)
#
#         inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
#
#         # Calculate union area
#         box1_area = (x12 - x11) * (y12 - y11)
#         box2_area = (x22 - x21) * (y22 - y21)
#         union_area = box1_area + box2_area - inter_area
#
#         return inter_area / union_area if union_area > 0 else 0
#
#     def _match_detections(self, pred_instances, gt_instances):
#         """Match predictions with ground truth boxes."""
#         if len(pred_instances) == 0:
#             return [], [], list(range(len(gt_instances)))
#
#         if len(gt_instances) == 0:
#             return [], list(range(len(pred_instances))), []
#
#         pred_boxes = pred_instances.bboxes.cpu().numpy()
#         pred_scores = pred_instances.scores.cpu().numpy()
#         pred_labels = pred_instances.labels.cpu().numpy()
#
#         gt_boxes = gt_instances.bboxes.cpu().numpy()
#         gt_labels = gt_instances.labels.cpu().numpy()
#
#         # Filter predictions by score threshold
#         keep = pred_scores >= self.score_thr
#         pred_boxes = pred_boxes[keep]
#         pred_scores = pred_scores[keep]
#         pred_labels = pred_labels[keep]
#
#         true_positives = []
#         false_positives = []
#         matched_gt_indices = set()
#
#         # Match predictions with ground truth
#         for i, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
#             matched = False
#             best_iou = 0
#             best_gt_idx = -1
#
#             for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
#                 if j in matched_gt_indices:
#                     continue
#
#                 iou = self._calculate_iou(pred_box, gt_box)
#                 if iou > best_iou and iou >= self.iou_thr and pred_label == gt_label:
#                     best_iou = iou
#                     best_gt_idx = j
#                     matched = True
#
#             if matched:
#                 true_positives.append({
#                     'bbox': pred_box,
#                     'label': pred_label,
#                     'score': pred_score,
#                     'matched_gt_idx': best_gt_idx
#                 })
#                 matched_gt_indices.add(best_gt_idx)
#             else:
#                 false_positives.append({
#                     'bbox': pred_box,
#                     'label': pred_label,
#                     'score': pred_score
#                 })
#
#         # Find false negatives (unmatched ground truth)
#         false_negatives = []
#         for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
#             if j not in matched_gt_indices:
#                 false_negatives.append({
#                     'bbox': gt_box,
#                     'label': gt_label
#                 })
#
#         return true_positives, false_positives, false_negatives
#
#     def _get_thermal_image_path(self, rgb_path: str) -> str:
#         """Get corresponding thermal image path from RGB image path."""
#         # 获取目录和文件名
#         rgb_dir = osp.dirname(rgb_path)
#         rgb_filename = osp.basename(rgb_path)
#
#         # 解析RGB文件名
#         rgb_name_without_ext = osp.splitext(rgb_filename)[0]
#         rgb_ext = osp.splitext(rgb_filename)[1]
#
#         # 构建红外图像文件名（替换后缀）
#         # 如果已经有_lwir后缀，保持原样；否则添加_lwir
#         if self.thermal_suffix in rgb_name_without_ext:
#             thermal_name_without_ext = rgb_name_without_ext
#         else:
#             thermal_name_without_ext = f"{rgb_name_without_ext}{self.thermal_suffix}"
#
#         # 保持相同的扩展名或者使用.png
#         thermal_ext = '.png' if rgb_ext.lower() == '.jpg' or rgb_ext.lower() == '.jpeg' else rgb_ext
#         thermal_filename = f"{thermal_name_without_ext}{thermal_ext}"
#
#         # 构建完整路径
#         thermal_path = osp.join(rgb_dir, thermal_filename)
#
#         # 如果不在同一目录，尝试images_lwir目录
#         if not osp.exists(thermal_path):
#             thermal_dir = rgb_dir.replace('images', 'images_lwir')
#             thermal_path = osp.join(thermal_dir, thermal_filename)
#
#         return thermal_path
#
#     def _load_image(self, img_path: str):
#         """Load image with error handling."""
#         try:
#             img_bytes = get(img_path, backend_args=self.backend_args)
#             img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
#             return img
#         except Exception as e:
#             print(f"❌ Failed to load image {img_path}: {e}")
#             return None
#
#     def _draw_detections_on_image(self, img, true_positives, false_positives, false_negatives):
#         """Draw detections on an image."""
#         visualizer = Visualizer()
#         visualizer.set_image(img)
#
#         # 绘制正确检测（绿色）
#         for tp in true_positives:
#             try:
#                 bbox = tp['bbox']
#                 if isinstance(bbox, np.ndarray) and len(bbox.shape) == 1:
#                     bbox = bbox[None, :]
#                 visualizer.draw_bboxes(
#                     bbox,
#                     edge_colors=self.colors['true_positive'],
#                     line_widths=3
#                 )
#             except Exception as e:
#                 print(f"⚠️  Error drawing TP: {e}")
#                 continue
#
#         # 绘制误检（红色）
#         for fp in false_positives:
#             try:
#                 bbox = fp['bbox']
#                 if isinstance(bbox, np.ndarray) and len(bbox.shape) == 1:
#                     bbox = bbox[None, :]
#                 visualizer.draw_bboxes(
#                     bbox,
#                     edge_colors=self.colors['false_positive'],
#                     line_widths=2
#                 )
#             except Exception as e:
#                 print(f"⚠️  Error drawing FP: {e}")
#                 continue
#
#         # 绘制漏检（蓝色）
#         for fn in false_negatives:
#             try:
#                 bbox = fn['bbox']
#                 if isinstance(bbox, np.ndarray) and len(bbox.shape) == 1:
#                     bbox = bbox[None, :]
#                 visualizer.draw_bboxes(
#                     bbox,
#                     edge_colors=self.colors['false_negative'],
#                     line_widths=2
#                 )
#             except Exception as e:
#                 print(f"⚠️  Error drawing FN: {e}")
#                 continue
#
#         return visualizer.get_image()
#
#     def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: Dict,
#                        outputs: Sequence[DetDataSample]) -> None:
#         """Run after every ``self.interval`` validation iterations."""
#         if not self.draw:
#             return
#
#         total_curr_iter = runner.iter + batch_idx
#
#         if total_curr_iter % self.interval == 0:
#             print(f"🚀 DetectionCompareHook.after_val_iter called! Batch: {batch_idx}")
#             self._visualize_comparison(outputs[0], runner, 'val')
#
#     def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: Dict,
#                         outputs: Sequence[DetDataSample]) -> None:
#         """Run after every testing iterations."""
#         print(f"🚀 DetectionCompareHook.after_test_iter called! Batch: {batch_idx}")
#         print(f"📊 Number of outputs: {len(outputs)}")
#
#         if not self.draw:
#             print("❌ draw is False, skipping...")
#             return
#         else:
#             print("✅ draw is True, proceeding...")
#
#         if self.test_out_dir is not None:
#             self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
#                                          self.test_out_dir)
#             mkdir_or_exist(self.test_out_dir)
#             print(f"📁 Output directory: {self.test_out_dir}")
#
#         for i, data_sample in enumerate(outputs):
#             self._test_index += 1
#             print(f"🖼️ Processing image {self._test_index}: {data_sample.img_path}")
#             if hasattr(data_sample, 'pred_instances') and data_sample.pred_instances is not None:
#                 print(f"📦 Pred instances: {len(data_sample.pred_instances)}")
#             else:
#                 print("📦 No pred instances")
#
#             if hasattr(data_sample, 'gt_instances') and data_sample.gt_instances is not None:
#                 print(f"🏷️ GT instances: {len(data_sample.gt_instances)}")
#             else:
#                 print("🏷️ No GT instances")
#
#             self._visualize_comparison(data_sample, runner, 'test')
#
#     def _visualize_comparison(self, data_sample: DetDataSample, runner: Runner, mode: str):
#         """Visualize comparison between predictions and ground truth."""
#         print(f"🎨 Starting visualization for {data_sample.img_path}")
#
#         rgb_path = data_sample.img_path
#
#         # 获取红外图像路径
#         thermal_path = self._get_thermal_image_path(rgb_path)
#         print(f"📷 RGB path: {rgb_path}")
#         print(f"🌡️ Thermal path: {thermal_path}")
#
#         # 加载RGB图像
#         rgb_img = self._load_image(rgb_path)
#         if rgb_img is None:
#             return
#
#         # 加载红外图像
#         thermal_img = None
#         if self.visualize_both_modalities and osp.exists(thermal_path):
#             thermal_img = self._load_image(thermal_path)
#             if thermal_img is None:
#                 print(f"⚠️ Could not load thermal image, will only visualize RGB")
#
#         # 详细检查数据样本的结构
#         print(f"📋 DataSample keys: {list(data_sample.__dict__.keys())}")
#
#         pred_instances = data_sample.pred_instances if hasattr(data_sample, 'pred_instances') else None
#         gt_instances = data_sample.gt_instances if hasattr(data_sample, 'gt_instances') else None
#
#         print(f"🔍 pred_instances exists: {pred_instances is not None}")
#         print(f"🔍 gt_instances exists: {gt_instances is not None}")
#
#         if pred_instances is not None:
#             print(f"📦 Pred instances attributes: {list(pred_instances.__dict__.keys())}")
#             if hasattr(pred_instances, 'bboxes') and pred_instances.bboxes is not None:
#                 print(f"📦 Pred bboxes shape: {pred_instances.bboxes.shape}")
#                 print(
#                     f"📦 Pred labels: {pred_instances.labels.cpu().numpy() if pred_instances.labels is not None else 'None'}")
#                 print(
#                     f"📦 Pred scores: {pred_instances.scores.cpu().numpy() if hasattr(pred_instances, 'scores') and pred_instances.scores is not None else 'None'}")
#             else:
#                 print("📦 No pred bboxes")
#
#         if gt_instances is not None:
#             print(f"🏷️ GT instances attributes: {list(gt_instances.__dict__.keys())}")
#             if hasattr(gt_instances, 'bboxes') and gt_instances.bboxes is not None:
#                 print(f"🏷️ GT bboxes shape: {gt_instances.bboxes.shape}")
#                 print(
#                     f"🏷️ GT labels: {gt_instances.labels.cpu().numpy() if gt_instances.labels is not None else 'None'}")
#             else:
#                 print("🏷️ No GT bboxes")
#
#         # 如果 GT 为空，尝试从其他位置获取
#         if gt_instances is None or (hasattr(gt_instances, 'bboxes') and gt_instances.bboxes is None):
#             print("🔄 Trying to find GT instances from other attributes...")
#             if hasattr(data_sample, 'gt_instances_ignore'):
#                 gt_instances = data_sample.gt_instances_ignore
#                 print(f"🔍 Found gt_instances_ignore: {gt_instances is not None}")
#             elif hasattr(data_sample, '_gt_instances'):
#                 gt_instances = data_sample._gt_instances
#                 print(f"🔍 Found _gt_instances: {gt_instances is not None}")
#
#         # 如果还是没有 GT，说明测试时确实没有加载真实标签
#         if gt_instances is None or (
#                 hasattr(gt_instances, 'bboxes') and (gt_instances.bboxes is None or len(gt_instances.bboxes) == 0)):
#             print("⚠️  Warning: No ground truth instances found!")
#             print("💡  This is normal for pure inference, but for comparison visualization,")
#             print("💡  you need to ensure test dataset includes annotations.")
#
#             # 只显示预测结果（全部显示为红色，因为没有GT对比）
#             visualizer = Visualizer()
#             visualizer.set_image(rgb_img)
#
#             if pred_instances is not None and hasattr(pred_instances,
#                                                       'bboxes') and pred_instances.bboxes is not None and len(
#                 pred_instances.bboxes) > 0:
#                 pred_boxes = pred_instances.bboxes.cpu().numpy()
#                 for bbox in pred_boxes:
#                     try:
#                         if len(bbox.shape) == 1:
#                             bbox = bbox[None, :]
#                         visualizer.draw_bboxes(
#                             bbox,
#                             edge_colors=(255, 0, 0),  # 全部显示为红色（只有预测）
#                             line_widths=3
#                         )
#                     except Exception as e:
#                         print(f"⚠️ Error drawing bbox: {e}")
#                         continue
#                 print(f"📸 Only showing {len(pred_boxes)} predictions in RED")
#
#             drawn_img = visualizer.get_image()
#
#             # 保存RGB图片
#             out_file = None
#             if self.test_out_dir is not None and mode == 'test':
#                 rgb_out_file = osp.basename(rgb_path)
#                 rgb_out_file = osp.join(self.test_out_dir, rgb_out_file)
#                 mkdir_or_exist(osp.dirname(rgb_out_file))
#                 mmcv.imwrite(drawn_img[..., ::-1], rgb_out_file)
#                 print(f"💾 Saved RGB to: {rgb_out_file}")
#
#                 # 如果存在红外图像，也保存红外图像
#                 if thermal_img is not None:
#                     thermal_visualizer = Visualizer()
#                     thermal_visualizer.set_image(thermal_img)
#
#                     if pred_instances is not None and hasattr(pred_instances,
#                                                               'bboxes') and pred_instances.bboxes is not None and len(
#                         pred_instances.bboxes) > 0:
#                         pred_boxes = pred_instances.bboxes.cpu().numpy()
#                         for bbox in pred_boxes:
#                             try:
#                                 if len(bbox.shape) == 1:
#                                     bbox = bbox[None, :]
#                                 thermal_visualizer.draw_bboxes(
#                                     bbox,
#                                     edge_colors=(255, 0, 0),
#                                     line_widths=3
#                                 )
#                             except Exception as e:
#                                 print(f"⚠️ Error drawing bbox on thermal: {e}")
#                                 continue
#
#                     thermal_drawn_img = thermal_visualizer.get_image()
#                     thermal_out_file = osp.basename(thermal_path)
#                     thermal_out_file = osp.join(self.test_out_dir, thermal_out_file)
#                     mkdir_or_exist(osp.dirname(thermal_out_file))
#                     mmcv.imwrite(thermal_drawn_img[..., ::-1], thermal_out_file)
#                     print(f"💾 Saved Thermal to: {thermal_out_file}")
#
#             if self.show:
#                 visualizer.show(drawn_img, wait_time=self.wait_time)
#
#             return
#
#         # 正常情况：有 GT 和预测，进行匹配和对比可视化
#         print("✅ Both predictions and ground truth found, performing matching...")
#
#         true_positives, false_positives, false_negatives = self._match_detections(pred_instances, gt_instances)
#
#         print(f"📊 Matching result - TP: {len(true_positives)}, FP: {len(false_positives)}, FN: {len(false_negatives)}")
#
#         # 绘制RGB图像
#         rgb_drawn_img = self._draw_detections_on_image(rgb_img, true_positives, false_positives, false_negatives)
#
#         # 保存RGB图片
#         if self.test_out_dir is not None and mode == 'test':
#             rgb_out_file = osp.basename(rgb_path)
#             rgb_out_file = osp.join(self.test_out_dir, rgb_out_file)
#             mkdir_or_exist(osp.dirname(rgb_out_file))
#             mmcv.imwrite(rgb_drawn_img[..., ::-1], rgb_out_file)
#             print(f"💾 Saved RGB comparison to: {rgb_out_file}")
#
#         # 绘制红外图像（如果存在）
#         if thermal_img is not None:
#             thermal_drawn_img = self._draw_detections_on_image(thermal_img, true_positives, false_positives,
#                                                                false_negatives)
#
#             # 保存红外图片
#             if self.test_out_dir is not None and mode == 'test':
#                 thermal_out_file = osp.basename(thermal_path)
#                 thermal_out_file = osp.join(self.test_out_dir, thermal_out_file)
#                 mkdir_or_exist(osp.dirname(thermal_out_file))
#                 mmcv.imwrite(thermal_drawn_img[..., ::-1], thermal_out_file)
#                 print(f"💾 Saved Thermal comparison to: {thermal_out_file}")
#
#         if self.show:
#             visualizer = Visualizer()
#             visualizer.show(rgb_drawn_img, wait_time=self.wait_time)

import os
import os.path as osp
import warnings
from typing import Dict, Optional, Sequence

import mmcv
import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.fileio import get
from mmengine.utils import mkdir_or_exist
from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample
from mmengine.visualization import Visualizer
from mmdet.apis import inference_detector


@HOOKS.register_module()
class DetectionCompareHook(Hook):
    """Detection Comparison Hook. Used to visualize comparison between
    prediction and ground truth for correct detections, false positives
    and false negatives.

    Supports separate inference and different thresholds for RGB and IR images.
    """

    def __init__(self,
                 draw: bool = True,
                 interval: int = 1,
                 rgb_score_thr: float = 0.3,  # RGB专用阈值
                 ir_score_thr: float = 0.01,  # IR专用阈值（较低）
                 rgb_iou_thr: float = 0.5,  # RGB专用IoU阈值
                 ir_iou_thr: float = 0.1,  # IR专用IoU阈值（较低）
                 show: bool = False,
                 wait_time: float = 0.,
                 test_out_dir: Optional[str] = None,
                 backend_args: dict = None,
                 colors: dict = None,
                 thermal_suffix: str = '_lwir',
                 infer_ir_separately: bool = True):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.rgb_score_thr = rgb_score_thr
        self.ir_score_thr = ir_score_thr
        self.rgb_iou_thr = rgb_iou_thr
        self.ir_iou_thr = ir_iou_thr
        self.show = show
        self.wait_time = wait_time
        self.backend_args = backend_args
        self.draw = draw
        self.test_out_dir = test_out_dir
        self._test_index = 0
        self.thermal_suffix = thermal_suffix
        self.infer_ir_separately = infer_ir_separately
        self.model = None

        # Set colors for different detection types
        self.colors = colors or {
            'true_positive': (0, 255, 0),  # Green - correct detections
            'false_positive': (255, 0, 0),  # Red - false positives
            'false_negative': (0, 0, 255),  # Blue - false negatives
        }

        # IR专用颜色（与RGB区分）
        self.ir_colors = {
            'true_positive': (0, 255, 0),
            'false_positive': (255, 0, 0),
            'false_negative': (0, 0, 255),
        }

        if self.show:
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

    def before_test(self, runner: Runner):
        """在测试开始前获取模型引用"""
        self.model = runner.model
        print(f"✅ Hook获取到模型引用")
        print(f"📊 阈值设置:")
        print(f"  - RGB: score_thr={self.rgb_score_thr}, iou_thr={self.rgb_iou_thr}")
        print(f"  - IR: score_thr={self.ir_score_thr}, iou_thr={self.ir_iou_thr}")

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes in [x1, y1, x2, y2] format."""
        x11, y11, x12, y12 = box1
        x21, y21, x22, y22 = box2

        # Calculate intersection area
        xi1 = max(x11, x21)
        yi1 = max(y11, y21)
        xi2 = min(x12, x22)
        yi2 = min(y12, y22)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Calculate union area
        box1_area = (x12 - x11) * (y12 - y11)
        box2_area = (x22 - x21) * (y22 - y21)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _match_detections_rgb(self, pred_instances, gt_instances):
        """Match predictions with ground truth boxes for RGB images (原始逻辑)."""
        if len(pred_instances) == 0:
            return [], [], list(range(len(gt_instances)))

        if len(gt_instances) == 0:
            return [], list(range(len(pred_instances))), []

        pred_boxes = pred_instances.bboxes.cpu().numpy()
        pred_scores = pred_instances.scores.cpu().numpy()
        pred_labels = pred_instances.labels.cpu().numpy()

        gt_boxes = gt_instances.bboxes.cpu().numpy()
        gt_labels = gt_instances.labels.cpu().numpy()

        # Filter predictions by score threshold (RGB使用较高阈值)
        keep = pred_scores >= self.rgb_score_thr
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        true_positives = []
        false_positives = []
        matched_gt_indices = set()

        # Match predictions with ground truth
        for i, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
            matched = False
            best_iou = 0
            best_gt_idx = -1

            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if j in matched_gt_indices:
                    continue

                iou = self._calculate_iou(pred_box, gt_box)
                if iou > best_iou and iou >= self.rgb_iou_thr and pred_label == gt_label:
                    best_iou = iou
                    best_gt_idx = j
                    matched = True

            if matched:
                true_positives.append({
                    'bbox': pred_box,
                    'label': pred_label,
                    'score': pred_score,
                    'matched_gt_idx': best_gt_idx
                })
                matched_gt_indices.add(best_gt_idx)
            else:
                false_positives.append({
                    'bbox': pred_box,
                    'label': pred_label,
                    'score': pred_score
                })

        # Find false negatives (unmatched ground truth)
        false_negatives = []
        for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if j not in matched_gt_indices:
                false_negatives.append({
                    'bbox': gt_box,
                    'label': gt_label
                })

        return true_positives, false_positives, false_negatives

    def _match_detections_ir(self, pred_instances, gt_instances):
        """Match predictions with ground truth boxes for IR images (宽松逻辑).
        限制误检显示数量，最多显示得分最高的3个红框。
        """
        if pred_instances is None or len(pred_instances) == 0:
            if gt_instances is None or len(gt_instances) == 0:
                return [], [], []
            else:
                # All GT are false negatives
                false_negatives = []
                gt_boxes = gt_instances.bboxes.cpu().numpy()
                gt_labels = gt_instances.labels.cpu().numpy() if hasattr(gt_instances, 'labels') else np.zeros(
                    len(gt_boxes))
                for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    false_negatives.append({
                        'bbox': gt_box,
                        'label': gt_label
                    })
                return [], [], false_negatives

        if gt_instances is None or len(gt_instances) == 0:
            # All predictions are false positives
            false_positives = []
            pred_boxes = pred_instances.bboxes.cpu().numpy()
            pred_scores = pred_instances.scores.cpu().numpy() if hasattr(pred_instances, 'scores') else np.ones(
                len(pred_boxes))
            pred_labels = pred_instances.labels.cpu().numpy() if hasattr(pred_instances, 'labels') else np.zeros(
                len(pred_boxes))

            # 按得分排序，只保留得分最高的3个
            indices = list(range(len(pred_boxes)))
            if len(indices) > 0:
                # 按得分降序排序
                if hasattr(pred_instances, 'scores'):
                    scores = pred_scores
                    indices = sorted(indices, key=lambda i: scores[i], reverse=True)

                # 只取前3个
                indices = indices[:3]

            for i in indices:
                pred_box = pred_boxes[i]
                pred_label = pred_labels[i]
                pred_score = pred_scores[i]
                # 使用IR的较低阈值
                if pred_score >= self.ir_score_thr:
                    false_positives.append({
                        'bbox': pred_box,
                        'label': pred_label,
                        'score': pred_score
                    })
            return [], false_positives, []

        pred_boxes = pred_instances.bboxes.cpu().numpy()
        pred_scores = pred_instances.scores.cpu().numpy()
        pred_labels = pred_instances.labels.cpu().numpy()

        gt_boxes = gt_instances.bboxes.cpu().numpy()
        gt_labels = gt_instances.labels.cpu().numpy()

        # Filter predictions by IR score threshold (使用较低阈值)
        keep = pred_scores >= self.ir_score_thr
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        true_positives = []
        false_positives = []
        matched_gt_indices = set()

        # 先收集所有未匹配的预测，然后按得分排序
        unmatched_predictions = []

        # 第一步：先匹配真阳性
        for i, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
            matched = False
            best_iou = 0
            best_gt_idx = -1

            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if j in matched_gt_indices:
                    continue

                iou = self._calculate_iou(pred_box, gt_box)
                if iou > best_iou and iou >= self.ir_iou_thr and pred_label == gt_label:
                    best_iou = iou
                    best_gt_idx = j
                    matched = True

            if matched:
                true_positives.append({
                    'bbox': pred_box,
                    'label': pred_label,
                    'score': pred_score,
                    'matched_gt_idx': best_gt_idx
                })
                matched_gt_indices.add(best_gt_idx)
            else:
                # 未匹配的预测加入列表，稍后按得分排序
                unmatched_predictions.append({
                    'index': i,
                    'bbox': pred_box,
                    'label': pred_label,
                    'score': pred_score
                })

        # 第二步：将未匹配的预测按得分排序，只取前3个作为误检
        if unmatched_predictions:
            # 按得分降序排序
            unmatched_predictions.sort(key=lambda x: x['score'], reverse=True)

            # 只取前3个（或者设定其他限制数量）
            max_fp_display = 3
            for pred_info in unmatched_predictions[:max_fp_display]:
                false_positives.append({
                    'bbox': pred_info['bbox'],
                    'label': pred_info['label'],
                    'score': pred_info['score']
                })

        # Find false negatives (unmatched ground truth)
        false_negatives = []
        for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if j not in matched_gt_indices:
                false_negatives.append({
                    'bbox': gt_box,
                    'label': gt_label
                })

        # 打印限制信息
        if len(unmatched_predictions) > 3:
            print(f"  ⚠️  IR误检太多，只显示得分最高的3个（原{len(unmatched_predictions)}个）")
            # 显示前3个误检的得分
            for i, pred_info in enumerate(unmatched_predictions[:3]):
                print(f"    误检{i + 1}: 得分={pred_info['score']:.4f}")

        return true_positives, false_positives, false_negatives

    def _get_ir_image_path(self, rgb_path: str) -> str:
        """Get corresponding IR image path from RGB image path."""
        rgb_dir = osp.dirname(rgb_path)
        rgb_filename = osp.basename(rgb_path)

        # 解析RGB文件名
        rgb_name_without_ext = osp.splitext(rgb_filename)[0]

        # 构建IR图像文件名
        if self.thermal_suffix in rgb_name_without_ext:
            ir_name_without_ext = rgb_name_without_ext
        else:
            ir_name_without_ext = f"{rgb_name_without_ext}{self.thermal_suffix}"

        # 尝试不同扩展名
        for ext in ['.png', '.jpg', '.jpeg']:
            ir_path = osp.join(rgb_dir, f"{ir_name_without_ext}{ext}")
            if osp.exists(ir_path):
                return ir_path

        return None

    def _load_image(self, img_path: str):
        """Load image with error handling."""
        try:
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            return img
        except Exception as e:
            return None

    def _infer_image(self, img_path):
        """Perform inference on a single image for TwoStreamATSS model."""
        if self.model is None:
            return None

        try:
            # 加载图像
            img = mmcv.imread(img_path, channel_order='rgb')

            # 准备输入数据
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            img_tensor = img_tensor.unsqueeze(0).cuda()

            batch_inputs = {
                'img_vis': img_tensor,
                'img_lwir': img_tensor.clone()
            }

            # 创建data_sample
            from mmdet.structures import DetDataSample
            data_sample = DetDataSample()
            data_sample.set_metainfo({
                'img_shape': img.shape[:2],
                'scale_factor': (1.0, 1.0),
                'ori_shape': img.shape[:2]
            })

            # 前向推理
            with torch.no_grad():
                results = self.model.predict(batch_inputs, [data_sample], rescale=True)

            result = results[0]

            # 调试信息
            if hasattr(result, 'pred_instances') and result.pred_instances is not None:
                pred_instances = result.pred_instances
                if hasattr(pred_instances, 'bboxes') and pred_instances.bboxes is not None:
                    num_preds = len(pred_instances.bboxes)
                    print(f"📦 {osp.basename(img_path)}预测框: {num_preds}")

                    if num_preds > 0 and hasattr(pred_instances, 'scores'):
                        scores = pred_instances.scores.cpu().numpy()
                        print(f"  分数范围: {scores.min():.4f} ~ {scores.max():.4f}")

            return result

        except Exception as e:
            print(f"❌ 推理失败 {img_path}: {e}")
            return None

    def _draw_detections_on_image(self, img, true_positives, false_positives, false_negatives, modality='rgb'):
        """Draw detections on an image with modality-specific coloring."""
        visualizer = Visualizer()
        visualizer.set_image(img)

        # 选择颜色方案
        colors = self.colors if modality == 'rgb' else self.ir_colors

        # 绘制正确检测
        for tp in true_positives:
            try:
                bbox = tp['bbox']
                if isinstance(bbox, np.ndarray) and len(bbox.shape) == 1:
                    bbox = bbox[None, :]
                visualizer.draw_bboxes(
                    bbox,
                    edge_colors=colors['true_positive'],
                    line_widths=3
                )
            except Exception as e:
                continue

        # 绘制误检
        for fp in false_positives:
            try:
                bbox = fp['bbox']
                if isinstance(bbox, np.ndarray) and len(bbox.shape) == 1:
                    bbox = bbox[None, :]
                visualizer.draw_bboxes(
                    bbox,
                    edge_colors=colors['false_positive'],
                    line_widths=2
                )
            except Exception as e:
                continue

        # 绘制漏检
        for fn in false_negatives:
            try:
                bbox = fn['bbox']
                if isinstance(bbox, np.ndarray) and len(bbox.shape) == 1:
                    bbox = bbox[None, :]
                visualizer.draw_bboxes(
                    bbox,
                    edge_colors=colors['false_negative'],
                    line_widths=2
                )
            except Exception as e:
                continue

        return visualizer.get_image()

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: Dict,
                        outputs: Sequence[DetDataSample]) -> None:
        """Run after every testing iterations."""
        if not self.draw:
            return

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        for i, data_sample in enumerate(outputs):
            self._test_index += 1


            self._visualize_comparison(data_sample, runner, 'test')

    def _visualize_comparison(self, data_sample: DetDataSample, runner: Runner, mode: str):
        """Visualize comparison between predictions and ground truth."""
        rgb_path = data_sample.img_path

        # 获取IR图像路径
        ir_path = self._get_ir_image_path(rgb_path)

        # 加载RGB图像
        rgb_img = self._load_image(rgb_path)
        if rgb_img is None:
            return

        # 获取RGB的预测和真实标签
        rgb_pred_instances = data_sample.pred_instances if hasattr(data_sample, 'pred_instances') else None
        rgb_gt_instances = data_sample.gt_instances if hasattr(data_sample, 'gt_instances') else None

        print(f"\n{'=' * 60}")
        print(f"🖼️ 处理图像 {self._test_index}: {osp.basename(rgb_path)}")
        print(f"{'=' * 60}")

        # 处理RGB图像
        if rgb_gt_instances is None or (hasattr(rgb_gt_instances, 'bboxes') and
                                        (rgb_gt_instances.bboxes is None or len(rgb_gt_instances.bboxes) == 0)):
            print("⚠️  RGB没有GT标签")
            if rgb_pred_instances is not None and hasattr(rgb_pred_instances, 'bboxes') and \
                    rgb_pred_instances.bboxes is not None and len(rgb_pred_instances.bboxes) > 0:
                rgb_true_positives = []
                rgb_false_positives = []
                rgb_false_negatives = []

                pred_boxes = rgb_pred_instances.bboxes.cpu().numpy()
                pred_scores = rgb_pred_instances.scores.cpu().numpy() if hasattr(rgb_pred_instances,
                                                                                 'scores') else np.ones(len(pred_boxes))
                pred_labels = rgb_pred_instances.labels.cpu().numpy() if hasattr(rgb_pred_instances,
                                                                                 'labels') else np.zeros(
                    len(pred_boxes))

                # 使用RGB阈值过滤
                keep = pred_scores >= self.rgb_score_thr
                filtered_boxes = pred_boxes[keep]
                filtered_scores = pred_scores[keep]

                print(f"📦 RGB预测框: {len(pred_boxes)} → 过滤后: {len(filtered_boxes)} (分数>{self.rgb_score_thr})")

                for i, (pred_box, pred_label, pred_score) in enumerate(
                        zip(filtered_boxes, pred_labels[keep], filtered_scores)):
                    rgb_false_positives.append({
                        'bbox': pred_box,
                        'label': pred_label,
                        'score': pred_score
                    })
            else:
                rgb_true_positives, rgb_false_positives, rgb_false_negatives = [], [], []
                print("❌ RGB没有预测结果")
        else:
            # 有GT，使用RGB专用匹配
            rgb_true_positives, rgb_false_positives, rgb_false_negatives = self._match_detections_rgb(
                rgb_pred_instances, rgb_gt_instances
            )

            print(f"📊 RGB匹配结果:")
            print(f"  - TP: {len(rgb_true_positives)} (分数>{self.rgb_score_thr}, IoU>{self.rgb_iou_thr})")
            print(f"  - FP: {len(rgb_false_positives)}")
            print(f"  - FN: {len(rgb_false_negatives)}")

            if len(rgb_true_positives) > 0:
                print(f"  📈 RGB最高分TP: {max([tp['score'] for tp in rgb_true_positives]):.4f}")

        # 绘制RGB图像
        rgb_drawn_img = self._draw_detections_on_image(
            rgb_img, rgb_true_positives, rgb_false_positives, rgb_false_negatives, modality='rgb'
        )

        # 保存RGB图片
        if self.test_out_dir is not None and mode == 'test':
            rgb_out_file = osp.basename(rgb_path)
            rgb_out_file = osp.join(self.test_out_dir, rgb_out_file)
            mkdir_or_exist(osp.dirname(rgb_out_file))
            mmcv.imwrite(rgb_drawn_img[..., ::-1], rgb_out_file)
            print(f"💾 保存RGB图像")

        # 处理IR图像（如果存在）
        if ir_path and osp.exists(ir_path):
            # 加载IR图像
            ir_img = self._load_image(ir_path)
            if ir_img is not None:
                print(f"🌡️  处理IR图像: {osp.basename(ir_path)}")

                # 关键：对IR图像进行单独的推理
                if self.infer_ir_separately and self.model is not None:
                    # 执行推理
                    ir_result = self._infer_image(ir_path)

                    if ir_result is not None:
                        ir_pred_instances = ir_result.pred_instances

                        # 使用与RGB相同的GT（标签共享）
                        ir_gt_instances = rgb_gt_instances

                        # 匹配IR检测结果（使用IR专用匹配）
                        if ir_gt_instances is None or (hasattr(ir_gt_instances, 'bboxes') and
                                                       (ir_gt_instances.bboxes is None or len(
                                                           ir_gt_instances.bboxes) == 0)):
                            print("⚠️  IR没有GT标签")
                            if ir_pred_instances is not None and hasattr(ir_pred_instances, 'bboxes') and \
                                    ir_pred_instances.bboxes is not None and len(ir_pred_instances.bboxes) > 0:
                                ir_true_positives = []
                                ir_false_positives = []
                                ir_false_negatives = []

                                pred_boxes = ir_pred_instances.bboxes.cpu().numpy()
                                pred_scores = ir_pred_instances.scores.cpu().numpy() if hasattr(ir_pred_instances,
                                                                                                'scores') else np.ones(
                                    len(pred_boxes))
                                pred_labels = ir_pred_instances.labels.cpu().numpy() if hasattr(ir_pred_instances,
                                                                                                'labels') else np.zeros(
                                    len(pred_boxes))

                                # 使用IR阈值过滤
                                keep = pred_scores >= self.ir_score_thr
                                filtered_boxes = pred_boxes[keep]
                                filtered_scores = pred_scores[keep]

                                print(
                                    f"📦 IR预测框: {len(pred_boxes)} → 过滤后: {len(filtered_boxes)} (分数>{self.ir_score_thr})")

                                for i, (pred_box, pred_label, pred_score) in enumerate(
                                        zip(filtered_boxes, pred_labels[keep], filtered_scores)):
                                    ir_false_positives.append({
                                        'bbox': pred_box,
                                        'label': pred_label,
                                        'score': pred_score
                                    })
                            else:
                                ir_true_positives, ir_false_positives, ir_false_negatives = [], [], []
                                print("❌ IR没有预测结果")
                        else:
                            # 有GT，使用IR专用匹配
                            ir_true_positives, ir_false_positives, ir_false_negatives = self._match_detections_ir(
                                ir_pred_instances, ir_gt_instances
                            )

                            print(f"📊 IR匹配结果:")
                            print(f"  - TP: {len(ir_true_positives)} (分数>{self.ir_score_thr}, IoU>{self.ir_iou_thr})")
                            print(f"  - FP: {len(ir_false_positives)}")
                            print(f"  - FN: {len(ir_false_negatives)}")

                            if len(ir_true_positives) > 0:
                                print(f"  📈 IR最高分TP: {max([tp['score'] for tp in ir_true_positives]):.4f}")

                        # 绘制IR图像（使用IR专用颜色）
                        ir_drawn_img = self._draw_detections_on_image(
                            ir_img,
                            ir_true_positives,
                            ir_false_positives,
                            ir_false_negatives,
                            modality='ir'
                        )

                        # 保存IR图片
                        if self.test_out_dir is not None and mode == 'test':
                            ir_out_file = osp.basename(ir_path)
                            ir_out_file = osp.join(self.test_out_dir, ir_out_file)
                            mkdir_or_exist(osp.dirname(ir_out_file))
                            mmcv.imwrite(ir_drawn_img[..., ::-1], ir_out_file)
                            print(f"💾 保存IR图像（使用独立预测）")

                else:
                    # 如果不进行单独推理，使用RGB的预测结果
                    print(f"⚠️  IR图像使用RGB的预测结果")
                    ir_drawn_img = self._draw_detections_on_image(
                        ir_img,
                        rgb_true_positives,
                        rgb_false_positives,
                        rgb_false_negatives,
                        modality='ir'  # 仍然使用IR颜色区分
                    )

                    # 保存IR图片
                    if self.test_out_dir is not None and mode == 'test':
                        ir_out_file = osp.basename(ir_path)
                        ir_out_file = osp.join(self.test_out_dir, ir_out_file)
                        mkdir_or_exist(osp.dirname(ir_out_file))
                        mmcv.imwrite(ir_drawn_img[..., ::-1], ir_out_file)
                        print(f"💾 保存IR图像（使用RGB预测）")