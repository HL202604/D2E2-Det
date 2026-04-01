# visualize_folder.py
import argparse
import os
import os.path as osp
import glob
import mmcv
import cv2
import numpy as np
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmengine.visualization import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize detections on folder images')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--img-dir', required=True, help='image directory')
    parser.add_argument('--output-dir', required=True, help='output directory')
    parser.add_argument('--score-thr', type=float, default=0.3, help='score threshold')
    parser.add_argument('--extensions', nargs='+', default=['.jpg', '.png', '.jpeg'],
                        help='image extensions to process')
    parser.add_argument('--no-gt', action='store_true', help='no ground truth, just show predictions')
    parser.add_argument('--color', type=str, default='red', choices=['red', 'green', 'blue', 'white'],
                        help='color for bboxes when no GT')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 初始化模型
    print(f"Loading model from {args.checkpoint}...")
    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 颜色映射
    colors = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'white': (255, 255, 255)
    }
    pred_color = colors[args.color]

    # 查找所有图像文件
    image_files = []
    for ext in args.extensions:
        pattern = osp.join(args.img_dir, f'*{ext}')
        image_files.extend(glob.glob(pattern))

    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images in {args.img_dir}")

    # 处理每张图像
    for i, img_path in enumerate(image_files):
        if i % 10 == 0:
            print(f"Processing {i + 1}/{len(image_files)}: {osp.basename(img_path)}")

        # 加载图像
        img = mmcv.imread(img_path, channel_order='rgb')

        # 推理
        result = inference_detector(model, img_path)

        # 获取预测结果
        pred_instances = result.pred_instances

        # 创建可视化器
        visualizer = Visualizer()
        visualizer.set_image(img)

        # 绘制预测框
        if pred_instances is not None and hasattr(pred_instances, 'bboxes') and \
                pred_instances.bboxes is not None and len(pred_instances.bboxes) > 0:

            # 过滤低分预测
            if hasattr(pred_instances, 'scores'):
                keep = pred_instances.scores >= args.score_thr
                pred_boxes = pred_instances.bboxes[keep].cpu().numpy()
                pred_labels = pred_instances.labels[keep].cpu().numpy() if hasattr(pred_instances, 'labels') else None
            else:
                pred_boxes = pred_instances.bboxes.cpu().numpy()
                pred_labels = None

            # 绘制每个bbox
            for j, bbox in enumerate(pred_boxes):
                # 如果有标签信息，可以按类别着色
                if pred_labels is not None:
                    # 简单的类别颜色映射
                    label = pred_labels[j]
                    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                             (255, 0, 255), (0, 255, 255)][label % 6]
                else:
                    color = pred_color

                visualizer.draw_bboxes(
                    bbox[None, :],
                    edge_colors=color,
                    line_widths=2
                )

        # 获取绘制的图像
        drawn_img = visualizer.get_image()

        # 保存结果
        out_filename = osp.basename(img_path)
        out_path = osp.join(args.output_dir, out_filename)
        mmcv.imwrite(drawn_img[..., ::-1], out_path)

    print(f"✅ All {len(image_files)} images processed and saved to {args.output_dir}")


if __name__ == '__main__':
    main()