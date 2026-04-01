# test_with_compare.py
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test with comparison visualization')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the directory to save evaluation metrics')
    parser.add_argument('--compare-dir', default='detection_comparison', help='directory for comparison results')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override settings')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')  # 添加 launcher 参数
    parser.add_argument('--local_rank', type=int, default=0)  # 添加 local_rank 参数
    args = parser.parse_args()

    # 设置本地 rank 环境变量
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher  # 设置 launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    cfg.load_from = args.checkpoint

    # 添加我们的对比可视化 Hook
    # cfg.custom_hooks = [
    #     dict(
    #         type='DetectionCompareHook',
    #         draw=True,
    #         interval=1,
    #         score_thr=0.3,
    #         iou_thr=0.5,
    #         show=False,
    #         test_out_dir=args.compare_dir,
    #     )
    # ]

    # cfg.custom_hooks = [
    #     dict(
    #         type='DetectionCompareHook',
    #         draw=True,
    #         interval=1,
    #         score_thr=0.3,
    #         iou_thr=0.5,
    #         show=False,
    #         test_out_dir=args.compare_dir,
    #         visualize_both_modalities=True,  # 新增：同时可视化RGB和红外
    #         thermal_suffix='_lwir',  # 新增：红外图像后缀
    #     )
    # ]

    cfg.custom_hooks = [
        dict(
            type='DetectionCompareHook',
            draw=True,
            interval=1,
            rgb_score_thr=0.3,  # RGB的高阈值
            ir_score_thr=0.01,  # IR的低阈值
            rgb_iou_thr=0.5,  # RGB的高IoU阈值
            ir_iou_thr=0.5,  # IR的低IoU阈值
            show=False,
            test_out_dir=args.compare_dir,
            thermal_suffix='_lwir',
            infer_ir_separately=True,
        )
    ]



    # build the runner from config
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
