# -*- encoding:utf-8 -*-
# !/usr/bin/env python

"""
@author：Ty Zhao
@fileName：flir.py
@Date：2023/9/17
@Description:
"""
dataset_type = 'MultispectralDataset'
data_root = '/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/RSDet-master/RGBTDronePerson-RGBIR/'
backend_args = None
classes = ('person', 'rider', 'crowd')
train_pipeline = [
    dict(type='LoadPairedImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PairedImagesResize', scale=(512, 640), keep_ratio=True),
    # dict(type='PairedImagesResize', scale=(320, 256), keep_ratio=True),
    dict(type='PairedImageRandomFlip', prob=0.5),
    # dict(type='PairedFrequencyProcess',alpha=a,beta=b),
    # dict(type='PairedImagesRandomResize', scale=(1280,1024),ratio_range=(0.1, 2.0), keep_ratio=True),
    # dict(
    #     type='AlignedImagesRandomCrop',
    #     crop_type='absolute_range',
    #     crop_size=image_size,
    #     recompute_bbox=True,
    #     allow_negative_crop=True),

    dict(type='PairedImagesPad', size_divisor=32),
    dict(type='PackPairedImagesDetInputs',
            meta_keys = ('img_id', 'img_path','img_lwir_path', 'ori_shape', 'img_shape',
                         'scale_factor'))
    # dict(type='Collect', keys=['img', 'img_lwir','gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadPairedImageFromFile', to_float32=True),
    dict(type='PairedImagesResize', scale=(512, 640), keep_ratio=True),
    # dict(type='PairedImagesResize', scale=(320, 256), keep_ratio=True),
    dict(type='PairedImagesPad', size_divisor=32),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackPairedImagesDetInputs',
         meta_keys=('img_id', 'img_path', 'img_lwir_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
        ]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    # persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='train/annotations/train.json',
        data_prefix=dict(img='train/images/',

                         ),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    # persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='val/annotations/val.json',
        data_prefix=dict(img='val/images/',
                         ),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/annotations/val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

