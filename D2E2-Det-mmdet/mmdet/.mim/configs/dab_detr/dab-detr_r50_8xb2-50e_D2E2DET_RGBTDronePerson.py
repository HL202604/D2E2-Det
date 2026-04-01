_base_ = [
    '../_base_/datasets/RGBTDronePerson.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='DABDETRD2E2DET',
    num_queries=300,
    with_random_refpoints=False,
    num_patterns=0,
    data_preprocessor=dict(
        type='PairedDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        mean_lwir=[136.63746562356317, 136.63746562356317, 136.63746562356317],
        std_lwir=[64.97730349740912, 64.97730349740912, 64.97730349740912],
        bgr_to_rgb=True,
        pad_size_divisor=1),

    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),

    neck=dict(
        type='ChannelMapper',
        in_channels=[4096],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=None,
        num_outs=1),

    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0., batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.,
                act_cfg=dict(type='PReLU')))),

    decoder=dict(
        num_layers=6,
        query_dim=4,
        query_scale_type='cond_elewise',
        with_modulated_hw_attn=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                attn_drop=0.,
                proj_drop=0.,
                cross_attn=False),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                attn_drop=0.,
                proj_drop=0.,
                cross_attn=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.,
                act_cfg=dict(type='PReLU'))),
        return_intermediate=True),

    positional_encoding=dict(
        num_feats=128,
        temperature=20,
        normalize=True),

    bbox_head=dict(
        type='DABDETRHead',
        num_classes=3,
        embed_dims=256,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),

    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2., eps=1e-8),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300)
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embedding': dict(lr_mult=1.0, decay_mult=0.0)
        }))

max_epochs = 50
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]

auto_scale_lr = dict(base_batch_size=16, enable=False)