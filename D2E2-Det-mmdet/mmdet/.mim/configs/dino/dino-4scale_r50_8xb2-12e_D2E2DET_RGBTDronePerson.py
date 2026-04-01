_base_ = [
    '../_base_/datasets/RGBTDronePerson.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='DINOD2E2DET',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='PairedDetDataPreprocessor',
        mean=[159.8808906080302, 162.22057018543336, 160.28301196773916],
        std=[56.96897676312916, 59.57937492901139, 63.11906486423505],
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
        in_channels=[1024, 2048, 4096],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),

    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_levels=4,
                dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0))),

    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                dropout=0.0),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_levels=4,
                dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0)),
        post_norm_cfg=None),

    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20),

    bbox_head=dict(
        type='DINOHead',
        num_classes=3,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),

    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),

    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300)
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone_vis': dict(lr_mult=0.1),
            'backbone_lwir': dict(lr_mult=0.1)
        })
)

max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

auto_scale_lr = dict(base_batch_size=16)