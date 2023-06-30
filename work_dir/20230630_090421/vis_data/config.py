norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[
        116.513,
        56.437,
        16.309,
    ],
    std=[
        80.206,
        41.232,
        13.293,
    ],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(
        960,
        1440,
    ))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[
            116.513,
            56.437,
            16.309,
        ],
        std=[
            80.206,
            41.232,
            13.293,
        ],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(
            960,
            1440,
        )),
    pretrained=None,
    backbone=dict(
        type='UNet_Bin',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(
            1,
            1,
            1,
            1,
            1,
        ),
        enc_num_convs=(
            2,
            2,
            2,
            2,
            2,
        ),
        dec_num_convs=(
            2,
            2,
            2,
            2,
        ),
        downsamples=(
            True,
            True,
            True,
            True,
        ),
        enc_dilations=(
            1,
            1,
            1,
            1,
            1,
        ),
        dec_dilations=(
            1,
            1,
            1,
            1,
        ),
        with_cp=False,
        conv_cfg=dict(type='BNNConv2d'),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='Hardtanh'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole', crop_size=256, stride=170, compute_aupr=True))
dataset_type = 'LesionDataset'
data_root = '/home/zp/BNN/Seg/data/IDRID'
image_scale = (
    1440,
    960,
)
crop_size = (
    960,
    1440,
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(
            1440,
            960,
        ),
        ratio_range=(
            0.5,
            2.0,
        ),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(
        960,
        1440,
    ), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(
        1440,
        960,
    ), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=0.5, keep_ratio=True),
                dict(type='Resize', scale_factor=0.75, keep_ratio=True),
                dict(type='Resize', scale_factor=1.0, keep_ratio=True),
                dict(type='Resize', scale_factor=1.25, keep_ratio=True),
                dict(type='Resize', scale_factor=1.5, keep_ratio=True),
                dict(type='Resize', scale_factor=1.75, keep_ratio=True),
            ],
            [
                dict(type='RandomFlip', prob=0.0, direction='horizontal'),
                dict(type='RandomFlip', prob=1.0, direction='horizontal'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ]),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='LesionDataset',
        data_root='/home/zp/BNN/Seg/data/IDRID',
        data_prefix=dict(
            img_path='image/train', seg_map_path='label/train/annotations'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='RandomResize',
                scale=(
                    1440,
                    960,
                ),
                ratio_range=(
                    0.5,
                    2.0,
                ),
                keep_ratio=True),
            dict(
                type='RandomCrop', crop_size=(
                    960,
                    1440,
                ), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LesionDataset',
        data_root='/home/zp/BNN/Seg/data/IDRID',
        data_prefix=dict(
            img_path='image/test', seg_map_path='label/test/annotations'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(
                1440,
                960,
            ), keep_ratio=False),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LesionDataset',
        data_root='/home/zp/BNN/Seg/data/IDRID',
        data_prefix=dict(
            img_path='image/test', seg_map_path='label/test/annotations'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(
                1440,
                960,
            ), keep_ratio=False),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ]))
val_evaluator = dict(
    type='IoUMetric', iou_metrics=[
        'mDice',
    ])
test_evaluator = dict(
    type='IoUMetric', iou_metrics=[
        'mDice',
    ])
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0001,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1))
launcher = 'none'
work_dir = 'work_dir'
