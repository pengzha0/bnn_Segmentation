_base_ = [
    '../_base_/models/fcn_unet_s5-d16_Bin.py', '../_base_/datasets/idrid.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k_idrid.py'
]

crop_size = (960, 1440)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[116.513, 56.437, 16.309],
    std=[80.206, 41.232, 13.293],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size = crop_size)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    data_preprocessor=data_preprocessor,
    # backbone=dict(
    #     conv_cfg = dict(type='BiRealConv2d',),
    # ),
    backbone=dict(
        type='UNet_Bin',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=dict(type='BNNConv2d',_delete_=True),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='Hardtanh'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        num_classes=5,
    ),
    auxiliary_head=dict(
        num_classes=5,
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole',compute_aupr=True))
