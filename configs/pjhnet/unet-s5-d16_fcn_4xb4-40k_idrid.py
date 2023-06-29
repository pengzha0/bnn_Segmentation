_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/idrid.py',
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
    decode_head=dict(
        num_classes=5,
    ),
    auxiliary_head=dict(
        num_classes=5,
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole',compute_aupr=True))
