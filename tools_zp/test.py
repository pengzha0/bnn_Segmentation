from mmengine.config import Config

cfg = Config.fromfile('configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py')
print(cfg.data_preprocessor)