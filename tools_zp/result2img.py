from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
import os

config_file = '/home/zp/BNN/Seg/bnn_segmentation/work_dir/first_fp_idrid/vis_data/config.py'
checkpoint_file = '/home/zp/BNN/Seg/bnn_segmentation/work_dir/first_fp_idrid/iter_40000.pth'

test_data  = "/home/zp/BNN/Seg/data/IDRID/image/test"
myFolders = os.listdir(test_data)
resultFolder = "/home/zp/BNN/Seg/bnn_segmentation/work_dir/first_fp_idrid/vis_data/vis_image/"

# 根据配置文件和模型文件建立模型
model = init_model(config_file, checkpoint_file, device='cuda:0')

# # 在单张图像上测试并可视化
# img = 'demo/demo.png'  # or img = mmcv.imread(img), 这样仅需下载一次
# result = inference_model(model, img)
# # 在新的窗口可视化结果
# show_result_pyplot(model, img, result, show=True)
# # 或者将可视化结果保存到图像文件夹中
# # 您可以修改分割 map 的透明度 (0, 1].
# show_result_pyplot(model, img, result, show=True, out_file='result.jpg', opacity=0.5)

# pass
for folder in myFolders:
    img_path = test_data + '/' + folder
    result_path = resultFolder + folder
    img = mmcv.imread(img_path, 'color')
    result = inference_model(model, img)
    show_result_pyplot(model, img, result, show=False, out_file=result_path, opacity=0.9)