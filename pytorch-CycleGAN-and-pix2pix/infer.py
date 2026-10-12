import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

# python infer.py --dataroot datasets/test/opt2sar --name opt2sar --model pix2pix --gpu_ids 7 --results_dir results/opt2sar
# python infer.py --dataroot datasets/test/sar2opt --name sar2opt --model pix2pix --gpu_ids 7 --results_dir results/sar2opt

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    if opt.name == 'opt2sar':
        opt.name = 'WHU-City_opt2sar_pix2pix'
    elif opt.name == 'sar2opt':
        opt.name = 'WHU-City_sar2opt_pix2pix'
    elif opt.name == 'ir2opt':
        opt.name == 'VEDAI_ir2opt_pix2pix'
    elif opt.name == 'opt2ir':
        opt.name == 'VEDAI_opt2ir_pix2pix'
        
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()

    img_list = []

    # walk through img files in dataroot
    for root, dirs, files in os.walk(opt.dataroot):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                img_list.append(os.path.join(root, file))

    transform = transforms.Compose([
        transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    os.makedirs(opt.results_dir, exist_ok=True)
    
    for img_path in img_list:
        img = Image.open(img_path).convert('RGB')
        # 使用 PIL 的 Image.open 方法打开当前图像路径 img_path 指向的图像文件，并使用 convert('RGB') 确保图像是 RGB 格式
        out = model.netG(transform(img)[None])[0].detach().cpu().float().numpy()
        # 应用之前定义的图像转换操作 transform 到图像 img 上
        out = (np.transpose(out, (1, 2, 0)) + 1) / 2.0 * 255.0
        # 使用 np.transpose 将 NumPy 数组的维度从 (C, H, W) 转换为 (H, W, C)，其中 C 是通道数，H 是高度，W 是宽度
        # 将像素值从 [-1.0, 1.0] 范围转换回 [0, 255] 范围，以便保存为图像文件
        out = out.astype(np.uint8)
        # 将 NumPy 数组的数据类型转换为无符号 8 位整数（np.uint8），这是保存图像文件所需的数据类型
        image_pil = Image.fromarray(out)
        # 使用 PIL 的 Image.fromarray 方法将 NumPy 数组转换回 PIL 图像
        image_pil.save(os.path.join(opt.results_dir, os.path.basename(img_path)))
        # 使用 os.path.join 和 os.path.basename 构造输出图像的完整路径
        # 使用 PIL 图像的 save 方法将图像保存到结果目录 opt.results_dir 中

       
       

        # out = model.netG(transform(img)[None])[0].detach().cpu().permute(1, 2, 0) * 0.5 + 0.5
        # out = cv2.cvtColor(np.array(out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(opt.results_dir, os.path.basename(img_path)), out)

