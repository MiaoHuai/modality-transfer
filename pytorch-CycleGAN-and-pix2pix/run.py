import os
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images


class Runner():
    def __init__(self):
        print("Runner initializing.")
        self.device = 'cuda'
        
        model_path_1 = '/checkpoints/VEDAI_ir2opt_pix2pix/latest_net_G.pth'
        self.transfer_model_1.load_state_dict(torch.load(model_path_1, map_location='cpu'))  
        self.transfer_model_1 = self.transfer_model_1.to(self.device).eval() 

        model_path_2 = '/checkpoints/VEDAI_opt2ir_pix2pix/latest_net_G.pth'
        self.transfer_model_2.load_state_dict(torch.load(model_path_2, map_location='cpu'))  
        self.transfer_model_2 = self.transfer_model_2.to(self.device).eval() 
        
        model_path_3 = '/checkpoints/WHU-City_opt2sar_pix2pix/latest_net_G.pth'
        self.transfer_model_3.load_state_dict(torch.load(model_path_3, map_location='cpu'))  
        self.transfer_model_3 = self.transfer_model_3.to(self.device).eval() 
        
        model_path_4 = '/checkpoints/WHU-City_sar2opt_pix2pix/latest_net_G.pth'
        self.transfer_model_4.load_state_dict(torch.load(model_path_4, map_location='cpu'))  
        self.transfer_model_4 = self.transfer_model_4.to(self.device).eval() 
        print("model loaded.") 
        
    
    def run_batch(self, config):
        
        self.checkpoints_dir = 'checkpoints'  # 主目录
        self.modalityAugmentConfig = config['modalityAugmentConfig']
        transfer_model_config = TestOptions().parse()
        transfer_model_config = {'num_threads': 0, 'batch_size': 1, 'serial_batches': True, 'no_flip': True, 'dataroot': config['filePaths'], 'result_dir': config['outputPath'], 'name': set_name(self, config['modalityAugmentConfig']) }
        self.transfer_model = create_model(**transfer_model_config)
        model_path = os.path.join(checkpoints_dir, set_name(self, config['modalityAugmentConfig']), 'latest_net_G.pth')
        
        out_img_path = []
        # 初始化一个空列表 out_img_path，用于存储所有输出图像的路径
        img_list = config['filePaths']
        out_img_path = config['outputPath']
        
        opt = TestOptions().parse()
        
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
            out = self.transfer_model.netG(transform(img)[None])[0].detach().cpu().float().numpy()
            out = (np.transpose(out, (1, 2, 0)) + 1) / 2.0 * 255.0
            out = out.astype(np.uint8)
            image_pil = Image.fromarray(out)
            out_path = os.path.join(opt.results_dir, os.path.basename(img_path)) 
            image_pil.save(out_path)
            out_img_path.append(out_path)

        print("process finish.")
        
        return out_img_path
        
        
        
    def set_name(self, modal):
        if modal == 'Vis-SAR':
            return 'WHU-City_opt2sar_pix2pix'
        elif modal == 'SAR-Vis':
            return 'WHU-City_sar2opt_pix2pix'
        elif modal == 'Vis-Infrared':
            return 'VEDAI_opt2ir_pix2pix'
        elif modal == 'Infrared-Vis':
            return 'VEDAI_ir2opt_pix2pix'
        else:
            raise ValueError("Invalid modalityAugmentConfig. Choose from ['Vis-SAR', 'SAR-Vis', 'Vis-Infrared', 'Infrared-Vis']")