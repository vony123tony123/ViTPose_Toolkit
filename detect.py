import argparse
import os.path as osp

import torch
from torch import Tensor

from pathlib import Path
import cv2
import numpy as np
import importlib

from time import time
from PIL import Image
from torchvision.transforms import transforms

from ViTPose.models.model import ViTPose
from utils.visualization import draw_points_and_skeleton, joints_dict
from utils.dist_util import get_dist_info, init_dist
from utils.top_down_eval import keypoints_from_heatmaps



def read_config(config_path):
    model_cfg = {}
    print(config_path)
    exec(open(config_path).read(), model_cfg)
    img_size, model_cfg = model_cfg['data_cfg']['image_size'], model_cfg['model']
    
    return img_size, model_cfg
            
class vitPose:
    def __init__(self, config_path, weights_path):
        # Prepare model
        self.img_size, model_cfg = read_config(config_path)
        self.vit_pose = ViTPose(model_cfg)
        self.device=torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        ckpt = torch.load(weights_path)
        if 'state_dict' in ckpt:
            self.vit_pose.load_state_dict(ckpt['state_dict'])
        else:
            self.vit_pose.load_state_dict(ckpt)
        self.vit_pose.to(self.device)
        print(f">>> Model loaded: {weights_path}")

    def detect(self, img):
        # Prepare input data
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        org_w, org_h = img.size
        print(f">>> Original image size: {org_h} X {org_w} (height X width)")
        print(f">>> Resized image size: {self.img_size[1]} X {self.img_size[0]} (height X width)")
        print(f">>> Scale change: {org_h/self.img_size[1]}, {org_w/self.img_size[0]}")
        img_tensor = transforms.Compose (
            [transforms.Resize((self.img_size[1], self.img_size[0])),
             transforms.ToTensor()]
        )(img).unsqueeze(0).to(self.device)

        # Feed to model
        tic = time()
        heatmaps = self.vit_pose(img_tensor).detach().cpu().numpy() # N, 17, h/4, w/4
        elapsed_time = time()-tic
        print(f">>> Output size: {heatmaps.shape} ---> {elapsed_time:.4f} sec. elapsed [{elapsed_time**-1: .1f} fps]\n")    
        
        # points = heatmap2coords(heatmaps=heatmaps, original_resolution=(org_h, org_w))
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w//2, org_h//2]]), scale=np.array([[org_w, org_h]]),
                                               unbiased=True, use_udp=True)
        points = np.concatenate([points[:, :, ::-1], prob], axis=2)
        return points[0]

    def visualization(img, points):
        img = draw_points_and_skeleton(img.copy(), points, joints_dict()['coco']['skeleton'],
                                       points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                       points_palette_samples=10, confidence_threshold=0.4)

        for point in points:
            x, y = int(point[1]), int(point[0])
            text = f'({x}, {y})'
            text_color = (0, 0, 255)  # 使用BGR格式指定顏色，這裡使用綠色
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.5
            text_offset_x = 10  # X方向上的偏移量
            text_offset_y = -10  # Y方向上的偏移量
            text_org = (x + text_offset_x, y + text_offset_y)
            cv2.putText(img, text, text_org, font, font_size, text_color, 1, cv2.LINE_AA)

        return img



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/vitpose-b-multi-coco.pth', help='model.pt path(s)')
    parser.add_argument('--config', type=str, default='configs/ViTPose_base_coco_256x192.py', help='config path(s)')
    parser.add_argument('--source', type=str, default='examples/img1.jpg', help='source)')
    parser.add_argument('--img-size', type=int, default=256, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    opt = parser.parse_args()
    print(opt)

    vitpose = vitPose(opt.config, opt.weights)
    img = cv2.imread(opt.source)
    points = vitpose.detect(img)
    print(points)
    cv2.namedWindow('l',cv2.WINDOW_NORMAL)
    cv2.imshow('l', vitPose.visualization(img, points))
    cv2.waitKey(0)
    cv2.destroyAllWindows()