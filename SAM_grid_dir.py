'''
运行脚本,可以在不同的shell里同时运行多个,不会冲突的
python SAM_grid_dir.py --image_dir images/sub1 --output_dir mask/sub1
'''

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
import os
import time
from tqdm import tqdm

def save_anns(anns, image, save_path):
    if len(anns) == 0:
        return
    # 按照区域大小进行排序
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # 创建黑底图像,mask颜色随机
    mask = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3), dtype=np.uint8)
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = (np.random.random(3) * 255).astype(np.uint8) # 0-255
        mask[m] = color_mask
    # 对mask和image进行加权叠加
    mask_weight = 0.35
    image_weight = 0.65
    # 确保mask和image的数据类型相同
    image = image.astype(np.uint8)
    result = cv2.addWeighted(mask, mask_weight, image, image_weight, 0)
    cv2.imwrite(save_path, result)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Segment Anything Model')
    parser.add_argument('--image_dir', type=str, default='images', help='输入图像文件夹')
    parser.add_argument('--output_dir', type=str, default='mask', help='输出图像文件夹')
    args = parser.parse_args()

    # 加载SAM模型
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    for filename in tqdm(os.listdir(args.image_dir)):
        image_path = os.path.join(args.image_dir, filename)
        save_path = os.path.join(args.output_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        # 保存
        save_anns(masks, image, save_path)