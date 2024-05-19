"""使用SAM和CLIP对原图进行增强, 相同语义的区域赋予相同的颜色"""

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
import clip
import shutil

# 输入图片，用白色padding为正方形，边长为原始图像的长边，原始图像尽可能居中
def pad_image_to_square(image):
    height, width, channels = image.shape
    
    # Find the longer side length
    longer_side = max(height, width)
    
    # Calculate the padding needed for height and width
    height_pad = longer_side - height
    width_pad = longer_side - width
    
    # Calculate the top, bottom, left, and right padding
    top_pad = height_pad // 2
    bottom_pad = height_pad - top_pad
    left_pad = width_pad // 2
    right_pad = width_pad - left_pad
    
    # Create a padded square image with white background
    padded_image = np.full((longer_side, longer_side, channels), (255, 255, 255), dtype=np.uint8)
    
    # Paste the original image in the center of the padded square
    padded_image[top_pad:top_pad+height, left_pad:left_pad+width] = image
    
    return padded_image

# 输入SAM生成的masks，并按照区域从大到小排序，挑选前1/4
def masks_select(masks):
    """
    masks: 输入基于SAM生成的masks

    masks_slct: 返回挑选后的结果
    """
    # 按照区域大小进行排序
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    # 挑选前1/4大的masks
    # masks_slct = masks[0:int(0.25*len(sorted_masks))]
    # 选取最大的3个masks
    masks_slct = masks[0:3]
    return masks_slct


# 输入图片和挑选出来的masks，输入每个masks对应的语义
def clip_semantic(masks_slct, image_ori, text):
    """
    masks_slct: 输入挑选后的masks
    image_ori: 输入原始图片
    text: clip tokenize后的文本列表

    masks_sem: 返回基于clip的masks_slct的语义
    """
    masks_sem = []
    for ann in masks_slct:
        # 得到只包含mask区域的图像，其他区域为白色
        image_sub = np.full((masks_slct[0]['segmentation'].shape[0], masks_slct[0]['segmentation'].shape[1], 3), (255, 255, 255), dtype=np.uint8)
        m = ann['segmentation']
        image_sub[m] = image_ori[m]
        # 使用白色padding为正方形，保持原图居中
        image_sub = pad_image_to_square(image_sub)
        # 这里的写法后面可以优化，目前暂且这么多一次io的来用
        # cv2.imshow('image_sub', image_sub)
        # cv2.waitKey(10)
        cv2.imwrite('image_sub.png', image_sub)
        image_clip = preprocess(Image.open('image_sub.png')).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image_clip, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        max_value = probs.max()
        max_index = np.argmax(probs)
        masks_sem.append(max_index)
        # print("Label probs:", probs)
        # print(f"Maximum value: {max_value}")
        # print(f"Index (id) of maximum value: {max_index}")
        print('类别为： ', classes[max_index])
    # print('masks_sem: ', masks_sem)
    return masks_sem
    

# 将语义masks叠加到原图并保存
def save_slip_aug(masks_slct, masks_sem, img):
    """
    masks_slct: 输入挑选后的masks
    masks_sem: 输入基于clip的masks的语义
    img: 输入原始图片
    """
    # 创建白底图像,mask颜色基于语义查询得到
    image_white = np.full((masks_slct[0]['segmentation'].shape[0], masks_slct[0]['segmentation'].shape[1], 3), (255, 255, 255), dtype=np.uint8)
    for i, ann in enumerate(masks_slct):
        m = ann['segmentation']
        color_mask = palette[masks_sem[i]]
        image_white[m] = color_mask
    # 对mask和image进行加权叠加
    mask_weight = 0.25
    image_weight = 0.75
    result = cv2.addWeighted(image_white, mask_weight, img, image_weight, 0)
    cv2.imwrite(save_path, result)
    

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Segment Anything Model')
    parser.add_argument('--image_dir', type=str, default='images', help='输入图像文件夹')
    parser.add_argument('--output_dir', type=str, default='output', help='输出图像文件夹')
    args = parser.parse_args()

    # 加载SAM模型
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # 加载CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    """cityscape
    classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle']
    text = clip.tokenize(classes).to(device)
    palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    """
    classes = ['Helmet', 'Shelf', 'Mug', 'Chair', 'Oven', 'Calculator', 'Refrigerator', 'Pencil',
                'Fork', 'Scissors', 'Clipboards', 'Speaker', 'Folder', 'Drill', 'Table', 'Alarm_Clock',
                  'Screwdriver', 'Couch', 'TV', 'Eraser', 'Webcam', 'Kettle', 'Lamp_Shade', 'Pan',
                    'Push_Pin', 'Soda', 'Sneakers', 'Calendar', 'Mop', 'Sink', 'Exit_Sign', 'Laptop',
                      'Backpack', 'Flowers', 'Hammer', 'Fan', 'Telephone', 'Ruler', 'Trash_Can',
                        'Knives', 'Bucket', 'Desk_Lamp', 'Pen', 'Bed', 'Bike', 'File_Cabinet', 'Keyboard',
                          'Paper_Clip', 'Flipflops', 'Notebook', 'Postit_Notes', 'Radio', 'ToothBrush',
                            'Bottle', 'Mouse', 'Marker', 'Spoon', 'Glasses', 'Candles', 'Batteries',
                              'Printer', 'Computer', 'Toys', 'Curtains', 'Monitor']
    text = clip.tokenize(classes).to(device)
    palette=[
        [145, 173, 190], [250, 154, 139], [168, 219, 230], [111, 73, 248], [20, 117, 51], 
        [151, 92, 245], [198, 177, 218], [164, 11, 224], [254, 100, 137], [111, 157, 52], 
        [10, 135, 25], [216, 69, 254], [78, 158, 126], [219, 198, 73], [152, 202, 231], 
        [226, 185, 206], [193, 71, 249], [194, 131, 138], [2, 194, 177], [203, 248, 221], 
        [26, 123, 203], [209, 13, 196], [23, 217, 34], [70, 159, 35], [83, 220, 171], 
        [109, 253, 236], [53, 143, 44], [74, 119, 113], [6, 162, 163], [205, 207, 9], 
        [226, 37, 38], [9, 39, 111], [101, 247, 230], [238, 243, 106], [228, 43, 159], 
        [166, 136, 226], [94, 111, 208], [233, 255, 41], [255, 191, 239], [57, 177, 175], 
        [57, 234, 213], [89, 85, 64], [54, 169, 174], [61, 218, 227], [66, 58, 187], 
        [113, 16, 47], [164, 78, 164], [41, 88, 111], [23, 151, 155], [60, 78, 100], 
        [139, 178, 151], [175, 22, 140], [7, 130, 70], [161, 241, 149], [206, 244, 124], 
        [103, 109, 157], [248, 206, 10], [124, 127, 69], [181, 113, 210], [231, 92, 166], 
        [146, 218, 126], [118, 234, 3], [77, 136, 174], [168, 195, 87], [45, 173, 115]
]

    for root, dirs, files in os.walk(args.image_dir):
        for filename in tqdm(files):
            # 获取文件扩展名
            ext = os.path.splitext(filename)[1].lower()
            # 如果扩展名不是.jpg或.png,则跳过该文件
            if ext not in ['.jpg', '.png']:
                continue
            print(filename)
            image_path = os.path.join(root, filename)
            relative_path = os.path.relpath(image_path, args.image_dir)
            save_path = os.path.join(args.output_dir, relative_path)
            # 如果目标路径存在，跳过，说明这个文件之前已经有了
            if os.path.exists(save_path):
                print(f"跳过文件：{save_path}，因为目标路径已存在文件")
                continue
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                # 输入图片，基于SAM生成masks
                masks = mask_generator.generate(image_rgb)
                # 挑选前1/4大的masks
                masks_slct = masks_select(masks)
                # 基于clip生成语义
                masks_sem = clip_semantic(masks_slct, image, text)
                # 保存 SAM and CLIP 增强后的图片
                save_slip_aug(masks_slct, masks_sem, image)
            except Exception as e:
                # 如果上面出现CUDA out of memory等异常，直接copy图片
                shutil.copy(image_path, save_path)

