# anything模式
# 分割区域面积从大到小排序,随机赋予颜色

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

def show_anns(anns):
    if len(anns) == 0:
        return
    # 对mask按照置信度进行排序
    # anns = sorted(anns, key=(lambda x: x['stability_score']), reverse=True)

    # 按照区域大小进行排序
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    # 取前20个
    sorted_anns = sorted_anns[0:30]

    ax = plt.gca()
    ax.set_autoscale_on(False)
    # 创建的img数组为四通道，而不是三通道，是因为其中包含了透明度通道,这里的img是mask
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    # 将透明度通道设置为0完全透明
    img[:,:,3] = 0
    for ann in sorted_anns:
        # print("ann: ", ann)
        m = ann['segmentation']
        # color_mask是一个长度为4的一维数组，其中前三个元素表示红色（R）、绿色（G）、蓝色（B）通道的数值，最后一个元素表示透明度（Alpha）通道的数值。
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        cv2.imshow('a', img)
        cv2.waitKey(0)
    # 显示分割区域信息
    ax.imshow(img)


def save_anns(anns, image):
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
    cv2.imwrite('ann.jpg', result)


image = cv2.imread('images_test/a/aachen_000000_000019_leftImg8bit.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 默认推理
start_time = time.time()
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
# print("单张图像推理耗时：", time.time() - start_time, " s")
print("mask的长度: ",len(masks))
print("mask的键: ", masks[0].keys())
print("mask的第一个: ", masks[1])

# 显示
# 原图,叠加显示mask，四通道，其中有一个透明度通道
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()

# 保存
save_anns(masks, image)
