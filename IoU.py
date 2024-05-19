"""有红绿蓝三个类别，计算各个类别的IoU，以及平均IoU"""

import numpy as np
import os
from PIL import Image

# 定义类别列表
classes = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255])]

# 获取真值和预测值文件夹路径
gt_dir = 'annotation'
pred_dir = 'result/sourceonly/mask'

# 遍历真值和预测值文件夹
gt_files = os.listdir(gt_dir)
pred_files = os.listdir(pred_dir)

# Initialize IoU accumulator
iou_accum = np.zeros(len(classes))

# Iterate over each file
for file in gt_files:
    print(file)
    # Load ground truth and prediction images
    gt = Image.open(os.path.join(gt_dir, file))
    pred = Image.open(os.path.join(pred_dir, file))

    # Convert images to numpy arrays
    gt_arr = np.array(gt)
    pred_arr = np.array(pred)

    # Calculate IoU for each class
    for i, cls in enumerate(classes):
        # Get ground truth and prediction masks
        gt_mask = (gt_arr == cls).all(axis=2).astype(np.uint8)
        pred_mask = (pred_arr == cls).all(axis=2).astype(np.uint8)

        # Calculate intersection and union
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()

        # Check if intersection is zero
        if intersection == 0:
            iou = 0
        else:
            iou = intersection / union
            # print('iou: ', iou)

        # Accumulate IoU
        iou_accum[i] += iou



# Print IoU for each class and average IoU
for i, cls in enumerate(classes):
    print(f"{cls} IoU: {np.sum(iou_accum[i]) / len(gt_files)}")

# Calculate average IoU
avg_iou = np.sum(iou_accum) / len(gt_files) / len(classes)
print(f"Average IoU: {avg_iou}")