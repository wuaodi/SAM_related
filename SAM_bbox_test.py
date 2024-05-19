import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

# tools
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   


# load image and prompts
image = cv2.imread('images/引言图目标域.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# load model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

predictor.set_image(image)
# 输入bbox，左上角和右下角，有多个时用逗号隔开
input_boxes = torch.tensor([
    [205, 90, 460, 341],
    [81, 48, 279, 127],
    [424, 192, 633, 395],
    [216, 131, 249, 172],
    [323, 113, 358, 145]
], device=predictor.device)

transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

# show results
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
for box in input_boxes:
    show_box(box.cpu().numpy(), plt.gca())
plt.axis('off')
plt.savefig('a.jpg',dpi=300)
plt.show()


# 保存mask
output_path = 'mask_output.jpg'
# 创建一个空白画布，用于绘制所有的掩码
canvas = np.zeros_like(image)
for i, mask in enumerate(masks):
    binary_mask = (mask > 0.5).cpu().numpy().astype(np.uint8) * 255  # 将掩码转换为二进制图像
    binary_mask = np.squeeze(binary_mask, axis=0)  # 去除多余的维度
    binary_mask_rgb = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)  # 扩展为 RGB 彩色图像
    canvas = np.maximum(canvas, binary_mask_rgb)  # 将当前掩码叠加到画布上
canvas_gray = np.mean(canvas, axis=2, keepdims=True).astype(np.uint8)  # 转换为单通道灰度图像
canvas_gray_image = Image.fromarray(canvas_gray.squeeze(), mode='L')
canvas_gray_image.save(output_path, dpi=(600,600))  # 保存结果图像
