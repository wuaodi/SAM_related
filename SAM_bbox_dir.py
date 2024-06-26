"""SAM_bbox.py，可以按照类别来赋予颜色，相同的类别赋予mask相同的颜色，不同的类别赋予不同的颜色"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import os
import time

# config
image_dir = "images"
yolov5_bbox_dir = "bbox"
pred_mask_dir = "mask"
vis_dir = "vis"

# tools
def show_mask(mask, ax, class_id):
    print('class_id: ', class_id)
    if class_id == 0:
        color = np.array([255/255, 0, 0, 1.0])  # Red，最后一个通道控制不透明程度的，0.6半透明，1完全不透明
    if class_id == 1:
        color = np.array([0, 0, 255/255, 1.0])  # Blue
    if class_id == 2:
        color = np.array([0, 255/255, 0, 1.0])  # Green
    if class_id == 3:
        color = np.array([0, 0, 0, 1.0])  # Black

    # color = np.array([30/255, 144/255, 255/255, 0.6]) # 固定颜色
    # color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) # 随机颜色
    print('color: ', color)

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

# load model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"    # vit_h, vit_l, vit_b
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

for image_name in os.listdir(image_dir):
    print("-----------------------------------------")
    
    image_path = os.path.join(image_dir, image_name)
    mask_path = os.path.join(pred_mask_dir, image_name)
    bbox_path = os.path.join(yolov5_bbox_dir, image_name[0:-4]+'.txt')
    vis_path = os.path.join(vis_dir, image_name)

    # load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("image: ", image_path)
    
    # 如果检测框不存在，创建一个和图片大小一样的全黑的预测结果
    if not os.path.exists(bbox_path):
        mask_image = np.zeros_like(image)
        mask_image_gray = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask_image_pil = Image.fromarray(mask_image, mode='RGB')
        mask_image_pil.save(mask_path)  # 保存结果图像
        print("bbox doesn't exist")
        continue
    
    # load yolov5 bbox
    with open(bbox_path, 'r') as f:
        # Process bbox coordinates
        bboxes = []
        class_ids = []  # 存储class_id的列表
        for line in f:
            # 解析标注信息
            class_id, center_x, center_y, width, height = map(float, line.strip().split())
            # 计算边界框的左上角和右下角坐标
            x = int((center_x - width / 2) * image.shape[1])
            y = int((center_y - height / 2) * image.shape[0])
            w = int(width * image.shape[1])
            h = int(height * image.shape[0])
            bboxes.append([x, y, x+w, y+h])
            class_ids.append(int(class_id))  # 将class_id添加到列表中
    
    # SAM input bbox output mask
    begin = time.time()
    predictor.set_image(image)
    input_boxes = torch.tensor(bboxes, device=predictor.device)

    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    print("time consuming: ", time.time()-begin, "s")

    # visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, (mask, class_id, box) in enumerate(zip(masks, class_ids, input_boxes)):
        show_mask(mask.cpu().numpy(), plt.gca(), class_id=class_id)  # Pass the class_id to show_mask
        show_box(box.cpu().numpy(), plt.gca())  # Show the corresponding bounding box

    plt.axis('off')
    plt.savefig(vis_path)
    # plt.show()

    # 保存mask
    mask_image = np.zeros_like(image)
    for i, mask in enumerate(masks):
        binarymask = (mask > 0.5).cpu().numpy().astype(np.uint8)
        class_id = class_ids[i]
        color = np.array([0, 0, 0])  # Black
        if class_id == 0:
            color = np.array([255, 0, 0])  # Red
        elif class_id == 1:
            color = np.array([0, 0, 255])  # Blue
        elif class_id == 2:
            color = np.array([0, 255, 0])  # Green
        
        mask_np = mask.cpu().numpy()
        mask_image[mask_np[0] == 1] = color
    
    mask_image_pil = Image.fromarray(mask_image, mode='RGB')
    mask_image_pil.save(mask_path)  # 保存结果图像
    print("Saved predicted mask:", mask_path)