'''从语义标签中随机采样一些点来做SAM, 语义标签和原始图像相同名字，各放一个文件夹'''

import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
from tqdm import tqdm

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
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# Function to save mask as image
def save_mask_as_image(mask, image_path):
    mask = mask.astype(np.uint8) * 255
    mask_image = np.zeros_like(image)
    mask_image[mask > 0] = 255
    cv2.imwrite(image_path, mask_image)

# Load SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)



# Folder path containing the images
folder_path = "images/" # 图像路径
semantic_folder_path = '250k_test_output/' # 语义标签路径,用的是黑乎乎的那种png图像

# Iterate over images in the folder
for filename in tqdm(os.listdir(folder_path)):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        print(filename)
        # Read and process the image
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 获取point prompt
        # 加载语义mask图像
        mask_image = Image.open(semantic_folder_path + filename[0:-4] + '.png')
        mask = np.array(mask_image)
        solar_coords = np.argwhere(mask == 2) # 获取标签为2(solar)的像素坐标
        cabin_coords = np.argwhere(mask == 1) # 获取标签为2(cabin)的像素坐标
        background_coords = np.argwhere(mask == 0) # 获取标签为0(background)的像素坐标
        # 如果存在solar标签的像素
        if solar_coords.size > 0:
            # 从solar坐标中随机选择500个
            random_solar_coords = solar_coords[np.random.choice(solar_coords.shape[0], size=5, replace=False)] # 当 replace=True 时，表示可以重复选择相同的元素
            random_cabin_coords = cabin_coords[np.random.choice(cabin_coords.shape[0], size=5, replace=False)]
            random_background_coords = background_coords[np.random.choice(background_coords.shape[0], size=5, replace=False)]
            # 将random_solar_coords转换为List并反转坐标的x和y
            random_solar_coords_list = [[y, x] for x, y in random_solar_coords]
            random_cabin_coords_list = [[y, x] for x, y in random_cabin_coords]
            random_background_coords_list = [[y, x] for x, y in random_background_coords]
        else:
            print("No solar pixels found in the mask image.")
            continue
        combined_coords_list = random_solar_coords_list + random_cabin_coords_list + random_background_coords_list
        input_point = np.array(combined_coords_list)
        input_label = np.concatenate((np.ones(len(random_solar_coords_list)), np.zeros(len(random_cabin_coords_list)+len(random_background_coords_list))))

        # Perform image segmentation
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )

        # Display and save the results，这个循环没啥用，就一个mask其实
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()

            # Save the mask as an image
            save_mask_as_image(mask, 'mask/'+f'{filename}')