import os
from tqdm import tqdm
import numpy as np
import cv2

save_img_path = "./images/"
save_mask_path = "./masks/"
if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)
if not os.path.exists(save_mask_path):
    os.makedirs(save_mask_path)

import os

dataset_path = "TNBC"

# 遍历Slide文件夹和GT文件夹
for slide_folder, gt_folder in zip(range(1, 12), range(1, 12)):
    slide_folder_name = f"Slide_{slide_folder:02d}"
    gt_folder_name = f"GT_{gt_folder:02d}"

    slide_folder_path = os.path.join(dataset_path, slide_folder_name)
    gt_folder_path = os.path.join(dataset_path, gt_folder_name)

    # 获取Slide文件夹中的所有图像文件
    slide_images = os.listdir(slide_folder_path)

    # 遍历Slide文件夹中的图像文件
    for slide_image in slide_images:
        image_name, image_ext = os.path.splitext(slide_image)

        # 构建对应的mask文件名
        mask_name = image_name + ".png"

        # 构建图像和mask的完整路径
        slide_image_path = os.path.join(slide_folder_path, slide_image)
        mask_path = os.path.join(gt_folder_path, mask_name)

        img = cv2.imread(slide_image_path)
        targets = np.zeros((*img.shape[:2], 2))

        mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        instance_idx = mask_ == 255
        # 计算连通区域
        num_labels, labels = cv2.connectedComponents(instance_idx.astype(np.uint8)*255)
        # 给每个连通区域分配不同的实例ID
        for label in range(1, num_labels):
            targets[..., 0][labels == label] = label
        targets[instance_idx,1] = 1
        targets = targets.astype(int)
        os.rename(slide_image_path, os.path.join(save_img_path, "TNBC_"+image_name + ".png"))
        np.save(os.path.join(save_mask_path, "TNBC_"+image_name + ".npy"), targets)
