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

root_path = "kvasir-seg"
image_path = os.path.join(root_path, "images")
mask_path = os.path.join(root_path, "masks")
image_ids = os.listdir(image_path)

for file in tqdm(image_ids):
    if file.lower().endswith(".jpg"):
        img_path = os.path.join(image_path, file)
        file_name = os.path.splitext(file)[0]
        img = cv2.imread(img_path)

        targets = np.zeros((*img.shape[:2], 2))

        mask_image = cv2.imread(os.path.join(mask_path, file), cv2.IMREAD_GRAYSCALE)

        # 提取大于threshold的部分
        threshold = 100
        instance_mask = (mask_image > threshold).astype(np.uint8) * 255
        semantic_mask = (mask_image > threshold).astype(np.uint8)

        # 处理实例分割通道
        _, connected_components = cv2.connectedComponents(instance_mask)

        # 将实例分割结果存入第一个通道
        targets[..., 0] = connected_components.astype(np.uint8)

        # 处理语义分割通道
        targets[..., 1] = semantic_mask
        targets = targets.astype(int)
        os.rename(img_path, os.path.join(save_img_path, file))
        np.save(os.path.join(save_mask_path, file_name+".npy"), targets)

