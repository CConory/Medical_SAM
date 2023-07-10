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

root_path = "./GlaS"

file_ids = os.listdir(root_path)

for file in tqdm(file_ids):
    if file.lower().endswith(".bmp"):
        # 提取图像文件名（不包含文件扩展名）
        image_name = os.path.splitext(file)[0]
        # 构建标注文件名
        annotation_name = image_name + "_anno.bmp"
        # 检查标注文件是否存在
        annotation_path = os.path.join(root_path, annotation_name)

        if os.path.isfile(annotation_path):
            img_path = os.path.join(root_path, file)
            img = cv2.imread(img_path)
            targets = np.zeros((*img.shape[:2], 2))

            mask_image = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)

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
            np.save(os.path.join(save_mask_path, image_name+".npy"), targets)

