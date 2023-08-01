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

color_mapping = {
    "green": 1,  # 绿色映射为1 non-neoplastic polyps
    "red": 2   # 红色映射为2 neoplastic polyps
}

root_path = "./bkai-igh-neopolyp"
train_path = os.path.join(root_path, "train", "train")
mask_path = os.path.join(root_path, "train_gt", "train_gt")
train_ids = os.listdir(train_path)

for file in tqdm(train_ids):
    if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
        img_path = os.path.join(train_path, file)
        file_name = os.path.splitext(file)[0]
        img = cv2.imread(img_path)

        targets = np.zeros((*img.shape[:2], 2))

        mask_image = cv2.imread(os.path.join(mask_path, file), cv2.IMREAD_COLOR)

        # 提取红色或绿色部分作为掩码
        threshold = 100
        red_mask = mask_image[..., 2] > threshold
        green_mask = mask_image[..., 1] > threshold

        # 将掩码映射为标识符
        red_mask_mapped = red_mask.astype(np.uint8) * color_mapping["red"]
        green_mask_mapped = green_mask.astype(np.uint8) * color_mapping["green"]

        # 合并红色和绿色掩码
        combined_mask = np.maximum(red_mask_mapped, green_mask_mapped)

        # 处理实例分割通道
        _, connected_components = cv2.connectedComponents(combined_mask)

        # 将实例分割结果存入第一个通道
        targets[..., 0] = connected_components.astype(np.uint8)

        # 处理语义分割通道
        targets[..., 1] = combined_mask

        targets = targets.astype(int)
        os.rename(img_path, os.path.join(save_img_path, file))
        np.save(os.path.join(save_mask_path, file_name+".npy"), targets)

