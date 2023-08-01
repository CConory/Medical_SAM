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

imgs_path = "./NuCLS/rgbs_colorNormalized"

masks_path = "./NuCLS/NuCLS_masks"

img_names = os.listdir(imgs_path)

for img_name in tqdm(img_names):
    if img_name.endswith('.png'):
        path = os.path.join(imgs_path, img_name)
        image_name, image_ext = os.path.splitext(img_name)
        img = cv2.imread(path)

        targets = np.zeros((*img.shape[:2],2))

        # 构建对应的mask文件名
        mask_name = image_name + ".png"
        mask_path = os.path.join(masks_path, mask_name)

        mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        unique_values = np.unique(mask_).tolist()[1:]

        instance_id = 0
        for i in unique_values:
            instance_idx = mask_==i
             # 计算连通区域
            num_labels, labels = cv2.connectedComponents(instance_idx.astype(np.uint8))
            # 给每个连通区域分配不同的实例ID
            for label in range(1, num_labels):
                instance_id += 1
                targets[..., 0][labels == label] = instance_id

            targets[..., 1][mask_ == i] = i
        targets = targets.astype(int)

        os.rename(path, os.path.join(save_img_path, "NuCLS_"+image_name + ".png"))
        np.save(os.path.join(save_mask_path, "NuCLS_"+image_name + ".npy"), targets)

