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

root_path = "./stage1_train"
train_ids = next(os.walk(root_path))[1]

for file_name in tqdm(train_ids):
    path = os.path.join(root_path,file_name)
    img_path = path + '/images/' + file_name + '.png'
    img = cv2.imread(img_path)

    targets = np.zeros((*img.shape[:2],2))

    instance_id = 0
    for _, mask_file in enumerate(next(os.walk(path + '/masks/'))[2]):
        instance_id += 1
        mask_ = cv2.imread(path + '/masks/' + mask_file,-1)
        instance_idx = mask_ == 255
        targets[instance_idx,0] = instance_id
        targets[instance_idx,1] = 1
        num_labels, labels = cv2.connectedComponents(instance_idx.astype(np.uint8) * 255)  # 计算联通域，instance——label的误差，导致同一个instance_id对应多个obj
        if num_labels > 2:
            for i in range(2, num_labels):
                instance_id += 1
                targets[..., 0][labels == i] = instance_id
                targets[..., 1][labels == i] = 1
    
    targets = targets.astype(int)
    os.rename(img_path,os.path.join(save_img_path, file_name+".png"))
    np.save(os.path.join(save_mask_path, file_name+".npy"), targets)

