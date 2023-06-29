import os
import cv2
import numpy as np
import os
from tqdm import tqdm

"""
    check 一下有没有重复的
"""
# train_names = os.listdir("./train/x")
# valid_names = os.listdir("./validation/x")

# inter = [tmp for tmp in valid_names if tmp in train_names]
# print(len(train_names))
# print(len(valid_names))
# print(inter)


save_img_path = "./images/"
save_mask_path = "./masks/"
if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)
if not os.path.exists(save_mask_path):
    os.makedirs(save_mask_path)

def convert_from_different(root_dir):
    img_dir = os.path.join(root_dir,"x")
    img_names = os.listdir(img_dir)
    img_names = [os.path.splitext(tmp)[0] for tmp in img_names ]
    target_dir = os.path.join(root_dir,"y")
    target_names = os.listdir(target_dir)

    for img_name in tqdm(img_names):
        img_path = os.path.join(img_dir,img_name+".bmp")

        img = cv2.imread(img_path)
        this_img_target_names = [tmp for tmp in target_names if tmp.split("_")[0]==img_name]

        np_file = np.zeros((*img.shape[:2],2), dtype='int16')
        new_instance_id = 1
        for instance_id,target_name in enumerate(this_img_target_names):
            target_path = os.path.join(target_dir,target_name)
            mask = cv2.imread(target_path, 0) # np.unique(mask) = [0,20,40]
            np_file[...,0][mask!=0] = instance_id+1
            np_file[...,1][mask==20] = 1 #Core
            np_file[...,1][mask==40] = 2
            temp_mask = mask!=0
            num_labels, labels = cv2.connectedComponents(temp_mask.astype(np.uint8)*255) #计算联通域，instance——label的误差，导致同一个instance_id对应多个obj
            if num_labels>2:
                for i in range(2,num_labels):
                    np_file[...,0][labels==i] = len(this_img_target_names)+new_instance_id
                    new_instance_id += 1

        np.save(os.path.join(save_mask_path,img_name+".npy"), np_file) 
        os.rename(img_path,os.path.join(save_img_path,img_name+".bmp"))


val_dir = "./validation"
train_dir = "./train"
convert_from_different(val_dir)
convert_from_different(train_dir)
