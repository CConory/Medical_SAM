import cv2
import os 
from tqdm import tqdm
import scipy.io as sio
import numpy as np

"""
    Combine images from different files
    check the duplicate name
"""

# img_dir1 = "./lizard_images1/Lizard_Images1" 
# img_dir2 = "./lizard_images2/Lizard_Images2"
# img_names1 = os.listdir(img_dir1)
# img_names2 = os.listdir(img_dir2)
# print(len(img_names1))
# print(len(img_names2))
# print([tmp for tmp in img_names2 if tmp in img_names1])

save_img_path = "./images/"
save_mask_path = "./masks/"
if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)
if not os.path.exists(save_mask_path):
    os.makedirs(save_mask_path)

img_dir1 = "./lizard_images1/Lizard_Images1" 
img_dir2 = "./lizard_images2/Lizard_Images2"

def convert_from_differ_dir(img_dir):
    img_names = os.listdir(img_dir)
    img_names = [os.path.splitext(tmp)[0] for tmp in img_names]
    label_dir = "./lizard_labels/Lizard_Labels/Labels"
    for img_name in tqdm(img_names):
        label_path = os.path.join(label_dir,img_name+".mat")
        label = sio.loadmat(label_path)
        inst_map = label['inst_map'] 
        unique_values = np.unique(inst_map).tolist()[1:]
        classes = label['class']
        np_file = np.zeros((*inst_map.shape[:2],2), dtype='int16')

        np_file[...,0] = inst_map
        for index, ins_id in enumerate(unique_values):
            instance_mask = inst_map==ins_id
            np_file[...,1][instance_mask] = classes[index]
        np.save(os.path.join(save_mask_path,img_name+".npy"), np_file) 
        os.rename(os.path.join(img_dir,img_name+".png"),os.path.join(save_img_path,img_name+".png"))

convert_from_differ_dir(img_dir1)
convert_from_differ_dir(img_dir2)
