import cv2
import os 
from tqdm import tqdm
import scipy.io as sio
import numpy as np
import multiprocessing

def convert_from_differ_dir(data_dir):
    img_dir = data_dir + "/Images"
    label_dir = data_dir + "/Labels"
    img_names = os.listdir(img_dir)
    img_names = [os.path.splitext(tmp)[0] for tmp in img_names]
    for img_name in tqdm(img_names):
        label_path = os.path.join(label_dir,img_name+".mat")
        label = sio.loadmat(label_path)
        inst_map = label['inst_map'] 
        unique_values = np.unique(inst_map).tolist()[1:]
        classes = label['inst_type']
        np_file = np.zeros((*inst_map.shape[:2],2), dtype='int16')

        
        new_instance_id = 1
        for index, ins_id in enumerate(unique_values):
            instance_mask = inst_map==ins_id

            np_file[...,0][instance_mask] = ins_id
            np_file[...,1][instance_mask] = classes[index]

            num_labels, labels = cv2.connectedComponents(instance_mask.astype(np.uint8)*255) #计算联通域，instance——label的误差，导致同一个instance_id对应多个obj
            if num_labels>2:
                for i in range(2,num_labels):
                    np_file[...,0][labels==i] = unique_values[-1]+new_instance_id
                    np_file[...,1][labels==i] = classes[index]
                    new_instance_id +=1

        np.save(os.path.join(save_mask_path,"CoNSeP_"+img_name+".npy"), np_file)
        os.rename(os.path.join(img_dir,img_name+".png"),os.path.join(save_img_path,"CoNSeP_"+img_name+".png"))

if __name__ == '__main__':

    save_img_path = "./images/"
    save_mask_path = "./masks/"
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)
    if not os.path.exists(save_mask_path):
        os.makedirs(save_mask_path)

    data_dir1 = "./CoNSeP/CoNSeP/Train"
    data_dir2 = "./CoNSeP/CoNSeP/Test"

    convert_from_differ_dir(data_dir1)
    convert_from_differ_dir(data_dir2)

    print('All workers completed')


