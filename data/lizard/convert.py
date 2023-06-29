import cv2
import os 
from tqdm import tqdm
import scipy.io as sio
import numpy as np
import multiprocessing

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


def convert_from_differ_dir(img_names):
    label_dir = "./lizard_labels/Lizard_Labels/Labels"
    for img_name in tqdm(img_names):
        label_path = os.path.join(label_dir,img_name+".mat")
        label = sio.loadmat(label_path)
        inst_map = label['inst_map'] 
        unique_values = np.unique(inst_map).tolist()[1:]
        classes = label['class']
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

        np.save(os.path.join(save_mask_path, img_name+".npy"), np_file)
        os.rename(os.path.join(img_dir, img_name+".png"),os.path.join(save_img_path, img_name+".png"))

if __name__ == '__main__':

    save_img_path = "./images/"
    save_mask_path = "./masks/"
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)
    if not os.path.exists(save_mask_path):
        os.makedirs(save_mask_path)

    img_dir1 = "./lizard_images1/Lizard_Images1"
    img_dir2 = "./lizard_images2/Lizard_Images2"
    img_dir = "./images"
    img_names = os.listdir(img_dir)
    img_names = [os.path.splitext(tmp)[0] for tmp in img_names]
    img_num = len(img_names)
    chunk_size = img_num//4
    chunks = [img_names[i:i+chunk_size] for i in range(0, img_num, chunk_size)]
    # 创建进程池，设置最大进程数为4
    pool = multiprocessing.Pool(processes=4)

    # 启动5个子进程
    for i in chunks:
        pool.apply_async(convert_from_differ_dir, args=(i,))

    # 等待所有子进程完成
    pool.close()
    pool.join()

    print('All workers completed')


