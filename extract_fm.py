import os
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch 

import argparse




# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30, 144, 255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     return mask_image

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PanNuke', help='the path of the dataset')
    parser.add_argument('--weight_path', type=str, default='/userhome/cs2/kuangww/segment-anything/notebooks/models/sam_vit_h_4b8939.pth', help='the path of the pre_train weight')
    parser.add_argument('--model_type', type=str, default='vit_h', help='the type of the model')
    args = parser.parse_args()

    ROOT_PATH = './datasets/'
    DATANAME = args.dataset

    TRAIN_PATH = os.path.join(ROOT_PATH,DATANAME)
    OUTPUT_DIR = os.path.join(TRAIN_PATH,"features")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_dir = os.path.join(TRAIN_PATH,"images")
    image_names = os.listdir(image_dir)

    sam_checkpoint = args.weight_path
    model_type = args.model_type

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) # 一个网络结构
    sam=sam.to(device=device)

    predictor = SamPredictor(sam) 

    for file_name in tqdm(image_names):
        suffix = os.path.splitext(file_name)[1]
        path = os.path.join(TRAIN_PATH,file_name)
        img = cv2.imread(os.path.join(image_dir,file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #BGR2RGB

        predictor.set_image(img)

        feature = predictor.features.cpu()

        origin_shape = img.shape
        data = {
            "fm":feature,
            "origin_shape":origin_shape
        }

        # masks, scores, logits = predictor.predict(multimask_output=False)
        # import pdb;pdb.set_trace()
        torch.save(data,os.path.join(OUTPUT_DIR,file_name.replace(suffix,".pt")))

        predictor.reset_image()

