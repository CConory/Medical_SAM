import os
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch 

ROOT_PATH = './datasets/'
DATANAME = "bowl_2018"


# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30, 144, 255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     return mask_image

if __name__ == '__main__':
    TRAIN_PATH = os.path.join(ROOT_PATH,DATANAME)
    train_ids = next(os.walk(TRAIN_PATH))[1]

    sam_checkpoint = "/userhome/cs2/kuangww/segment-anything/notebooks/models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) # 一个网络结构
    sam=sam.to(device=device)

    predictor = SamPredictor(sam) 

    for file_name in tqdm(train_ids):
        path = os.path.join(TRAIN_PATH,file_name)
        img = cv2.imread(path + '/images/' + file_name + '.png')
        output_dir = os.path.join(path,'features')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
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
        torch.save(data,os.path.join(output_dir,file_name+".pt"))

        predictor.reset_image()

