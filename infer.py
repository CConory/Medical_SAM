import cv2
import numpy as np
import sys
from segment_anything import sam_model_registry, SamPredictor
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.8])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

df = pd.read_csv('segpc2021/test_train_data.csv')
imgs = df[df.category!='train']['image_id']

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

with torch.no_grad():

    for img_name in tqdm(imgs):
        img_path = f'segpc2021/data/images/x/{img_name}'
        ipt= cv2.imread(img_path)
        ipt = cv2.cvtColor(ipt, cv2.COLOR_BGR2RGB)

        img_id = list(img_name.split('.'))[0]
        masks = os.listdir(f'segpc2021/data/images/y/{img_id}/')

        core_bboxes = []
        cell_bboxes = []

        for e_mask in masks:
            mask_path = f'segpc2021/data/images/y/{img_id}/{e_mask}'

            mask_img = cv2.imread(mask_path, 0)
            img_core = np.copy(mask_img)
            img_core[img_core==20] = 0
            img_cell = np.copy(mask_img)
            img_cell[img_cell==40] = 20
    
            contours1, _ = cv2.findContours(img_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c1 = cv2.boundingRect(contours1[0])

            contours2, _ = cv2.findContours(img_cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c2 = cv2.boundingRect(contours2[0])

            core_bboxes.append([c1[0], c1[1], c1[0]+c1[2], c1[1]+c1[3]])
            cell_bboxes.append([c2[0], c2[1], c2[0]+c2[2], c2[1]+c2[3]])

            del img_core, img_cell
    
        plt.figure(figsize=(10, 10))
        plt.imshow(ipt)

        ## core/nuclei

        input_core_boxes = torch.tensor(core_bboxes, device=predictor.device)

        predictor.set_image(ipt)

        transformed_boxes = predictor.transform.apply_boxes_torch(input_core_boxes, ipt.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # for mask in masks:
        #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box in input_core_boxes:
        #     show_box(box.cpu().numpy(), plt.gca())

        core_zeros = torch.zeros(masks.shape[-2:]) #h, w

        for i in range(masks.shape[0]): #number of objects 

            pred = masks[i,0,:,:]
            core_zeros[pred==True]=40
        
        core_save_path = f'segpc2021/bbox/mask/nuclei/{img_id}.png'
        cv2.imwrite(core_save_path, core_zeros.numpy())


        input_cell_boxes = torch.tensor(cell_bboxes, device=predictor.device)

        transformed_boxes = predictor.transform.apply_boxes_torch(input_cell_boxes, ipt.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # for mask in masks:
        #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box in input_cell_boxes:
        #     show_box(box.cpu().numpy(), plt.gca())

        cell_zeros = torch.zeros(masks.shape[-2:]) #h, w

        for i in range(masks.shape[0]): #number of objects 

            pred = masks[i,0,:,:]
            cell_zeros[pred==True] = 20
        
        cell_save_path = f'segpc2021/bbox/mask/cell/{img_id}.png'
        cv2.imwrite(cell_save_path, cell_zeros.numpy())

        mix_zeros = core_zeros + cell_zeros
        mix_zeros[mix_zeros > 20] = 40
        mix_save_path = f'segpc2021/bbox/mask/mix/{img_id}.png'
        cv2.imwrite(mix_save_path, mix_zeros.numpy())

        # plt.axis('off')
        # plt.savefig(f'segpc2021/bbox/plot/{img_id}.png')

        del input_core_boxes, transformed_boxes, core_zeros, input_cell_boxes, masks, cell_zeros, mix_zeros
        torch.cuda.empty_cache()
    






