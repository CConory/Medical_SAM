import cv2
import numpy as np
import sys
from segment_anything import sam_model_registry, SamPredictor
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json
from evaluate_from_pt import show_mask,show_box,mean_iou_and_dice
import wandb

def evaluate(predictor,img_paths,max_vis=10):

    mIoU = []
    dice = []
    vis_results = []
    vis_count = 0
    with torch.no_grad():
        for img_path in tqdm(img_paths):
            ipt= cv2.imread(img_path)
            ipt = cv2.cvtColor(ipt, cv2.COLOR_BGR2RGB)

            mask = np.load(img_path.replace("/images/","/masks/").replace(".png",".npy"),allow_pickle=True)

            semantic_mask = mask[...,1]!=0
            instances_mask = mask[...,0]
            max_instance_nums = np.max(instances_mask)
            instance_bboxes = []
            for instance_id in range(1,max_instance_nums+1):
                instance =  (instances_mask==instance_id).astype(np.uint8)*255
                c1 = cv2.boundingRect(instance)
                instance_bboxes.append([c1[0], c1[1], c1[0]+c1[2], c1[1]+c1[3]])
            
            # Visualization
            # plt.figure(figsize=(10,10))
            # plt.imshow(ipt)
            # for box in instance_bboxes:
            #     show_box(box, plt.gca())
            # plt.axis('off')
            # plt.savefig("./tmp.jpg")


            # 一类一类地推理画mask

            if len(instance_bboxes):
                input_boxes = torch.tensor(instance_bboxes, device=predictor.device)

                predictor.set_image(ipt)

                transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, ipt.shape[:2])

                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
            else:
                masks = torch.zeros((1,1,*ipt.shape[:2]))
            

            pred_fg = torch.sum(masks,dim=0).bool().cpu().numpy()[0]
            _mIoU,_dice = mean_iou_and_dice(semantic_mask[None][None],pred_fg[None][None])
            mIoU.append(_mIoU)
            dice.append(_dice)

            if vis_count< max_vis:
                target_fg = semantic_mask
                inter_fg = pred_fg & target_fg
                target_fg = (~inter_fg) & target_fg
                pred_fg = (~inter_fg) & pred_fg

                plt.figure(figsize=(10,10))
                plt.imshow(ipt)
                show_mask(pred_fg, plt.gca(),np.array([0/255, 0/255, 255/255, 0.4]))
                show_mask(target_fg, plt.gca(),np.array([0/255, 255/255, 0/255, 0.4]))
                show_mask(inter_fg, plt.gca(),np.array([255/255, 255/255, 0/255, 0.4]))
                for box in instance_bboxes:
                    show_box(box, plt.gca())
                plt.axis('off')
                vis_results.append(wandb.Image(plt))
            vis_count += 1
    return  mIoU, dice, vis_results
            


wandb_flag = False  
dataset_name = "PanNuke"
prompt_type = "All_Boxes"
if wandb_flag:
    wandb.init(project="Medical_SAM",config={
        "dataset": dataset_name,
        "prompt": prompt_type
    })
    wandb.run.name = wandb.run.id
    wandb.run.save()

sam_checkpoint = "/userhome/cs2/kuangww/segment-anything/notebooks/models/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)


with open("./datasets/PanNuke/data_split.json", 'r') as f:
    ds_dict = json.load(f)
valid_names = ds_dict["valid"]
test_names = ds_dict["test"]


img_paths = [os.path.join("./datasets/PanNuke/images/",tmp+".png") for tmp in valid_names]
mIoU,dice, valid_vis_results = evaluate(predictor,img_paths)

valid_mIoU = round(sum(mIoU)/len(mIoU),3)
valid_dice = round(sum(dice)/len(dice),3)
print("valid_mIoU: ",valid_mIoU)
print("valid_Dice: ",valid_dice)

img_paths = [os.path.join("./datasets/PanNuke/images/",tmp+".png") for tmp in test_names]
mIoU,dice, test_vis_results = evaluate(predictor,img_paths)

test_mIoU = round(sum(mIoU)/len(mIoU),3)
test_dice = round(sum(dice)/len(dice),3)
print("test_mIoU: ",test_mIoU)
print("test_Dice: ",test_dice)


if wandb_flag:
    wandb.log({
        "valid_results": valid_vis_results,
        "test_results": test_vis_results,
        "valid/mIoU":valid_mIoU,
        "valid/dice":valid_dice,
        "test/mIoU":test_mIoU,
        "test/dice":test_dice
        })
    






