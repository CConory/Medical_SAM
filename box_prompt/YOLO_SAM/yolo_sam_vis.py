import torch
import torchvision
import sys
from ultralytics import YOLO
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import json
from dataset import Box_prompt_Dataset
from Prompt_plut_decoder import Prompt_plut_decoder
import argparse
import os
from tqdm import tqdm
import wandb

# draw
def show_mask(mask, ax, color):
    # if random_color:
    #     color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    # else:
    #     color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=50):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax,color='green'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))

    
# metrics
def calculate_iou(y_true, y_pred):
    '''Calculate Intersection over Union (IOU) for a given class'''
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
#     show_mask(intersection, plt.gca(), random_color=True)
#     plt.show()
#     show_mask(union, plt.gca(), random_color=True)
#     plt.show()
    return iou_score

def dice_score(y_true, y_pred):
    '''Calculate Dice score for a given class'''
    intersection = np.sum(y_true * y_pred)
    dice_score = (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))
    return dice_score


def mean_iou_and_dice(y_true,y_pred): #[category_nums,1,w,h]
    num_classes = y_true.shape[0]
    iou_scores = []
    dice_scores = []
    for class_idx in range(num_classes):
        y_true_class = y_true[class_idx]
        y_pred_class = y_pred[class_idx]
        if np.sum(y_true_class) == 0:
            if np.sum(y_pred_class) == 0:
                iou_scores.append(1.0)
                dice_scores.append(1.0)
            else:
                iou_scores.append(0.0)
                dice_scores.append(0.0)
        else:
            iou_scores.append(calculate_iou(y_true_class, y_pred_class))
            dice_scores.append(dice_score(y_true_class, y_pred_class))
    mean_iou_score = np.mean(iou_scores)
    ean_dice_score = np.mean(dice_scores)
    return mean_iou_score,ean_dice_score


def evaluation(model,dataloader,device,max_vis=10):
    '''
        based on the features embedding;
        inference include the prompt encoder and masks decoder.
        return:
            mIoU,Dice
            visualization result
                Blue for prediction
                Green for target
                Yellow for intersection
    '''

    mIoU = []
    dice = []
    vis_results = []
    vis_count = 0
    for batched_input,masks,image_paths in tqdm(dataloader):
        for batch_id in range(len(batched_input)):
            batched_input[batch_id]['feature'] = batched_input[batch_id]['feature'].to(device)
        # batched_input = [
        #     {
        #         "feature":feature,
        #         'original_size':data["origin_shape"][:2]
        #     }
        # ]
        # masks.shape : (BS, category_num, 1, width, height)
        batched_output = predictor.predict(batched_input, multimask_output=False)

        for batch_id in range(len(batched_output)):
            instance_bboxes = batched_input[batch_id].get("boxes", None)
            mask_output = batched_output[batch_id]['masks']
            mask_output = mask_output!=0
            mask_output = mask_output.cpu().numpy()
            
            if instance_bboxes is not None and not len(instance_bboxes): 
                mask_output = np.zeros_like(mask_output)


            target = masks[batch_id]
            _mIoU,_dice = mean_iou_and_dice(target,mask_output) # bowl-2018 no points should do ~ operation

            mIoU.append(_mIoU)
            dice.append(_dice)
            
            
            # visualization
            if vis_count< max_vis:
                img = cv2.imread(image_paths[batch_id])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                pred_fg = mask_output[0][0]!=0
                target_fg = target[0][0]!=0
                inter_fg = (mask_output[0][0]!=0) & (target[0][0]!=0)
                target_fg = (~inter_fg) & target_fg
                pred_fg = (~inter_fg) & pred_fg

                plt.figure(figsize=(10,10))
                plt.imshow(img)
                show_mask(pred_fg, plt.gca(),np.array([0/255, 0/255, 255/255, 0.4]))
                show_mask(target_fg, plt.gca(),np.array([0/255, 255/255, 0/255, 0.4]))
                show_mask(inter_fg, plt.gca(),np.array([255/255, 255/255, 0/255, 0.4]))
                if batched_input[batch_id].get("point_coords", None) is not None:
                    show_points(batched_input[batch_id]['point_coords'], batched_input[batch_id]['point_labels'], plt.gca())
                if instance_bboxes is not None:
                    for boxes in instance_bboxes:
                        for box in boxes:
                            show_box(box, plt.gca())
                plt.axis('off')
                # show_points(input_point, input_label, plt.gca())

                # img[pred_fg] = (img[pred_fg]*0.6 + tuple(tmp*0.4 for tmp in (255,0,0))).astype(np.uint8) # Blue for prediction
                # img[target_fg] = (img[target_fg]*0.6 + tuple(tmp*0.4 for tmp in (0,255,0))).astype(np.uint8) # Green for target
                # img[inter_fg] = (img[inter_fg]*0.6 + tuple(tmp*0.4 for tmp in (0,255,255))).astype(np.uint8) # Yellow for intersection
                vis_results.append(wandb.Image(plt))
            vis_count += 1
            
    return  mIoU, dice, vis_results
    
    


if __name__ == '__main__':
    
    torch.multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bowl_2018', help='the path of the dataset')
    parser.add_argument('--weight_path', type=str, default='../../weights/sam_vit_h_4b8939.pth', help='the path of the pre_train weight')
    parser.add_argument('--model_type', type=str, default='vit_h', help='the type of the model')
    parser.add_argument('--wandb_log', action='store_true', help='save the result to wandb or not')
    args = parser.parse_args()
    
    
    wandb_flag = args.wandb_log
    
    YOLO_SAM = "YOLO + SAM"

    
    # yolo pretrained model + sam infer
    dataset_root_dir = "../../data"
    dataset_name = args.dataset
    dataset_dir = os.path.join(dataset_root_dir,dataset_name)
    
    # yolo model
    model = YOLO("../YOLO/pretrained/yolov8m_300e_dsbowl/weights/best.pt")

    sam_checkpoint = args.weight_path
    model_type = args.model_type
    device ='cuda'
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
#     predictor = SamPredictor(sam)
    predictor = Prompt_plut_decoder(sam)
    
    
    if wandb_flag:
        wandb.init(project="Medical_SAM",config={
            "dataset": dataset_name,
            "box_prompt": YOLO_SAM
        })
        wandb.run.name = wandb.run.id
        wandb.run.save()
    
    
    # Load dataset
    test_dataset = Box_prompt_Dataset(os.path.join(dataset_dir,"data_split.json"),"test",device,model)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
        collate_fn = Box_prompt_Dataset.collate_fn,
        drop_last=False
    )
    
    mIoU,dice, test_vis_results= evaluation(predictor,test_loader,device)

    test_mIoU = round(sum(mIoU)/len(mIoU),3)
    test_dice = round(sum(dice)/len(dice),3)
    print("test_mIoU: ",test_mIoU)
    print("test_Dice: ",test_dice)
    
    
    if wandb_flag:
        wandb.log({
            "test_results": test_vis_results,
            "test/mIoU":test_mIoU,
            "test/dice":test_dice
            })

    
