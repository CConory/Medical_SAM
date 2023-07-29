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

# draw
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
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
            
    return  mIoU, dice
    
    


if __name__ == '__main__':
    
    torch.multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bowl_2018', help='the path of the dataset')
    parser.add_argument('--weight_path', type=str, default='../../weights/sam_vit_h_4b8939.pth', help='the path of the pre_train weight')
    parser.add_argument('--model_type', type=str, default='vit_h', help='the type of the model')
    args = parser.parse_args()

    
    # yolo pretrained model + sam infer
    dataset_root_dir = "../../data"
    dataset_name = args.dataset
    dataset_dir = os.path.join(dataset_root_dir,dataset_name)
    
    # YOLO MODEL
    # Bowl_2018
#     model = YOLO("../YOLO/pretrained/yolov8m_300e_dsbowl/weights/best.pt")
#     model = YOLO("../YOLO/pretrained/yolov8m_300e_imgsz640_dsbowl/weights/best.pt")
#     model = YOLO("../YOLO/pretrained/yolov8m_300e_imgsz320_dsbowl/weights/best.pt")

    # SegPC-2021
#     model = YOLO("/root/autodl-tmp/Medical_SAM/box_prompt/YOLO/pretrained/yolov8m_300e_imgsz640_seq2021/weights/best.pt")
#     model = YOLO("/root/autodl-tmp/Medical_SAM/box_prompt/YOLO/pretrained/yolov8m_300e_imgsz640_SegPC2021_semantic/weights/best.pt")
#     model = YOLO("/root/autodl-tmp/Medical_SAM/box_prompt/YOLO/pretrained/yolov8m_300e_imgsz640_SegPC2021_instance/weights/best.pt")
    
    # MoNuSeg
#     model = YOLO("/root/autodl-tmp/Medical_SAM/box_prompt/YOLO/pretrained/yolov8m_300e_imgsz1000_monuseg/weights/best.pt")

    # CryoNuSeg
#     model = YOLO("/root/autodl-tmp/Medical_SAM/box_prompt/YOLO/pretrained/yolov8m_300e_imgsz640_cryonuseg/weights/best.pt")

    # TNBC
#     model = YOLO("/root/autodl-tmp/Medical_SAM/box_prompt/YOLO/pretrained/yolov8m_300e_imgsz640_TNBC/weights/best.pt")

    # TNBC_CryoNuSeg_MoNuSeg
#     model = YOLO("/root/autodl-tmp/Medical_SAM/box_prompt/YOLO/pretrained/yolov8m_300e_imgsz640_TNBC_CryoNuSeg_MoNuSeg/weights/best.pt")

    # HuBMAP
#     model = YOLO("/root/autodl-tmp/Medical_SAM/box_prompt/YOLO/pretrained/yolov8m_300e_imgsz640_HuBMAP_semantic/weights/best.pt")
#     model = YOLO("/root/autodl-tmp/Medical_SAM/box_prompt/YOLO/pretrained/yolov8m_300e_imgsz512_HuBMAP_semantic/weights/best.pt")
#     model = YOLO("/root/autodl-tmp/Medical_SAM/box_prompt/YOLO/pretrained/yolov8m_300e_imgsz512_HuBMAP_instance/weights/best.pt")
#     model = YOLO("/root/autodl-tmp/Medical_SAM/box_prompt/YOLO/pretrained/yolov8m_300e_imgsz640_HuBMAP_instance/weights/best.pt")
    
    # bkai-igh-neopolyp
#     model = YOLO("/root/autodl-tmp/Medical_SAM/box_prompt/YOLO/pretrained/yolov8m_300e_imgsz640_bkai_semantic/weights/best.pt")
#     model = YOLO("/root/autodl-tmp/Medical_SAM/box_prompt/YOLO/pretrained/yolov8m_300e_imgsz1280_bkai_semantic/weights/best.pt")

    # CVC-ClinicDB
#     model = YOLO("/root/autodl-tmp/Medical_SAM/box_prompt/YOLO/pretrained/yolov8m_300e_imgsz384_CVC/weights/best.pt")
    model = YOLO("/root/autodl-tmp/Medical_SAM/box_prompt/YOLO/weights/yolov8m.pt")


    sam_checkpoint = args.weight_path
    model_type = args.model_type
    device ='cuda'
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
#     predictor = SamPredictor(sam)
    predictor = Prompt_plut_decoder(sam)
    
    
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
    
    mIoU,dice = evaluation(predictor,test_loader,device)

    test_mIoU = round(sum(mIoU)/len(mIoU),3)
    test_dice = round(sum(dice)/len(dice),3)
    print("test_mIoU: ",test_mIoU)
    print("test_Dice: ",test_dice)

    
