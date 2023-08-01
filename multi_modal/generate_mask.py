import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import groundingdino.datasets.transforms as T
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from groundingdino.util.slconfig import SLConfig
from multi_modal.models.groundingdino_v1 import build_groundingdino
from groundingdino.util.misc import nested_tensor_from_tensor_list
from groundingdino.util import get_tokenlizer, box_ops
from Grounding_dino_infer import PostProcessGrounding,load_model
from finetune_GDINO import select_predn_mask
from evaluate_from_pt import mean_iou_and_dice
from segment_anything import  sam_model_registry,SamPredictor


import argparse
import json
from PIL import Image
import cv2
import torch
import numpy as np
from tqdm import tqdm
import wandb
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco', help='the path of the dataset')
    parser.add_argument('--dataset_type', type=str, default='valid', help='test set or valid set')
    parser.add_argument('--model_type', type=str, default='grounding_dino', help='the type of the model')
    parser.add_argument('--mAP_threshold', type=float, default=0.5, help='the threshold that pedictio is corrected')
    parser.add_argument('--wandb_log', action='store_true', help='save the result to wandb or not')
    args = parser.parse_args()

    device = "cuda"

    wandb_id = "fallen-sky-451"
    threshold = 0.36
    checkpoint_path = "/userhome/cs2/kuangww/medical_sam/multi_modal/weights/yek3xgo2/best.pth"
    category_id = {
        "HuBMAP": 1,
        "bowl_2018" :1,
        "SegPC-2021": 2
    }
    cat_list = ["plasma cell core"]
    caption_list = ["plasma cell core"]


    mask_save_dir ="./results/"
    mask_save_dir = os.path.join(mask_save_dir,wandb_id)

    if args.wandb_log:
        wandb_run =  wandb.init(id=wandb_id,
                            project="Medical_SAM",
                            resume='allow',
                            allow_val_change=True)

    dataset_root_dir = "../data"
    dataset_name = args.dataset
    dataset_dir = os.path.join(dataset_root_dir,dataset_name)
    json_path = os.path.join(dataset_dir,"data_split.json")
    split = args.dataset_type
    with open(json_path, 'r') as f:
        ds_dict = json.load(f)
    dataset_dir = os.path.dirname(json_path)
    file_names = ds_dict[split]
    predic_masks_dir_name = ["label", "pred_bbox","pred_mask","pred_sam_box"]
    for dir_name in predic_masks_dir_name:
        save_dir = os.path.join(mask_save_dir,dir_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    config_file_path = "./config/GroundingDINO_SwinT_OGC.py"
    
    cfg = SLConfig.fromfile(config_file_path)
    model = load_model(config_file_path, checkpoint_path)
    model = model.to(device)
    model = model.eval()
    
    caption = " . ".join(caption_list) + ' .'
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessGrounding(
        category_list=cat_list, tokenlizer=tokenlizer)

    dino_transformer = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
        ]
    )


    # For SAM
    sam_checkpoint = "/userhome/cs2/kuangww/segment-anything/notebooks/models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    # sam_transformer = 

    
    bbox_iou = []
    bbox_dice = []

    mask_iou = []
    mask_dice = []

    sam_bbox_iou = []
    sam_bbox_dice = []


    for file_name in tqdm(file_names):
        suffix = os.path.splitext(file_name)[1]
        img_name = os.path.splitext(file_name)[0]
        img_path = os.path.join(dataset_dir,"images",file_name)
        target_path = os.path.join(dataset_dir,"masks",file_name.replace(suffix,".npy"))

        image = Image.open(img_path).convert('RGB')
        w,h = image.size
        img_size = torch.tensor([h, w])
        mask = np.load(target_path,allow_pickle=True)
        mask = mask.astype(np.int32)
        cate_id = category_id[dataset_name]
        if cate_id == 0:
            semantic_mask = mask[...,1]!= 0
        else:
            semantic_mask = mask[...,1]== cate_id

        cv2.imwrite(os.path.join(mask_save_dir,"label",img_name+".jpg"),(semantic_mask*255))
        

        img,_ = dino_transformer(image,None)
        bs = 1 
        img = img[None,...]
        inputs = nested_tensor_from_tensor_list(img)
        inputs = inputs.to(device)

        

        with torch.no_grad():
            outputs = model(inputs, captions=[caption]*bs)
            img_size = torch.stack(((img_size),),dim=0).to(device)
            predn,pred_masks = postprocessor(outputs, img_size)
            pred_masks,bbox_mask,predn = select_predn_mask(predn[0],pred_masks[0],threshold,valid_mask = inputs.mask[0],origin_size=img_size[0],return_bbox_mask=True) 
            bbox_mask =bbox_mask.cpu().numpy()
            pred_masks = pred_masks.cpu().numpy()
            predn = predn.cpu().numpy()

            if len(predn)>100:
                continue

            cv2.imwrite(os.path.join(mask_save_dir,"pred_bbox",img_name+".jpg"),(bbox_mask*255))
            cv2.imwrite(os.path.join(mask_save_dir,"pred_mask",img_name+".jpg"),(pred_masks*255))

            _mIoU,_dice = mean_iou_and_dice(semantic_mask[None,None,...],pred_masks[None,None,...])
            mask_iou.append(_mIoU)
            mask_dice.append(_dice)
            _mIoU,_dice = mean_iou_and_dice(semantic_mask[None,None,...],bbox_mask[None,None,...])
            bbox_iou.append(_mIoU)
            bbox_dice.append(_dice)
        
        
        # For sam
        if len(predn):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                predictor.set_image(image)
                input_boxes = torch.tensor(predn,device=predictor.device)
                transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
            masks = masks.sum(dim=0).cpu().numpy()[0]
            masks[masks!=0] = 1
        else:
            masks = np.zeros_like(pred_masks)
        _mIoU,_dice = mean_iou_and_dice(semantic_mask[None,None,...],masks[None,None,...])
        cv2.imwrite(os.path.join(mask_save_dir,"pred_sam_box",img_name+".jpg"),(masks*255))
        sam_bbox_iou.append(_mIoU)
        sam_bbox_dice.append(_dice)
        predictor.reset_image()
    
    log_result = {
        "mIoU/pred_bbox": round(sum(bbox_iou)/len(bbox_iou),3),
        "dice/pred_bbox": round(sum(bbox_dice)/len(bbox_dice),3),
        "mIoU/pred_mask": round(sum(mask_iou)/len(mask_iou),3),
        "dice/pred_mask": round(sum(mask_dice)/len(mask_dice),3),
        "mIoU/pred_sam_box": round(sum(sam_bbox_iou)/len(sam_bbox_iou),3),
        "dice/pred_sam_box": round(sum(sam_bbox_dice)/len(sam_bbox_dice),3),
    }

    print(log_result)

    if args.wandb_log:
        wandb.log(log_result)
