import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
from dataset import Medical_Detecton
import requests

import torch
import cv2
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from multi_modal.eval_utils import processs_batch,ap_per_class
from tqdm import tqdm


import argparse
CATEGORY_CLASSES=["vessel"]




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HuBMAP', help='the path of the dataset')
    parser.add_argument('--dataset_type', type=str, default='test', help='test set or valid set')
    parser.add_argument('--model_type', type=str, default='owlvit', help='the type of the model')
    parser.add_argument('--mAP_threshold', type=float, default=0.3, help='the threshold that pedictio is corrected')
    # parser.add_argument('--prompt_type', type=str, default='text_v1', help='the type of the prompt')
    parser.add_argument('--wandb_log', action='store_true', help='save the result to wandb or not')
    args = parser.parse_args()

    if args.wandb_log:
        wandb.init(project="Medical_SAM",config={
            "dataset": args.dataset,
            "model_type": args.model_type,
            "mAP_threshold": args.mAP_threshold,
            "data_type": args.dataset_type,
            "fine-tune": False
        })
        wandb.run.name = wandb.run.id
        wandb.run.save()
    
    mAP_threshold = args.mAP_threshold
    dataset_root_dir = "../data"
    dataset_name = args.dataset
    dataset_dir = os.path.join(dataset_root_dir,dataset_name)
    device = "cuda"
    dataset = Medical_Detecton(os.path.join(dataset_dir,"data_split.json"),args.dataset_type,device)
    collate_fn = Medical_Detecton.collate_fn
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
        collate_fn = collate_fn,
        drop_last=False
    )

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16").to(device)
    model.eval()
    stats = []

    visulization_imgs = []
    vis_count = 0
    max_vis = 9
    for (imgs_size, images,targets,_) in tqdm(dataloader):
        inputs = processor(text=["cell"]*len(images), images=images, return_tensors="pt")
        inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        
        target_sizes = torch.stack(imgs_size,dim=0).to(device)
        results = processor.post_process_object_detection(outputs=outputs, threshold=0.0001,target_sizes=target_sizes)
        
        predn = []
        for i in range(len(images)):
            boxes, scores = results[i]["boxes"], results[i]["scores"]
            boxes[:,::2] = torch.clip(boxes[:,::2], min=0, max=target_sizes[0][1])
            boxes[:,1::2] = torch.clip(boxes[:,1::2], min=0, max=target_sizes[0][0])
            category = torch.zeros_like(scores).unsqueeze(-1)
            predn.append(torch.cat((boxes,scores.unsqueeze(-1),category),dim=-1))
        
        # Logger_visualization
        if vis_count<max_vis and args.wandb_log:
            vis_pred = predn[0][predn[0][:,-2]>0.5].cpu().numpy()
            visulization_imgs.append(visualization_bboxes(
                images[0],
                targets[0][:,1:],
                vis_pred
                ))
            vis_count += 1

        stats.extend(processs_batch(predn,targets,mAP_threshold))
        
    stats = [np.concatenate(x, 0) for x in zip(*stats)] 
    if len(stats) and stats[0].any():
        result = ap_per_class(*stats)

    if args.wandb_log:
        wandb_logger(result,visulization_imgs,args,CATEGORY_CLASSES)
