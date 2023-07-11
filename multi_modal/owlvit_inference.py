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
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from evaluate_from_pt import show_box
import wandb


CATEGORY_CLASSES=["vessel"]

def tag_images(img,objs,color="green",category_dict=None):
    '''
    objs = [N,5] for targets [ xyxy, category_id]
         = [N,6] for predn, [xyxy,score,category_id]
    '''
    W,H = img.size
    img1 = img.copy()
    draw = ImageDraw.Draw(img1)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 10)
    for i,obj in enumerate(objs):
        box = obj[:4]
        draw.rectangle(box,outline=color,width=4)
        x1,y1,x2,y2 = box
        if len(obj) == 6:
            label = label = category_dict[int(obj[-1])] if category_dict else str(int(int(obj[-1])))
            label +=  '({:.2f})'.format(obj[4])
        else:
            label = category_dict[int(obj[-1])] if category_dict else str(int(int(obj[-1])))
        if len(obj) > 4:
            w,h = font.getsize(label)
            if x1+w > W or y2+h > H:
                draw.rectangle((x1, y2-h, x1 + w, y2), fill=color)
                draw.text((x1,y2-h),label,fill='white',font=font)
            else:
                draw.rectangle((x1, y2, x1 + w, y2 + h), fill=color)
                draw.text((x1,y2),label,fill='white',font=font)
    return img1

def visualization_bboxes(image,target,predn,category_dict = None):
    image = tag_images(image,target,color = "green",category_dict = category_dict)
    image = tag_images(image,predn,color = "blue",category_dict = category_dict)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    # for box in target: 
    #     show_box(box, plt.gca(),"yellow")
    # for box in predn: 
    #     show_box(box, plt.gca(),"#0099FF")
    plt.axis('off')
    return wandb.Image(plt)

def wandb_logger(result,visulization_imgs,args,category_names):
    class_id2name = {id:name for id,name in enumerate(category_names)}
    f1 = result['f1']
    p = result['precision']
    r = result['recall']
    conf = f1.mean(0).argmax()

    print("### Evaluation Result ###")
    print(" The best Mean F1 score is under conf: ".ljust(5, ' '),round(conf / 1000, 2))
    print(" ")
    print("Class_Name".ljust(20, ' '), f"AP_{int(args.mAP_threshold*100)}".ljust(15, ' '),"Recall".ljust(15, ' '),"Precision".ljust(15, ' '),"F1".ljust(15, ' '))
    print(" ")

    data = []
    for i,class_id in enumerate(result["classes"]):
        class_name = class_id2name[class_id]
        print(f'{class_name}'.ljust(20, ' '), f'{result["ap"][i]*100:.2f}'.ljust(15, ' '), f'{r[i][conf]*100:.2f}'.ljust(15, ' '),f'{p[i][conf]*100:.2f}'.ljust(15, ' '),f'{f1[i][conf]*100:.2f}'.ljust(15, ' '))
        data.append([class_name,result["ap"][i]*100,r[i][conf]*100,p[i][conf]*100,f1[i][conf]*100,round(conf / 1000, 2)])
    print('Mean'.ljust(20, ' '), f'{result["ap"].mean(0)*100:.2f}'.ljust(15, ' '), f'{r.mean(0)[conf]*100:.2f}'.ljust(15, ' '),f'{p.mean(0)[conf]*100:.2f}'.ljust(15, ' '),f'{f1.mean(0)[conf]*100:.2f}'.ljust(15, ' '))
    data.append(["Mean",result["ap"].mean(0)*100,r.mean(0)[conf]*100,p.mean(0)[conf]*100,f1.mean(0)[conf]*100,round(conf / 1000, 2)])
    table = wandb.Table(data=data, columns = ["Class_Name", f"AP_{int(args.mAP_threshold*100)}","Recall","Precision","F1"," Best Confidence"])
    wandb.log({"val/result": table})

    wandb.log({
            f"test/mAP_{int(args.mAP_threshold*100)}": result["ap"].mean(0)*100,
            "test/recall": r.mean(0)[conf]*100,
            "test/precsion":p.mean(0)[conf]*100,
            "test/f1":f1.mean(0)[conf]*100,
            })
    
    # 类别太多了，画的太慢了
    # for index,class_id in enumerate(result["classes"][:5]):
    #     class_name = category_names[class_id]
    #     recall = result['recall'][index]
    #     precision = result['precision'][index]
    #     data = [[x, y] for (x, y) in zip(recall, precision)]
    #     table = wandb.Table(data=data, columns = ["recall", "precision"])
    #     plot = wandb.plot.line(table, "recall", "precision", stroke=None, title="Average Precision: "+class_name)
    #     wandb.log({"val/AP/"+class_name : plot})
    
    class_name = "Mean"
    recall = result['recall'].mean(0)
    precision = result['precision'].mean(0)
    data = [[x, y] for (x, y) in zip(recall, precision)]
    table = wandb.Table(data=data, columns = ["recall", "precision"])
    plot = wandb.plot.line(table, "recall", "precision", stroke=None, title="Average Precision: "+class_name)
    wandb.log({"val/AP/"+class_name : plot})

    wandb.log({"visualization": visulization_imgs})


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
