import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import groundingdino.datasets.transforms as T
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


from groundingdino.util.misc import nested_tensor_from_tensor_list
from groundingdino.util.slconfig import SLConfig
from groundingdino.util import get_tokenlizer,box_ops

from multi_modal.models.groundingdino_v1 import build_groundingdino
from multi_modal.Loss.loss import ATSSLossComputation
from multi_modal.Grounding_dino_infer import load_model,PostProcessGrounding
from multi_modal.solver import make_optimizer,make_lr_scheduler
from multi_modal.eval_utils import processs_batch,ap_per_class
from dataset import CocoDetection,Medical_Detecton
from logger import visualization_bboxes


import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import numpy as np 
import wandb
import shutil
from torch.cuda.amp import GradScaler
scaler = GradScaler()


def build_dataset_and_dataloader(cfg,args,is_train=False):

    if args.dataset == "coco":
        img_dir = "../data/coco_2017/val2017"
        anno_path = "../data/coco_2017/annotations/instances_val2017.json"

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
        ]
    )
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)

    if args.dataset == "coco":
        dataset = CocoDetection(img_dir,anno_path,transform,is_train=is_train,tokenizer=tokenlizer)
        # collate_fn = CocoDetection.collate_fn
        collate_fn = CocoDetection.collate_fn_for_train if is_train else CocoDetection.collate_fn
    else:
        dataset_root_dir = "../data"
        dataset_name = args.dataset
        dataset_dir = os.path.join(dataset_root_dir,dataset_name)
        dataset_type = "train" if is_train else "valid"
        dataset = Medical_Detecton(os.path.join(dataset_dir,"data_split.json"),dataset_type,device,transform,is_train=is_train,tokenizer=tokenlizer)
        collate_fn = Medical_Detecton.collate_fn_for_train if is_train else Medical_Detecton.collate_fn
    
    shuffle = True if is_train else False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,
        shuffle=shuffle,
        pin_memory=False,
        collate_fn = collate_fn,
        drop_last=False
    )
    return dataset,dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco', help='the path of the dataset')
    parser.add_argument('--mAP_threshold', type=float, default=0.5, help='the threshold that pedictio is corrected')
    parser.add_argument('--wandb_log', action='store_true', help='save the result to wandb or not')
    parser.add_argument("--output_dir", default="OUTPUT", type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    args = parser.parse_args()

    if args.wandb_log:
        wandb.init(project="Medical_SAM",config={
            "dataset": args.dataset,
            "mAP_threshold": args.mAP_threshold,
            "fine-tune": True
        })
        wandb.run.name = wandb.run.id
        wandb.run.save()
        args.output_dir = os.path.join(args.output_dir,wandb.run.id)
    else:
        args.output_dir = os.path.join(args.output_dir,"exp")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    config_file_path = "./config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "/userhome/cs2/kuangww/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    cfg = SLConfig.fromfile(config_file_path)
    shutil.copy(config_file_path,os.path.join(args.output_dir,os.path.basename(config_file_path)))


    device = "cuda"

    # Construct the dataset and dataloader
    trian_dataset,train_loader = build_dataset_and_dataloader(cfg,args,is_train=True)
    category_dict = {id:cat_item for id,cat_item in enumerate(trian_dataset.cat_list)}

    valid_dataset,valid_loader = build_dataset_and_dataloader(cfg,args,is_train=False)

    if cfg.max_iter is None:
        cfg.max_iter = (len(train_loader)//cfg.gradient_calculate_step)*cfg.max_epoch
    if cfg.warmup_iters is None:
        cfg.warmup_iters = min(round(3 * len(train_loader)), 2000)//cfg.gradient_calculate_step


    #Construct model
    model = load_model(config_file_path, checkpoint_path)
    model = model.to(device)
    model = model.train()

    
    
    if cfg.image_backbone_freeze:
        for p in model.backbone.parameters():
            p.requires_grad = False
    if cfg.language_backbone_freeze:
        for p in model.bert.parameters():
            p.requires_grad = False
    if cfg.transformer_freeze:
        for p in model.transformer.parameters():
            p.requires_grad = False
    
    if not cfg.box_cls_embed_freeze:  # 因为transformer 包含box，cls_embed 的共享权重
        for p in model.bbox_embed.parameters():
            p.requires_grad = True
        for p in model.class_embed.parameters():
            p.requires_grad = True
    
    optimizer = make_optimizer(cfg, model)
    optimizer.zero_grad()
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Construct Loss
    criterion = ATSSLossComputation(cfg).to(device)

    # Construct the Post-processing for the predn of training or evaluate processing
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessGrounding(
        category_list=trian_dataset.cat_list,
        instruction = trian_dataset.instruction,
        cat_2_instruction =trian_dataset.cat_2_instruction,
        tokenlizer=tokenlizer)

    # gradient_accumlate
    nb = len(train_loader)
    last_opt_step = -1
    
    # Start Training
    best_val_map = 0
    patience = 0

    for epoch in range(cfg.max_epoch):
        model.train()

        total_loss = 0.

        train_log_loss = {
            "train/loss/loss_ce":0,
            "train/loss/l1_bbox":0,
            "train/loss/loss_giou":0
        }
        print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
        
        progress_bar = tqdm(enumerate(train_loader), total=nb)
        train_stats = []

        # Training process
        for i, (imgs_size, images,targets, ori_img, captions, one_hot_positive_map,instruction) in progress_bar:
            
            ni = i + nb * epoch  #num_iteration
            targets = [tmp.to(device) for tmp in targets]
            one_hot_positive_map = one_hot_positive_map.to(device)
            inputs = nested_tensor_from_tensor_list(images)
            inputs = inputs.to(device)
            outputs = model(inputs, captions=captions)
            loss_dict,sum_loss = criterion(outputs,targets,one_hot_positive_map)
            scaler.scale(sum_loss).backward()

            for key,value in loss_dict.items():
                train_log_loss[f"train/loss/{key}"]+=value
            total_loss += sum_loss.data
            

            lr = optimizer.param_groups[0]['lr']
            

            if ni - last_opt_step >= cfg.gradient_calculate_step:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                last_opt_step = ni
            
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch + 1, cfg.max_epoch), total_loss / (i + 1), mem)
            progress_bar.set_description(s)

            


            imgs_size = torch.stack(imgs_size,dim=0).to(device)
            # targets cxcywh -> original image xxyy
            img_h, img_w = imgs_size.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            for bs_id in range(len(targets)):
                targets[bs_id][:,1:-1] = box_ops.box_cxcywh_to_xyxy(targets[bs_id][:,1:-1]) * scale_fct[bs_id]
                

            # Preprocess for the predn and get the evaluation result for the training dataset
            # with torch.no_grad():
            #     predn = postprocessor(outputs, imgs_size,instruction)
            #     train_stats.extend(processs_batch(predn,targets,args.mAP_threshold))

            # 可视化训练过程中标签对不对, 保存前10个batch的第一张图片:
            if args.wandb_log and ni<10 :
                # vis_pred = predn[0][predn[0][:,-2]>0.3].cpu().numpy() # 输出预测结果的可视化
                vis_img = visualization_bboxes(ori_img[0], targets[0][:,1:].cpu().numpy(), predn =[],category_dict=category_dict)
            else:
                vis_img = None
            
            if args.wandb_log:
                log_result = {
                    "train/loss/total":total_loss / (i + 1),
                    "train/lr": lr}
                for key,value in train_log_loss.items():
                    log_result[key] = train_log_loss[key]/(i + 1)
                if vis_img:
                    log_result.update({"visualization/labels": vis_img})
                wandb.log(log_result)
            
            break

        # train_mean_AP = 0
        # train_stats = [np.concatenate(x, 0) for x in zip(*train_stats)] 
        # if len(train_stats) and train_stats[0].any():
        #     result = ap_per_class(*train_stats)
        #     train_mean_AP = result["ap"].mean(0)*100
        
        # Evaluate Processing
        val_stats = []
        model.eval()
        progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for i, (imgs_size, images,targets, ori_img, captions) in progress_bar:
            with torch.no_grad():
                targets = [tmp.to(device) for tmp in targets]
                inputs = nested_tensor_from_tensor_list(images)
                inputs = inputs.to(device)
                outputs = model(inputs, captions=captions)
                imgs_size = torch.stack(imgs_size,dim=0).to(device)
                predn = postprocessor(outputs, imgs_size)

                # targets cxcywh -> original image xxyy         
                img_h, img_w = imgs_size.unbind(1)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
                for bs_id in range(len(targets)):
                    targets[bs_id][:,1:-1] = box_ops.box_cxcywh_to_xyxy(targets[bs_id][:,1:-1]) * scale_fct[bs_id]
                val_stats.extend(processs_batch(predn,targets,args.mAP_threshold))
            
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' + '%10s') % ('%g/%g' % (epoch + 1, cfg.max_epoch), mem)
            progress_bar.set_description(s)
        
        # Visulization the valid prediction result
        visulization_imgs = []
        for bs_id in range(len(images)):
            vis_pred = predn[bs_id][predn[bs_id][:,-2]>0.3].cpu().numpy()
            visulization_imgs.append(visualization_bboxes(ori_img[bs_id], targets[bs_id][:,1:].cpu().numpy(), predn =vis_pred,category_dict=category_dict))


                
        val_stats = [np.concatenate(x, 0) for x in zip(*val_stats)] 
        valid_mean_AP = 0
        if len(val_stats) and val_stats[0].any():
            result = ap_per_class(*val_stats)
            valid_mean_AP = result["ap"].mean(0)*100
        
        if valid_mean_AP > best_val_map:
            patience = 0
            best_val_map = valid_mean_AP
            save = {'state_dict': model.state_dict()}
            torch.save(save, os.path.join(args.output_dir, 'best.pth'))
        else:
            patience+=1
            save = {'state_dict': model.state_dict()}
            torch.save(save, os.path.join(args.output_dir, 'last.pth'))
        
        
        if args.wandb_log:
            log_result = {
                f"val/mAP_{int(args.mAP_threshold*100)}":valid_mean_AP,
                "visualization/valid_result":visulization_imgs
                }
            wandb.log(log_result)


        

