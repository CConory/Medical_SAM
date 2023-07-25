import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys

from segment_anything.utils.transforms import ResizeLongestSide


from groundingdino.util.misc import nested_tensor_from_tensor_list
from groundingdino.util.slconfig import SLConfig
from groundingdino.util import get_tokenlizer,box_ops

from multi_modal.models.groundingdino_v1 import build_groundingdino
from multi_modal.Loss.loss import ATSSLossComputation
from multi_modal.models.build_sam import sam_model_registry
from multi_modal.solver import make_optimizer,make_lr_scheduler
from multi_modal.eval_utils import processs_batch,ap_per_class
from dataset import Medical_SAM
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

    transform = resize_transform = ResizeLongestSide(cfg.encoder_img_size)
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)


    dataset_root_dir = "./data"
    dataset_name = args.dataset
    dataset_dir = os.path.join(dataset_root_dir,dataset_name)
    dataset_type = "train" if is_train else "valid"
    dataset = Medical_SAM(os.path.join(dataset_dir,"data_split.json"),dataset_type,device,transform,is_train=is_train,tokenizer=tokenlizer)
    collate_fn = Medical_SAM.collate_fn_for_train if is_train else Medical_SAM.collate_fn
    
    shuffle = True if is_train else False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
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
    
    config_file_path = "./multi_modal/config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "/userhome/cs2/kuangww/medical_sam/weights/sam_vit_b_01ec64.pth"
    cfg = SLConfig.fromfile(config_file_path)
    shutil.copy(config_file_path,os.path.join(args.output_dir,os.path.basename(config_file_path)))


    device = "cuda"

    #Construct model
    model = sam_model_registry["vit_b"](args=cfg,checkpoint = checkpoint_path) 
    model = model.to(device)
    model = model.train()
    cfg.encoder_img_size = model.image_encoder.img_size

    # Construct the dataset and dataloader
    trian_dataset,train_loader = build_dataset_and_dataloader(cfg,args,is_train=True)
    category_dict = {id:cat_item for id,cat_item in enumerate(trian_dataset.cat_list)}

    # valid_dataset,valid_loader = build_dataset_and_dataloader(cfg,args,is_train=False)

    if cfg.max_iter is None:
        cfg.max_iter = (len(train_loader)//cfg.gradient_calculate_step)*cfg.max_epoch
    if cfg.warmup_iters is None:
        cfg.warmup_iters = min(round(3 * len(train_loader)), 2000)//cfg.gradient_calculate_step

    
    
    if cfg.image_backbone_freeze:
        for p in model.image_encoder.parameters():
            p.requires_grad = False
    if cfg.language_backbone_freeze:
        for p in model.new_decoder.bert.parameters():
            p.requires_grad = False
    # if cfg.transformer_freeze:
    #     for p in model.new_decoder.transformer.parameters():
    #         p.requires_grad = False
    
    # if not cfg.box_cls_embed_freeze:  # 因为transformer 包含box，cls_embed 的共享权重
    #     for p in model.new_decoder.bbox_embed.parameters():
    #         p.requires_grad = True
    #     for p in model.new_decoder.class_embed.parameters():
    #         p.requires_grad = True
    
    optimizer = make_optimizer(cfg, model)
    optimizer.zero_grad()
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Construct Loss
    criterion = ATSSLossComputation(cfg).to(device)

    # Construct the Post-processing for the predn of training or evaluate processing
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    # postprocessor = PostProcessGrounding(
    #     category_list=trian_dataset.cat_list,
    #     instruction = trian_dataset.instruction,
    #     cat_2_instruction =trian_dataset.cat_2_instruction,
    #     tokenlizer=tokenlizer)

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
            "train/loss/loss_giou":0,
            "train/loss/loss_mask":0,
            "train/loss/loss_dice":0
        }
        print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
        
        progress_bar = tqdm(enumerate(train_loader), total=nb)
        train_stats = []

        # Training process
        for i, (imgs_size,images,targets,ori_img,captions,one_hot_positive_map,instruction,masks) in progress_bar:
            
            ni = i + nb * epoch  #num_iteration
            images = images.to(device)
            outputs = model(images, captions=captions)
            targets = [tmp.to(device) for tmp in targets]
            one_hot_positive_map = one_hot_positive_map.to(device)
            masks = masks.to(device)
            loss_dict,sum_loss = criterion(outputs,targets,one_hot_positive_map,masks)
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
        category_AP = {}
        if len(val_stats) and val_stats[0].any():
            result = ap_per_class(*val_stats)
            valid_mean_AP = result["ap"].mean(0)*100
            for i,class_id in enumerate(result["classes"]):
                class_name = category_dict[class_id]
                category_AP.update({f"val/{class_name}_AP_{int(args.mAP_threshold*100)}":result["ap"][i]*100})


        
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
            log_result.update(category_AP)
            wandb.log(log_result)


        


