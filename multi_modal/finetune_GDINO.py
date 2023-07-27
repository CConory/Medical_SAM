import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import groundingdino.datasets.transforms as T
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


from groundingdino.util.misc import nested_tensor_from_tensor_list,NestedTensor
from groundingdino.util.slconfig import SLConfig
from groundingdino.util import get_tokenlizer,box_ops

from multi_modal.models.groundingdino_v1 import build_groundingdino
from multi_modal.Loss.loss import ATSSLossComputation
from multi_modal.Grounding_dino_infer import load_model,PostProcessGrounding
from multi_modal.solver import make_optimizer,make_lr_scheduler
from multi_modal.eval_utils import processs_batch,ap_per_class
from dataset import CocoDetection,Medical_Detecton
from logger import visualization_bboxes,visualization_masks
from torch.nn import functional as F
from evaluate_from_pt import mean_iou_and_dice
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import numpy as np 
import wandb
import shutil
from torch.cuda.amp import GradScaler
scaler = GradScaler()

def post_process_mask_for_GD(masks,origin_sizes,valid_mask = None,pred_flag=False):

    return_masks = []
    if pred_flag:
        assert valid_mask is not None
        masks = F.interpolate(masks, valid_mask.shape[-2:], mode="bilinear", align_corners=False)
        masks = NestedTensor(masks,valid_mask).to_img_list()
    for img_id, mask in enumerate(masks):
        origin_size = tuple(origin_sizes[img_id].int().tolist())
        _masks = mask[None,...]
        _masks = F.interpolate(_masks, origin_size, mode="bilinear", align_corners=False)[0]
        if not pred_flag:
            _masks = _masks.sum(dim=0)
            _masks[_masks!=0] = 1
        else:
            _masks = _masks>0 # sigmoid 前大与0， 等于sigmoid 后大于0.5
    
        return_masks.append(_masks)
    return return_masks


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
        num_workers=4,
        shuffle=shuffle,
        pin_memory=False,
        collate_fn = collate_fn,
        drop_last=False
    )
    return dataset,dataloader

def select_predn_mask(predn_per_img,pred_mask_per_img,conf_threshold=0.3):

    pred_mask_per_img = pred_mask_per_img[predn_per_img[:,-2]>conf_threshold] # 与可视化相同的阈值， 只有 bbox 的conf>0.3 才会发上 masks
    pred_mask_per_img = pred_mask_per_img.sum(dim=0) #instance_mask -> foreground semantic mask
    pred_mask_per_img [pred_mask_per_img!=0] =1 #[h,w]

    if True: #Version3 mask decoder
        h,w = pred_mask_per_img.shape[-2:]
        _predn_per_img = predn_per_img[predn_per_img[:,-2]>conf_threshold][:,:4].int()
        _predn_per_img[::2].clamp(min=0, max=w)
        _predn_per_img[1::2].clamp(min=0, max=h)
        _predn_per_img = _predn_per_img
        bbox_mask = torch.zeros_like(pred_mask_per_img,dtype=torch.bool)
        for (x1,y1,x2,y2) in _predn_per_img:
            bbox_mask[y1:y2,x1:x2] = True
        pred_mask_per_img *= bbox_mask

    return pred_mask_per_img


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
            "train/loss/loss_giou":0,
            "train/loss/loss_mask":0,
            "train/loss/loss_dice":0
        }
        print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
        
        progress_bar = tqdm(enumerate(train_loader), total=nb)
        train_stats = []

        # Training process
        for i, (imgs_size, images,targets, ori_img, captions, one_hot_positive_map,instruction,masks) in progress_bar:
            ni = i + nb * epoch  #num_iteration
            targets = [tmp.to(device) for tmp in targets]
            one_hot_positive_map = one_hot_positive_map.to(device)
            inputs = nested_tensor_from_tensor_list(images)
            inputs = inputs.to(device)
            outputs = model(inputs, captions=captions)
            
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
                target_masks = post_process_mask_for_GD(masks,imgs_size)

                predn,masks = postprocessor(outputs, imgs_size)
                pred_masks = post_process_mask_for_GD(masks,imgs_size,valid_mask = inputs.mask,pred_flag=True)
                if pred_masks[0].shape[0]==1:
                    pred_masks =pred_masks[0].squeeze(0)
                else:
                    pred_masks = select_predn_mask(predn[0],pred_masks[0],0.1)
                vis_img = visualization_masks(vis_img,target_masks[0].cpu().numpy(),pred_mask=pred_masks.cpu().numpy(),img_style="plt")

                vis_img =  wandb.Image(vis_img)
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
        mIoU = []
        dice = []
        model.eval()
        progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for i, (imgs_size, images,targets, ori_img, captions,masks) in progress_bar:
            with torch.no_grad():
                targets = [tmp.to(device) for tmp in targets]
                inputs = nested_tensor_from_tensor_list(images)
                inputs = inputs.to(device)
                outputs = model(inputs, captions=captions)
                imgs_size = torch.stack(imgs_size,dim=0).to(device)

                
                predn,pred_masks = postprocessor(outputs, imgs_size)
                if pred_masks is not None:
                    pred_masks = post_process_mask_for_GD(pred_masks,imgs_size,valid_mask = inputs.mask,pred_flag=True)
                    target_masks = post_process_mask_for_GD(masks,imgs_size)
                    for per_img_id in range(len(pred_masks)):
                        if pred_masks[per_img_id].shape[0] == 1:
                            pred_mask_per_img = pred_masks[per_img_id].squeeze(0)
                        else:
                            pred_mask_per_img = select_predn_mask(predn[per_img_id],pred_masks[per_img_id],0.3)
                        target_mask_per_img = target_masks[per_img_id] 
                        _mIoU,_dice = mean_iou_and_dice(target_mask_per_img[None,None,...].cpu().numpy(),pred_mask_per_img[None,None,...].cpu().numpy())
                        mIoU.append(_mIoU)
                        dice.append(_dice)
                else:
                    mIoU.append(0)
                    dice.append(0)

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
            vis_img = visualization_bboxes(ori_img[bs_id], targets[bs_id][:,1:].cpu().numpy(), predn =vis_pred,category_dict=category_dict)
            if pred_masks is not None:
                if pred_masks[bs_id].shape[0] == 1:
                    pred_mask_per_img = pred_masks[bs_id].squeeze(0)
                else:
                    pred_mask_per_img = select_predn_mask(predn[bs_id],pred_masks[bs_id],0.3)
                target_mask_per_img = target_masks[bs_id] 
                vis_img = visualization_masks(vis_img,target_mask_per_img.cpu().numpy(),pred_mask=pred_mask_per_img.cpu().numpy(),img_style="plt")
            vis_img =  wandb.Image(vis_img)
            visulization_imgs.append(vis_img)

        mIoU = round(sum(mIoU)/len(mIoU),3)
        dice = round(sum(dice)/len(dice),3)
        print(f"epochs: {epoch}, valid_mIoU: ",mIoU)
        print(f"epochs: {epoch}, valid_dice: ",dice)

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
                "val/mIoU": mIoU,
                "val/dice": dice,
                f"val/mAP_{int(args.mAP_threshold*100)}":valid_mean_AP,
                "visualization/valid_result":visulization_imgs
                }
            log_result.update(category_AP)
            wandb.log(log_result)


        


