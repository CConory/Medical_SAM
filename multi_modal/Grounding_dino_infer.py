import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import groundingdino.datasets.transforms as T
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from groundingdino.util.misc import nested_tensor_from_tensor_list
from groundingdino.util.slconfig import SLConfig
# from groundingdino.models import build_model
from multi_modal.models.groundingdino_v1 import build_groundingdino
from groundingdino.util.misc import clean_state_dict
from groundingdino.util import get_tokenlizer, box_ops
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span

from dataset import CocoDetection,Medical_Detecton
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F

from multi_modal.eval_utils import processs_batch,ap_per_class
from multi_modal.Loss.loss import ATSSLossComputation
from logger import visualization_bboxes,wandb_logger

import argparse
import numpy as np 
import wandb

def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    # model = build_model(args)
    model = build_groundingdino(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

class PostProcessGrounding(nn.Module):
    def __init__(self, num_select=300,category_list=None,instruction=None,cat_2_instruction=None, tokenlizer=None,encoder_input_size=1024) -> None:
        super().__init__()
        self.num_select = num_select
        self.encoder_input_size = encoder_input_size
        # captions 拼接成一个句子， cat2tokenspan : captions 在句子中的起始位置
        
        self.category_list = category_list
        self.cat_2_instruction = cat_2_instruction
        self.tokenlizer = tokenlizer

        if cat_2_instruction is None:
            captions, cat2tokenspan = build_captions_and_token_span(category_list, True)
            tokenspanlist = [cat2tokenspan[cat] for cat in category_list]

        else:
            captions, cat2tokenspan = build_captions_and_token_span(instruction, True)
            tokenspanlist = []
            for cls_name in category_list:
                positive_token = []
                for per_instruction_of_cls in cat_2_instruction[cls_name]:
                    positive_token.extend(cat2tokenspan[per_instruction_of_cls])
                tokenspanlist.append(positive_token)
        # caption 对应的 token 的 position mask. [0,7] 代表了 “love . e” 对应的是同一个 caption，
        # 则该caption 对应 对应了 两个 token ： love 以及 eight.
        # 设置了 最多有 256 类 token 来表示 caption。
        # Dog , Big Dog, Small Dog 一共有 两类 token， 一个是 Dog， 一个是 Big, 以及 Small
        # 则 Dog 对应的 postive_map 是 [1,0,0]
        # Big Dog 对应 [0.5,0.5,0]
        # Small Dog 则是 [0.5,0,0.5]
        self.positive_map = create_positive_map_from_span(tokenlizer(captions), tokenspanlist) # COCO 只统计了其中80个类别
    @torch.no_grad()
    def forward(self, outputs, target_sizes,instruction=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            post_information : 用于 instruction 打乱后， 各部分token对应不同类别 id 的 映射
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        out_masks = outputs.get("pred_masks",None)
        prob_to_token = out_logits.sigmoid()
        if instruction:
            # FIXME 在训练过程中，要用for循环 把不同 batch_id 的 image 的instrcution 重新计算得到pos_map 再concat起来
            captions, cat2tokenspan = build_captions_and_token_span(instruction, True)
            tokenspanlist = []
            for cls_name in self.category_list:
                positive_token = []
                for per_instruction_of_cls in self.cat_2_instruction[cls_name]:
                    positive_token.extend(cat2tokenspan[per_instruction_of_cls])
                tokenspanlist.append(positive_token)
            pos_maps = create_positive_map_from_span(tokenlizer(captions), tokenspanlist).to(prob_to_token.device)
        else:
            pos_maps = self.positive_map.to(prob_to_token.device)


        # (bs, 900, 256) @ (80, 256).T -> (bs, 900, 80) # 900 means the max bboxes
        # [bs,i,j] 第 i 个 bbox 对 j category 的 scores， 是multi-label， 即每一类别的positive的probability
        prob_to_label = prob_to_token @ pos_maps.T

        topk_values, topk_indexes = torch.topk(prob_to_label.view(out_logits.shape[0], -1), self.num_select, dim=1) # 同一个bbox 可能对应不同的类别
        scores = topk_values
        topk_boxes = topk_indexes // prob_to_label.shape[2]
        labels = topk_indexes % prob_to_label.shape[2]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox) #坐标转换
        boxes = torch.gather( boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        if out_masks is not None:
            if out_masks.shape[1]==1:
                masks = out_masks
            else:
                masks = torch.gather( out_masks, 1, topk_boxes.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, *(out_masks.shape[-2:])))
            # masks = F.interpolate(masks,(self.encoder_input_size, self.encoder_input_size),mode="bilinear",align_corners=False,)
        else:
            masks = None
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        boxes = boxes * scale_fct[:, None, :]

        result = torch.cat((boxes,scores.unsqueeze(-1),labels.unsqueeze(-1)),dim=-1)
        return result,masks
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco', help='the path of the dataset')
    parser.add_argument('--dataset_type', type=str, default='valid', help='test set or valid set')
    parser.add_argument('--model_type', type=str, default='grounding_dino', help='the type of the model')
    parser.add_argument('--mAP_threshold', type=float, default=0.5, help='the threshold that pedictio is corrected')
    parser.add_argument('--wandb_log', action='store_true', help='save the result to wandb or not')
    args = parser.parse_args()
    
    if args.dataset == "coco":
        img_dir = "../data/coco_2017/val2017"
        anno_path = "../data/coco_2017/annotations/instances_val2017.json"

        
    
    device = "cuda"
    mAP_threshold = args.mAP_threshold

    #Construct model
    config_file_path = "./config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "/userhome/cs2/kuangww/GroundingDINO/weights/groundingdino_swint_ogc.pth"

    cfg = SLConfig.fromfile(config_file_path)
    model = load_model(config_file_path, checkpoint_path)
    model = model.to(device)
    model = model.eval()

     

    # T.Normalize just for image,
    # if want to normalize boxes, please change the code in groundingdino/datasets/transforms.py.Normalize.call()
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
        ]
    )

    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    if args.dataset == "coco":
        dataset = CocoDetection(img_dir,anno_path,transform,is_train=True,tokenizer=tokenlizer)
        # collate_fn = CocoDetection.collate_fn
        collate_fn = CocoDetection.collate_fn_for_train
    else:
        dataset_root_dir = "../data"
        dataset_name = args.dataset
        dataset_dir = os.path.join(dataset_root_dir,dataset_name)
        dataset = Medical_Detecton(os.path.join(dataset_dir,"data_split.json"),args.dataset_type,device,transform)
        collate_fn = Medical_Detecton.collate_fn

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        collate_fn = collate_fn,
        drop_last=False
    )

    if args.dataset == "coco":
        # build coco caption
        category_dict = dataset.coco.dataset['categories']
        cat_list = [item['name'] for item in category_dict]
        caption_list = cat_list
    else:
        cat_list = ['cell']
        caption_list = ["hollow circle"]
    
    caption = " . ".join(caption_list) + ' .'
    category_dict = {id:cat_item for id,cat_item in enumerate(cat_list)}


    if args.wandb_log:
        wandb.init(project="Medical_SAM",config={
            "dataset": args.dataset,
            "model_type": args.model_type,
            "mAP_threshold": args.mAP_threshold,
            "data_type": args.dataset_type,
            "text_prompt": caption,
            "fine-tune": False
        })
        wandb.run.name = wandb.run.id
        wandb.run.save()


    # build post processor
    postprocessor = PostProcessGrounding(
        category_list=cat_list, tokenlizer=tokenlizer)
    

    stats = []
    visulization_imgs = []
    vis_count = 0
    max_vis = 9


    for (imgs_size, images,targets, ori_img,_) in tqdm(dataloader):
        # 会按照batch最大的width 跟height 对每张图像 进行 padding
        bs = len(imgs_size)
        inputs = nested_tensor_from_tensor_list(images) #Got inputs.tensors & inputs.mask
        inputs = inputs.to(device)
        with torch.no_grad():
            # batch_size >1 , 应该 把 inputs.mask 也一起传进去
            outputs = model(inputs, captions=[caption]*bs)
            imgs_size = torch.stack(imgs_size,dim=0).to(device)
            predn = postprocessor(outputs, imgs_size) #results 映射回原图大小， 注意 targets['bbox]是在Transform后的尺寸。
        
        # Logger_visualization
        if vis_count<max_vis and args.wandb_log:
            vis_pred = predn[0][predn[0][:,-2]>0.1].cpu().numpy()
            visulization_imgs.append(visualization_bboxes(ori_img[0], targets[0][:,1:].numpy(), vis_pred,category_dict))
            vis_count += 1

        stats.extend(processs_batch(predn,targets,mAP_threshold))

    stats = [np.concatenate(x, 0) for x in zip(*stats)] 
    if len(stats) and stats[0].any():
        result = ap_per_class(*stats)
    
    if args.wandb_log:
        wandb_logger(result,visulization_imgs,args,cat_list)
        
        
        