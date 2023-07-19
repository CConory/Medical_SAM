import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import groundingdino.datasets.transforms as T
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


from groundingdino.util.misc import nested_tensor_from_tensor_list
from groundingdino.util.slconfig import SLConfig

from multi_modal.models.groundingdino_v1 import build_groundingdino
from multi_modal.Loss.loss import ATSSLossComputation
from multi_modal.Grounding_dino_infer import load_model
from multi_modal.solver import make_optimizer,make_lr_scheduler
from dataset import CocoDetection,Medical_Detecton

import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import numpy as np 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco', help='the path of the dataset')
    args = parser.parse_args()
    if args.dataset == "coco":
        img_dir = "../data/coco_2017/val2017"
        anno_path = "../data/coco_2017/annotations/instances_val2017.json"

    device = "cuda"

    #Construct model
    config_file_path = "./config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "/userhome/cs2/kuangww/GroundingDINO/weights/groundingdino_swint_ogc.pth"

    cfg = SLConfig.fromfile(config_file_path)
    model = load_model(config_file_path, checkpoint_path)
    model = model.to(device)
    model = model.eval()

    
    
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
    scheduler = make_lr_scheduler(cfg, optimizer)
    import pdb;pdb.set_trace()
