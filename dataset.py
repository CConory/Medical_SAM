'''
Pytorch Dataloader
'''
import json
import numpy as np
import torch
import torch.utils.data as data
import random
import cv2
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide
import os
import torchvision
import matplotlib.pyplot as plt

from PIL import Image

from groundingdino.util.vl_utils import build_captions_and_token_span,create_positive_map_from_span
from groundingdino.util import get_tokenlizer, box_ops


def get_path(dataset_dir,file_name,suffix):
    image_path = os.path.join(dataset_dir,"images",file_name)
    feature_path = os.path.join(dataset_dir,"features",file_name.replace(suffix,".pt"))
    mask_path = os.path.join(dataset_dir,"masks",file_name.replace(suffix,".npy"))
    return image_path,feature_path,mask_path

def mask_2_boxes(mask):
    mask = mask.astype(np.int32)

    

    instance_bboxes = []
    instances_mask = mask[...,0]
    max_instance_nums = np.max(instances_mask)
    max_category_idx = np.max(mask[...,1])
    for category_idx in range(max_category_idx): #[0,1,2,3...n-1]
        semantic_mask = mask[...,1]== category_idx+1 # [1,2,3,....n]
        for instance_id in range(1,max_instance_nums+1):
            instance =  ((instances_mask==instance_id) * semantic_mask).astype(np.uint8)*255
            num_labels, labels = cv2.connectedComponents(instance)
            for i in range(1,num_labels):
                c1 = cv2.boundingRect((labels==i).astype(np.uint8)*255)
                if c1[2]<=0 or c1[3]<=0:
                    continue
                instance_bboxes.append([0,c1[0], c1[1], c1[0]+c1[2], c1[1]+c1[3],category_idx]) # bs_id, x1y1x2y2, class_index 
    if len(instance_bboxes):
        result = np.array(instance_bboxes,dtype=np.float32)
    else:
        result = np.zeros((0,6),dtype=np.float32)
    return result

class Dataset(data.Dataset):
    def __init__(self, json_path,split,device):
        with open(json_path, 'r') as f:
            ds_dict = json.load(f)
        self.dataset_dir = os.path.dirname(json_path)
        self.file_names = ds_dict[split]
        self.nF = len(self.file_names)
        self.device = device
    def __len__(self):
        return self.nF
    def __getitem__(self, index):
        file_name = self.file_names[index]
        
        suffix = os.path.splitext(file_name)[1]
        image_path,feature_path,mask_path  = get_path(self.dataset_dir,file_name,suffix)
        data = torch.load(feature_path)
        feature = data["fm"] #.to(self.device)
        original_size = data["origin_shape"][:2]
        mask = np.load(mask_path,allow_pickle=True)
        mask = mask.astype(np.int)
        semantic_mask = mask[...,1]!=0
        return feature,original_size,semantic_mask,image_path
    @staticmethod
    def collate_fn(batch):
        # batched_input = [
        #     {
        #         "feature":feature,
        #         'original_size':data["origin_shape"][:2]
        #     }
        # ]
        feature,original_size,mask,image_path = zip(*batch)
        batched_input = []
        masks = []
        for i in range(len(feature)):
            batched_input.append(
                {
                    "feature":feature[i],
                    'original_size':original_size[i]
                }
            )
            masks.append(mask[i][None][None][None])
        return batched_input, masks,image_path


class One_point_Dataset(Dataset):
    def __getitem__(self, index):
        file_name = self.file_names[index]
        suffix = os.path.splitext(file_name)[1]
        image_path,feature_path,mask_path  = get_path(self.dataset_dir,file_name,suffix)

        data = torch.load(feature_path)
        feature = data["fm"] #.to(self.device)
        original_size = data["origin_shape"][:2]
        mask = np.load(mask_path,allow_pickle=True)
        mask = mask.astype(np.int)

        semantic_mask = mask[...,1]!=0


        instances_mask = mask[...,0]
        max_instance_nums = np.max(instances_mask)
        points = np.array([])
        point_label = np.array([])

        if max_instance_nums > 1:
            instance_id = random.randint(1,max_instance_nums)
            instance_indices = np.where(instances_mask == instance_id)
            if len(instance_indices[0])>10:
                point_index = random.randint(0,len(instance_indices[0])-1)
                points = np.array([[instance_indices[1][point_index],instance_indices[0][point_index]]])
                point_label = np.array([1]*len(points)) #1 is foreground , 0 is background



        return feature,original_size,semantic_mask,points,point_label,image_path
    @staticmethod
    def collate_fn(batch):
        feature,original_size,mask,points,point_label,image_path = zip(*batch)
        batched_input = []
        masks = []
        for i in range(len(feature)):
            batched_input.append(
                {
                    "feature":feature[i],
                    "original_size":original_size[i],
                    "point_coords":points[i] if len(points[i]) else None,
                    "point_labels":point_label[i] if len(points[i]) else None

                }
            )
            masks.append(mask[i][None][None][None])
        return batched_input, masks,image_path


class Two_point_Dataset(Dataset):
    def __getitem__(self, index):
        file_name = self.file_names[index]

        suffix = os.path.splitext(file_name)[1]
        image_path,feature_path,mask_path  = get_path(self.dataset_dir,file_name,suffix)

        data = torch.load(feature_path)
        feature = data["fm"] #.to(self.device)
        original_size = data["origin_shape"][:2]
        mask = np.load(mask_path,allow_pickle=True)
        mask = mask.astype(np.int)
        semantic_mask = mask[...,1]!=0


        instances_mask = mask[...,0]
        max_instance_nums = np.max(instances_mask)
        points = []
        point_label = []
        if max_instance_nums > 1:
            instance_id = random.randint(1,max_instance_nums)
            instance_indices = np.where(instances_mask == instance_id)
            if len(instance_indices[0])>10:
                point_index = random.randint(0,len(instance_indices[0])-1)
                points.append([instance_indices[1][point_index],instance_indices[0][point_index]])
                point_label.append(1) #1 is foreground , 0 is background
                # points = np.array([[instance_indices[1][point_index],instance_indices[0][point_index]]])
                # point_label = np.array([1]) 
        
        bg_indices = np.where(instances_mask == 0)
        point_index = random.randint(0,len(bg_indices[0])-1)
        points.append([bg_indices[1][point_index],bg_indices[0][point_index]])
        point_label.append(0) #1 is foreground , 0 is background

        points = np.array(points)
        point_label = np.array(point_label)


        return feature,original_size,semantic_mask,points,point_label,image_path
    @staticmethod
    def collate_fn(batch):
        feature,original_size,mask,points,point_label,image_path = zip(*batch)
        batched_input = []
        masks = []
        for i in range(len(feature)):
            batched_input.append(
                {
                    "feature":feature[i],
                    "original_size":original_size[i],
                    "point_coords":points[i] if len(points[i]) else None,
                    "point_labels":point_label[i] if len(points[i]) else None

                }
            )
            masks.append(mask[i][None][None][None])
        return batched_input, masks,image_path


class Five_point_Dataset(Dataset):
    def __getitem__(self, index):
        file_name = self.file_names[index]

        suffix = os.path.splitext(file_name)[1]
        image_path,feature_path,mask_path  = get_path(self.dataset_dir,file_name,suffix)

        data = torch.load(feature_path)
        feature = data["fm"] #.to(self.device)
        original_size = data["origin_shape"][:2]
        mask = np.load(mask_path,allow_pickle=True)
        mask = mask.astype(np.int)

        semantic_mask = mask[...,1]!=0


        instances_mask = mask[...,0]
        max_instance_nums = np.max(instances_mask)
        points = []
        point_label = []
        if max_instance_nums>1:
            point_nums = min(5,max_instance_nums)
            instance_ids = random.sample(range(1,max_instance_nums+1), point_nums) # should generate include max_instance_nums, thus need max_instance_nums+1
            for instance_id in instance_ids:
                instance_indices = np.where(instances_mask == instance_id)
                if len(instance_indices[0])<10:
                    continue
                point_index = random.randint(0,len(instance_indices[0])-1)

                points.append([instance_indices[1][point_index],instance_indices[0][point_index]])
                point_label.append(1) #1 is foreground , 0 is background
                # points = np.array([[instance_indices[1][point_index],instance_indices[0][point_index]]])
                # point_label = np.array([1]) 
            
        for bg_ids in range(5-len(points)):
            bg_indices = np.where(instances_mask == 0)
            point_index = random.randint(0,len(bg_indices[0])-1)
            points.append([bg_indices[1][point_index],bg_indices[0][point_index]])
            point_label.append(0) #1 is foreground , 0 is background

        points = np.array(points)
        point_label = np.array(point_label)


        return feature,original_size,semantic_mask,points,point_label,image_path
    @staticmethod
    def collate_fn(batch):
        feature,original_size,mask,points,point_label,image_path = zip(*batch)
        batched_input = []
        masks = []
        for i in range(len(feature)):
            batched_input.append(
                {
                    "feature":feature[i],
                    "original_size":original_size[i],
                    "point_coords":points[i] if len(points[i]) else None,
                    "point_labels":point_label[i] if len(points[i]) else None

                }
            )
            masks.append(mask[i][None][None][None])
        return batched_input, masks,image_path


class Twenty_point_Dataset(Dataset):
    def __getitem__(self, index):
        file_name = self.file_names[index]

        suffix = os.path.splitext(file_name)[1]
        image_path,feature_path,mask_path  = get_path(self.dataset_dir,file_name,suffix)


        data = torch.load(feature_path)
        feature = data["fm"] #.to(self.device)
        original_size = data["origin_shape"][:2]
        mask = np.load(mask_path,allow_pickle=True)
        mask = mask.astype(np.int)
        semantic_mask = mask[...,1]!=0


        instances_mask = mask[...,0]
        max_instance_nums = np.max(instances_mask)
        points = []
        point_label = []
        if max_instance_nums>1:
            point_nums = min(10,max_instance_nums)
            instance_ids = random.sample(range(1,max_instance_nums+1), point_nums) # should generate include max_instance_nums, thus need max_instance_nums+1
            for instance_id in instance_ids:
                instance_indices = np.where(instances_mask == instance_id)
                if len(instance_indices[0])<10:
                    continue
                point_index = random.randint(0,len(instance_indices[0])-1)

                points.append([instance_indices[1][point_index],instance_indices[0][point_index]])
                point_label.append(1) #1 is foreground , 0 is background
                # points = np.array([[instance_indices[1][point_index],instance_indices[0][point_index]]])
                # point_label = np.array([1]) 
            
        for bg_ids in range(20-len(points)):
            bg_indices = np.where(instances_mask == 0)
            point_index = random.randint(0,len(bg_indices[0])-1)
            points.append([bg_indices[1][point_index],bg_indices[0][point_index]])
            point_label.append(0) #1 is foreground , 0 is background

        points = np.array(points)
        point_label = np.array(point_label)


        return feature,original_size,semantic_mask,points,point_label,image_path
    @staticmethod
    def collate_fn(batch):
        feature,original_size,mask,points,point_label,image_path = zip(*batch)
        batched_input = []
        masks = []
        for i in range(len(feature)):
            batched_input.append(
                {
                    "feature":feature[i],
                    "original_size":original_size[i],
                    "point_coords":points[i] if len(points[i]) else None,
                    "point_labels":point_label[i] if len(points[i]) else None

                }
            )
            masks.append(mask[i][None][None][None])
        return batched_input, masks,image_path


class All_point_Dataset(Dataset):
    def __getitem__(self, index):
        file_name = self.file_names[index]

        suffix = os.path.splitext(file_name)[1]
        image_path,feature_path,mask_path  = get_path(self.dataset_dir,file_name,suffix)

        data = torch.load(feature_path)
        feature = data["fm"] #.to(self.device)
        original_size = data["origin_shape"][:2]
        mask = np.load(mask_path,allow_pickle=True)
        mask = mask.astype(np.int)
        semantic_mask = mask[...,1]!=0


        instances_mask = mask[...,0]
        max_instance_nums = np.max(instances_mask)
        points = []
        point_label = []
        if max_instance_nums>1:
            for instance_id in range(1,max_instance_nums+1):
                instance_indices = np.where(instances_mask == instance_id)
                if len(instance_indices[0])<10:
                    continue
                point_index = random.randint(0,len(instance_indices[0])-1)

                points.append([instance_indices[1][point_index],instance_indices[0][point_index]])
                point_label.append(1) #1 is foreground , 0 is background
                # points = np.array([[instance_indices[1][point_index],instance_indices[0][point_index]]])
                # point_label = np.array([1]) 
            
        for bg_ids in range(len(points)):
            bg_indices = np.where(instances_mask == 0)
            point_index = random.randint(0,len(bg_indices[0])-1)
            points.append([bg_indices[1][point_index],bg_indices[0][point_index]])
            point_label.append(0) #1 is foreground , 0 is background

        points = np.array(points)
        point_label = np.array(point_label)


        return feature,original_size,semantic_mask,points,point_label,image_path
    @staticmethod
    def collate_fn(batch):
        feature,original_size,mask,points,point_label,image_path = zip(*batch)
        batched_input = []
        masks = []
        for i in range(len(feature)):
            batched_input.append(
                {
                    "feature":feature[i],
                    "original_size":original_size[i],
                    "point_coords":points[i] if len(points[i]) else None,
                    "point_labels":point_label[i] if len(points[i]) else None

                }
            )
            masks.append(mask[i][None][None][None])
        return batched_input, masks,image_path


class All_boxes_Dataset(Dataset):
    def __getitem__(self, index):
        file_name = self.file_names[index]

        suffix = os.path.splitext(file_name)[1]
        image_path,feature_path,mask_path  = get_path(self.dataset_dir,file_name,suffix)
        data = torch.load(feature_path)
        feature = data["fm"] #.to(self.device)
        original_size = data["origin_shape"][:2]
        mask = np.load(mask_path,allow_pickle=True)
        mask = mask.astype(np.int)
        semantic_mask = mask[...,1]!=0


        instances_mask = mask[...,0]
        max_instance_nums = np.max(instances_mask)
        instance_bboxes = []
        for instance_id in range(1,max_instance_nums+1):
            instance =  (instances_mask==instance_id).astype(np.uint8)*255
            # num_labels, labels = cv2.connectedComponents(instance) #计算联通域，instance——label的误差，导致同一个instance_id对应多个obj
            # if num_labels > 2:
            #     print(f"{image_path} has muiltiple objects correspond to one instance_id")
            #     continue
            c1 = cv2.boundingRect(instance)
            if c1[2]<=0 or c1[3]<=0:
                continue
            instance_bboxes.append([c1[0], c1[1], c1[0]+c1[2], c1[1]+c1[3]])

        return feature,original_size,semantic_mask,instance_bboxes,image_path
    @staticmethod
    def collate_fn(batch):
        feature,original_size,mask,instance_bboxes,image_path = zip(*batch)
        batched_input = []
        masks = []
        for i in range(len(feature)):
            batched_input.append(
                {
                    "feature":feature[i],
                    "original_size":original_size[i],
                    "boxes": np.array(instance_bboxes[i])
                }
            )
            masks.append(mask[i][None][None][None])
        return batched_input, masks,image_path

class Medical_Detecton(All_boxes_Dataset):
    def __init__(self, json_path,split,device,transforms=None,is_train=False,tokenizer=None):
        super().__init__(json_path, split,device)
        self._transforms = transforms
        self.is_train = is_train
        dataset_name = json_path.split("/")[-2]

        '''
            能让类别跟对应的instruction区分开，并且在后处理的时候能关联起来
        '''
        self.cat_list = ["vessel","glomerulus","unsure"]
        # self.cat_list = ["vessel"]
        instruction = [
                ["arterioles, capillaries or venules tissue"], # for category: 0
                ["glomerulus tissue"], #for category:1
                ["unsure tissue","unsure"], # for category:2
                ["normal_cell","cell","tissue"]
            ] # for others
        self.instruction = []
        self.cat_2_instruction = {}
        for instr_id in range(len(instruction)):
            self.instruction.extend(instruction[instr_id])
            if instr_id < len(self.cat_list):
                self.cat_2_instruction[self.cat_list[instr_id]] = instruction[instr_id]
        
        if self.is_train:
            assert tokenizer is not None
            self.tokenlizer = tokenizer
    def __getitem__(self, index):
        file_name = self.file_names[index]
        suffix = os.path.splitext(file_name)[1]
        image_path,feature_path,mask_path  = get_path(self.dataset_dir,file_name,suffix)
        image = Image.open(image_path)
        w,h = image.size
        img_size = torch.tensor([h, w])

        mask = np.load(mask_path,allow_pickle=True)
        mask = mask.astype(np.int32)
        target = mask_2_boxes(mask)
        target = torch.tensor(target)

        target[:, 1:-1:2].clamp_(min=0, max=w)
        target[:, 2:-1:2].clamp_(min=0, max=h)
        keep = (target[:, 4] > target[:, 2]) & (target[:, 3] > target[:, 1])
        target = target[keep]
        target[:,1:-1] = box_ops.box_xyxy_to_cxcywh(target[:,1:-1] / torch.tensor([w, h, w, h], dtype=torch.float32)) # Normalization 

        categor_names = [self.cat_list[int(cls_id.item())] for cls_id in target[:,-1]]

        # plt.figure(figsize=(10,10))
        # plt.imshow(image)
        # for box in instance_bboxes: 
        #     show_box(box[1:-1], plt.gca(),"green")
        # plt.savefig("./tmp.jpg")
        # import pdb;pdb.set_trace()
        if self._transforms is not None:
            img,_ = self._transforms(image,None)

        # generation caption instruction and target_class for the corresponding positive
        instruction = self.instruction.copy() 
        if self.is_train:
            all_category_idx = list(range(len(instruction))) #Used to calculate the scores of the shuffle category_id_list
            random.shuffle(all_category_idx)
            instruction = [ instruction[c_id] for c_id in all_category_idx]
        captions, cat2tokenspan = build_captions_and_token_span(instruction, True)

        if self.is_train:
            positive_tokens = []
            for cls_name in categor_names:
                positive_token = []
                for per_instruction_of_cls in self.cat_2_instruction[cls_name]:
                    positive_token.extend(cat2tokenspan[per_instruction_of_cls])
                positive_tokens.append(positive_token)
            one_hot_positive_map = create_positive_map_from_span(self.tokenlizer(captions), positive_tokens)  #Used to train
            one_hot_positive_map[one_hot_positive_map!=0] =1
            return img_size,img,target,image,captions,one_hot_positive_map,instruction
        else:
            return img_size,img,target,image,captions


    @staticmethod
    def collate_fn(batch):
        img_size,images,instance_bboxes,ori_img,captions = zip(*batch)
        for i,l in enumerate(instance_bboxes):
            l[:, 0] = i 
        return img_size,images,instance_bboxes,ori_img,list(captions)

    @staticmethod
    def collate_fn_for_train(batch):
        img_size,images,instance_bboxes,ori_img,captions,one_hot_positive_map,instruction = zip(*batch)
        one_hot_positive_map = torch.cat(one_hot_positive_map, dim=0)
        for i,l in enumerate(instance_bboxes):
            l[:, 0] = i 
        return img_size,images,instance_bboxes,ori_img,list(captions),one_hot_positive_map,instruction

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms,is_train=False,tokenizer=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms # from groundiing_dino
        self.inverse_continue_map= {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                  41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
        self.continue_map = {value:key for key,value in self.inverse_continue_map.items()}

        self.is_train = is_train

        category_dict = self.coco.dataset['categories']
        self.cat_list = [item['name'] for item in category_dict]

        if self.is_train:
            assert tokenizer is not None
            self.tokenlizer = tokenizer

        
    
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        w, h = img.size
        boxes = [obj["bbox"] for obj in target]
        categor_ids = [self.continue_map[obj["category_id"]] for obj in target] # COCO 类别有 80类， 但是类别id并不连续，因此需要把他们映射到【0～80】之间
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        categor_ids = torch.as_tensor(categor_ids, dtype=torch.float32)
        target = torch.zeros((len(boxes),6))
        target[:,1:-1] = boxes
        target[:,-1] = categor_ids

        target[:, 3:-1] += target[:, 1:3]
        target[:, 1:-1:2].clamp_(min=0, max=w)
        target[:, 2:-1:2].clamp_(min=0, max=h)
        keep = (target[:, 4] > target[:, 2]) & (target[:, 3] > target[:, 1])
        target = target[keep]
        target[:,1:-1] = box_ops.box_xyxy_to_cxcywh(target[:,1:-1] / torch.tensor([w, h, w, h], dtype=torch.float32)) # Normalization 

        categor_names = [self.cat_list[int(cls_id.item())] for cls_id in target[:,-1]]

        ori_img = img

        img_size = torch.tensor([h, w])

        if self._transforms is not None:
            img,_ = self._transforms(img,None)


        # generation caption instruction and target_class for the corresponding positive
        caption_list = self.cat_list.copy()
        if self.is_train:
            all_category_idx = list(range(len(caption_list))) #Used to calculate the scores of the shuffle category_id_list
            random.shuffle(all_category_idx)
            caption_list = [ caption_list[c_id] for c_id in all_category_idx]
            post_process_information = {"idx":all_category_idx}
        captions, cat2tokenspan = build_captions_and_token_span(caption_list, True)
        
        if self.is_train:
            positive_token = [cat2tokenspan[cls_name] for cls_name in categor_names]
            one_hot_positive_map = create_positive_map_from_span(self.tokenlizer(captions), positive_token)  #Used to train
            one_hot_positive_map[one_hot_positive_map!=0] =1

            '''
            用于可视化不同语序的文本输入 instruction, 转换成标签对应关系是否正确。
            '''
            visualization_during_train = False
            if visualization_during_train:
                tokenspanlist = [cat2tokenspan[cat] for cat in caption_list]
                positive_map = create_positive_map_from_span(self.tokenlizer(captions), tokenspanlist) #Used to post-porcessing
                post_process_information["map"]= positive_map
            else:
                post_process_information = {}
            return img_size,img,target,ori_img,captions,one_hot_positive_map,post_process_information
        else:
            return img_size,img,target,ori_img,captions

    
    @staticmethod
    def collate_fn(batch):
        img_size,images,instance_bboxes,ori_img,captions = zip(*batch)
        for i,l in enumerate(instance_bboxes):
            l[:, 0] = i 
        return img_size,images,instance_bboxes,ori_img,list(captions)

    @staticmethod
    def collate_fn_for_train(batch):
        img_size,images,instance_bboxes,ori_img,captions,one_hot_positive_map,post_process_information = zip(*batch)
        one_hot_positive_map = torch.cat(one_hot_positive_map, dim=0)
        for i,l in enumerate(instance_bboxes):
            l[:, 0] = i 
        return img_size,images,instance_bboxes,ori_img,list(captions),one_hot_positive_map,post_process_information