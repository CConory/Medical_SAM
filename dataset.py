'''
Pytorch Dataloader
'''
import json
import numpy as np
import torch
import torch.utils.data as data
import random
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide


class PanNuke_Dataset(data.Dataset):
    def __init__(self, json_path,split,device):
        with open(json_path, 'r') as f:
            ds_dict = json.load(f)
        self.file_names = ds_dict[split]
        self.nF = len(self.file_names)
        self.device = device
    def __len__(self):
        return self.nF
    def __getitem__(self, index):
        file_name = self.file_names[index]
        image_path = "./datasets/PanNuke/images/" + file_name + ".png"
        data = torch.load("./datasets/PanNuke/features/" + file_name + '.pt')
        feature = data["fm"] #.to(self.device)
        original_size = data["origin_shape"][:2]
        mask = np.load(f'./datasets/PanNuke/masks/{file_name}.npy',allow_pickle=True)
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
        return batched_input, np.concatenate(masks,axis=0),image_path


class PanNuke_one_point_prompt(PanNuke_Dataset):
    def __getitem__(self, index):
        file_name = self.file_names[index]
        image_path = "./datasets/PanNuke/images/" + file_name + ".png"
        data = torch.load("./datasets/PanNuke/features/" + file_name + '.pt')
        feature = data["fm"] #.to(self.device)
        original_size = data["origin_shape"][:2]
        mask = np.load(f'./datasets/PanNuke/masks/{file_name}.npy',allow_pickle=True)
        semantic_mask = mask[...,1]!=0


        instances_mask = mask[...,0]
        max_instance_nums = np.max(instances_mask)
        if max_instance_nums > 1:
            instance_id = random.randint(1,max_instance_nums)
            instance_indices = np.where(instances_mask == instance_id)
            point_index = random.randint(0,len(instance_indices[0])-1)

            points = np.array([[instance_indices[1][point_index],instance_indices[0][point_index]]])


            point_label = np.array([1]*len(points)) #1 is foreground , 0 is background
        else:
            points = np.array([])
            point_label = np.array([])


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
        return batched_input, np.concatenate(masks,axis=0),image_path


class PanNuke_two_point_prompt(PanNuke_Dataset):
    def __getitem__(self, index):
        file_name = self.file_names[index]
        image_path = "./datasets/PanNuke/images/" + file_name + ".png"
        data = torch.load("./datasets/PanNuke/features/" + file_name + '.pt')
        feature = data["fm"] #.to(self.device)
        original_size = data["origin_shape"][:2]
        mask = np.load(f'./datasets/PanNuke/masks/{file_name}.npy',allow_pickle=True)
        semantic_mask = mask[...,1]!=0


        instances_mask = mask[...,0]
        max_instance_nums = np.max(instances_mask)
        points = []
        point_label = []
        if max_instance_nums > 1:
            instance_id = random.randint(1,max_instance_nums)
            instance_indices = np.where(instances_mask == instance_id)
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
        return batched_input, np.concatenate(masks,axis=0),image_path


class PanNuke_five_point_prompt(PanNuke_Dataset):
    def __getitem__(self, index):
        file_name = self.file_names[index]
        image_path = "./datasets/PanNuke/images/" + file_name + ".png"
        data = torch.load("./datasets/PanNuke/features/" + file_name + '.pt')
        feature = data["fm"] #.to(self.device)
        original_size = data["origin_shape"][:2]
        mask = np.load(f'./datasets/PanNuke/masks/{file_name}.npy',allow_pickle=True)
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
        return batched_input, np.concatenate(masks,axis=0),image_path


class PanNuke_Twenty_point_prompt(PanNuke_Dataset):
    def __getitem__(self, index):
        file_name = self.file_names[index]
        image_path = "./datasets/PanNuke/images/" + file_name + ".png"
        data = torch.load("./datasets/PanNuke/features/" + file_name + '.pt')
        feature = data["fm"] #.to(self.device)
        original_size = data["origin_shape"][:2]
        mask = np.load(f'./datasets/PanNuke/masks/{file_name}.npy',allow_pickle=True)
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
        return batched_input, np.concatenate(masks,axis=0),image_path


class PanNuke_All_point_prompt(PanNuke_Dataset):
    def __getitem__(self, index):
        file_name = self.file_names[index]
        image_path = "./datasets/PanNuke/images/" + file_name + ".png"
        data = torch.load("./datasets/PanNuke/features/" + file_name + '.pt')
        feature = data["fm"] #.to(self.device)
        original_size = data["origin_shape"][:2]
        mask = np.load(f'./datasets/PanNuke/masks/{file_name}.npy',allow_pickle=True)
        semantic_mask = mask[...,1]!=0


        instances_mask = mask[...,0]
        max_instance_nums = np.max(instances_mask)
        points = []
        point_label = []
        if max_instance_nums>1:
            for instance_id in range(1,max_instance_nums+1):
                instance_indices = np.where(instances_mask == instance_id)
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
        return batched_input, np.concatenate(masks,axis=0),image_path