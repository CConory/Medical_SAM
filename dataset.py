'''
Pytorch Dataloader
'''
import json
import numpy as np
import torch
import torch.utils.data as data


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