import os
from skimage import io, transform, color,img_as_ubyte
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensor

class sam_inputer(Dataset):
        def __init__(self,path,data):
            self.path = path
            self.feature_pt = '/userhome/cs2/xq141839/hku_project/Medical_SAM/segpc2021/features/'
            self.folders = data
            self.transforms = transform
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            file_name = str(self.folders[idx])
            image_id = list(file_name.split('.'))[0]

            ft_path = os.path.join(self.feature_pt,image_id)
            mask_folder = os.path.join(self.path,'y/',image_id)
            sam_dict = torch.load(ft_path + '.pt')
            sam_feature = sam_dict['fm']
            sam_feature = torch.squeeze(sam_feature) 
                  
            mask = self.get_mask(mask_folder, 256, 256)

            '''
            mask_exist = os.path.exists(f'cache/{file_name}')
            if mask_exist:
                mask = io.imread(f'cache/{file_name}', as_gray=True)
            else:
                mask = self.get_mask(mask_folder, 256, 256)
                mask_save = mask.astype(np.uint8)
                io.imsave(f'cache/{file_name}', mask_save)
            '''
            
            mask = img_as_ubyte(mask) 
            mask = np.squeeze(mask)
            mask[(mask > 0) & (mask < mask.max())] = 1
            mask[mask == mask.max()] = 2
            mask = torch.from_numpy(mask)
            mask = torch.squeeze(mask)
            mask = torch.nn.functional.one_hot(mask.to(torch.int64),3)
            mask = mask.permute(2, 0, 1)
            
            return (sam_feature, mask) 


        def get_mask(self,mask_folder,IMG_HEIGHT, IMG_WIDTH):
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
            for mask_ in os.listdir(mask_folder):
                    mask_ = io.imread(os.path.join(mask_folder,mask_), as_gray=True)
                    mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
                    mask_ = np.expand_dims(mask_,axis=-1)
                    mask = np.maximum(mask, mask_)

            return mask