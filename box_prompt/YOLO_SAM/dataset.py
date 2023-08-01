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


def get_path(dataset_dir,file_name,suffix):
    image_path = os.path.join(dataset_dir,"images",file_name)
    feature_path = os.path.join(dataset_dir,"features",file_name.replace(suffix,".pt"))
    mask_path = os.path.join(dataset_dir,"masks",file_name.replace(suffix,".npy"))
    return image_path,feature_path,mask_path

# yolo
def yolov8_detection(model, image_path):
#     print(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = model(image, stream=True)  # generator of Results objects
    results = model.predict(image, conf=0.01, iou=0.6)
#     results = model(image, stream=True, conf=0.01)
#     results = model(image, stream=True, max_det=500)

    boxes_list = []
    classes_list = []
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
#         print("boxes:", boxes)
        class_id = result.boxes.cls.long().tolist()
        boxes_list.append(boxes.xyxy.tolist())
        classes_list.append(class_id)
    
    

    bbox = [[int(i) for i in box] for boxes in boxes_list for box in boxes]
    class_id = [class_id for classes in classes_list for class_id in classes]
    
#     res_plotted = results[0].plot()
#     cv2.imshow("result", res_plotted)

    return bbox, class_id, image



class Dataset(data.Dataset):
    def __init__(self, json_path,split,device,model):
        with open(json_path, 'r') as f:
            ds_dict = json.load(f)
        self.dataset_dir = os.path.dirname(json_path)
        self.file_names = ds_dict[split]
        self.nF = len(self.file_names)
        self.device = device
        
        # add yolo model
        self.model = model
        
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
    
    
# Add box_prompt dataset
class Box_prompt_Dataset(Dataset):
    def __getitem__(self, index):
        file_name = self.file_names[index]
#         print("file_name:", file_name)
#         print("index:", index)

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
#         for instance_id in range(1,max_instance_nums+1):
#             instance =  (instances_mask==instance_id).astype(np.uint8)*255
#             c1 = cv2.boundingRect(instance)
#             if c1[2]<=0 or c1[3]<=0:
#                 continue
#             instance_bboxes.append([c1[0], c1[1], c1[0]+c1[2], c1[1]+c1[3]])

        yolov8_boxes,yolov8_class_id, image = yolov8_detection(self.model, image_path)
        instance_bboxes.append(yolov8_boxes)

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