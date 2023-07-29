import os
import argparse
import json
import numpy as np
import cv2

# detect for different classes
def convert_to_yolo_format(mask_path, label_path, class_num):
    mask = np.load(mask_path, allow_pickle=True)
    mask = mask.astype(np.int)
#     semantic_mask = mask[..., 1]!=0
    semantic_mask = mask[..., 1]
#     print("semantic_mask:", semantic_mask)
    instances_mask = mask[..., 0]
    max_instance_nums = np.max(instances_mask)
    instance_bboxes = []
    
    semantic_bboxes = []
    
#     # class detection
#     for instance_id in range(1, max_instance_nums + 1):
#         instance = (instances_mask == instance_id).astype(np.uint8) * 255
#         for class_id in range(1, class_num + 1):
#             semantic = (semantic_mask == class_id).astype(np.uint8) * instance
#             c1 = cv2.boundingRect(semantic)
#             if c1[2] <= 0 or c1[3] <= 0:
#                 continue
#             # Convert bounding box to YOLO Darknet format
#             img_width, img_height = instances_mask.shape[1], instances_mask.shape[0]
#             bbox_x = (c1[0] + c1[2] / 2) / img_width
#             bbox_y = (c1[1] + c1[3] / 2) / img_height
#             bbox_width = c1[2] / img_width
#             bbox_height = c1[3] / img_height
#             # Append the bounding box coordinates with class id to the list
#             train_class_id = class_id-1
#             semantic_bboxes.append([train_class_id, bbox_x, bbox_y, bbox_width, bbox_height])
    

    # instance detection
    for instance_id in range(1, max_instance_nums + 1):
        instance = (instances_mask == instance_id).astype(np.uint8) * 255

        c1 = cv2.boundingRect(instance)
        
        if c1[2] <= 0 or c1[3] <= 0:
            continue
        
        # Convert bounding box to YOLO Darknet format
        img_width, img_height = instances_mask.shape[1], instances_mask.shape[0]
        bbox_x = (c1[0] + c1[2] / 2) / img_width
        bbox_y = (c1[1] + c1[3] / 2) / img_height
        bbox_width = c1[2] / img_width
        bbox_height = c1[3] / img_height
        
        # Append the bounding box coordinates with class id to the list
        instance_bboxes.append([0, bbox_x, bbox_y, bbox_width, bbox_height])

    # Write the bounding boxes to a text file in YOLO Darknet format
    
    with open(label_path, "w") as txt_file:
        for bbox in instance_bboxes:
            line = " ".join([str(coord) for coord in bbox]) + "\n"
            txt_file.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MoNuSeg', help='the path of the dataset')
    args = parser.parse_args()
    
    dataset_root_dir = "../../data"
    dataset_name = args.dataset
    dataset_dir = os.path.join(dataset_root_dir, dataset_name)
    json_file = "data_split.json"
    json_path = os.path.join(dataset_root_dir, dataset_name, json_file)
    
    class_num = 1
    
 
    with open(json_path, "r") as f:
        data_split = json.load(f)

    # Get the list of train and valid data
    train_data = data_split["train"]
    valid_data = data_split["valid"]
    
    if dataset_name == 'MoNuSeg':
        class_num = 1
        
    elif dataset_name == 'SegPC-2021':
        class_num = 1
        
    elif dataset_name == 'bowl_2018':
        class_num = 1
        
    elif dataset_name == 'CryoNuSeg':
        class_num = 1
        
    elif dataset_name == 'TNBC':
        class_num = 1
        
    elif dataset_name == 'HuBMAP':
        class_num = 1
        
    elif dataset_name == 'bkai-igh-neopolyp':
        class_num = 1
        
    else:
        print("Wrong dataset")
        
#     print("class_num:", class_num)

    # generate validset labels
    for data_id in valid_data:
        suffix = os.path.splitext(data_id)[1]
        mask_path = os.path.join(dataset_dir,"masks",data_id.replace(suffix,".npy"))
        label_path = os.path.join(dataset_dir, "valid_ins/labels")
        if not os.path.exists(label_path):
            os.makedirs(label_path)

        txt_file_path = os.path.join(label_path, data_id.replace(suffix,".txt"))
        convert_to_yolo_format(mask_path, txt_file_path, class_num)
        
    # generate trainset labels    
    for data_id in train_data:
        suffix = os.path.splitext(data_id)[1]
        mask_path = os.path.join(dataset_dir,"masks",data_id.replace(suffix,".npy"))
        label_path = os.path.join(dataset_dir, "train_ins/labels")
        if not os.path.exists(label_path):
            os.makedirs(label_path)

        txt_file_path = os.path.join(label_path, data_id.replace(suffix,".txt"))
        convert_to_yolo_format(mask_path, txt_file_path, class_num)
        
        