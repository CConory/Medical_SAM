import json
import numpy as np
import cv2
import os


# 读取JSONL文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


# 提取每个ID的类型和坐标
def extract_coordinates(data):
    coordinates = {}
    for item in data:
        item_id = item['id']
        annotations = item['annotations']
        for annotation in annotations:
            annotation_type = annotation['type']
            annotation_coordinates = annotation['coordinates']
            if item_id not in coordinates:
                coordinates[item_id] = {}
            if annotation_type not in coordinates[item_id]:
                coordinates[item_id][annotation_type] = []
            coordinates[item_id][annotation_type].append(annotation_coordinates)
    return coordinates


# 合并坐标为一个instance_mask
def merge_coordinates(coordinates):
    instance_masks = {}
    semantic_masks = {}
    for item_id, annotation_types in coordinates.items():
        tif_file_path = f'./train/{item_id}.tif'  # 图像文件路径，请根据实际文件位置进行调整
        img = cv2.imread(tif_file_path, cv2.IMREAD_UNCHANGED)
        instance_mask = np.zeros(img.shape[:2])
        semantic_mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)
        for annotation_type, annotation_coordinates in annotation_types.items():
            for coordinates_list in annotation_coordinates:
                coordinates_array = np.array(coordinates_list, dtype=np.int32)
                cv2.fillPoly(instance_mask, [coordinates_array], 1)
                if annotation_type == "blood_vessel":
                    cv2.fillPoly(semantic_mask, [coordinates_array], 1)
                if annotation_type == "glomerulus":
                    cv2.fillPoly(semantic_mask, [coordinates_array], 2)
                if annotation_type == "unsure":
                    cv2.fillPoly(semantic_mask, [coordinates_array], 3)

        mask = (instance_mask != 0)
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8) * 255)  # 计算联通域
        instance_id = 0
        for i in range(1, num_labels):
            instance_id += 1
            instance_mask[labels == i] = instance_id
        instance_masks[item_id] = instance_mask
        semantic_masks[item_id] = semantic_mask
    return instance_masks, semantic_masks


def merge_instance_semantic_masks(instance_masks, semantic_masks):
    obj_masks = {}
    for item_id in instance_masks:
        obj_mask = np.zeros((*instance_masks[item_id].shape, 2), dtype=np.uint8)
        obj_mask[..., 0] = instance_masks[item_id]
        obj_mask[..., 1] = semantic_masks[item_id]
        obj_masks[item_id] = obj_mask
    return obj_masks


# 保存obj_mask到文件
def save_obj_mask(obj_mask, item_id, save_dir):
    save_path = os.path.join(save_dir, f'HuBMAP_{item_id}.npy')
    np.save(save_path, obj_mask)


save_mask_path = 'masks'  # 保存文件的文件夹路径，请根据实际需求进行调整
save_img_path = "images"


def move_tif_images(coordinates):
    for item_id, annotation_types in coordinates.items():
        tif_file_path = f'./train/{item_id}.tif'  # 图像文件路径，请根据实际文件位置进行调整
        os.rename(tif_file_path, os.path.join(save_img_path, "HuBMAP_" + item_id + ".tif"))


file_path = './polygons.jsonl'
data = read_jsonl(file_path)
coordinates = extract_coordinates(data)
instance_masks, semantic_masks = merge_coordinates(coordinates)
obj_masks = merge_instance_semantic_masks(instance_masks, semantic_masks)

if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)
if not os.path.exists(save_mask_path):
    os.makedirs(save_mask_path)

for item_id, obj_mask in obj_masks.items():
    save_obj_mask(obj_mask, item_id, save_mask_path)
move_tif_images(coordinates)
