import os
import json
import argparse

def add_prefix_json(dataset_name):
    with open(dataset_name + "/data_split.json", 'r') as f:
        ds_dict = json.load(f)

    ds_dict['train'] = [dataset_name + "_" + tmp for tmp in ds_dict['train']]
    ds_dict['valid'] = [dataset_name + "_" + tmp for tmp in ds_dict['valid']]
    ds_dict['test'] = [dataset_name + "_" + tmp for tmp in ds_dict['test']]

    with open(dataset_name + "/data_split.json", 'w') as f:
        json.dump(ds_dict, f)

def add_prefix_file(dataset_name, folder_path):
    # 获取文件夹中的所有文件名
    file_list = os.listdir(folder_path)

    # 遍历文件列表
    for file_name in file_list:
        # 构造新的文件名
        new_file_name = dataset_name + "_" + file_name

        # 构造原始文件的完整路径
        original_file_path = os.path.join(folder_path, file_name)

        # 构造新文件的完整路径
        new_file_path = os.path.join(folder_path, new_file_name)

        # 重命名文件
        os.rename(original_file_path, new_file_path)

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Add prefix to files in a dataset folder')
parser.add_argument('--dataset_name', type=str, default='bowl_2018', help='Name of the dataset')
args = parser.parse_args()

dataset_name = args.dataset_name
img_path = dataset_name + "/images/"
mask_path = dataset_name + "/masks/"
feature_path = dataset_name + "/features/"

add_prefix_json(dataset_name)
add_prefix_file(dataset_name, img_path)
add_prefix_file(dataset_name, mask_path)
add_prefix_file(dataset_name, feature_path)
