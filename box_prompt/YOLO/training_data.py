import os
import json
import shutil
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MoNuSeg', help='the path of the dataset')
    args = parser.parse_args()
    
    dataset_root_dir = "../../data"
    dataset_name = args.dataset
    json_file = "data_split.json"
    json_path = os.path.join(dataset_root_dir, dataset_name, json_file)
#     print(json_path)

    with open(json_path, "r") as f:
        data_split = json.load(f)

    # Get the list of train and valid data
    train_data = data_split["train"]
    valid_data = data_split["valid"]

    source_dir = "images"
    source_directory = os.path.join(dataset_root_dir, dataset_name, source_dir)
#     print(source_directory)
    
    destination_dir_valid = "valid/images"
    destination_directory_valid = os.path.join(dataset_root_dir, dataset_name, destination_dir_valid)
#     print(destination_directory_valid)
    
    destination_dir_train = "train/images"
    destination_directory_train = os.path.join(dataset_root_dir, dataset_name, destination_dir_train)
#     print(destination_directory_train)
    
    if not os.path.exists(destination_directory_valid):
        os.makedirs(destination_directory_valid)

    if not os.path.exists(destination_directory_train):
        os.makedirs(destination_directory_train)

    for data_id in train_data:

        source_file = os.path.join(source_directory, data_id)
        destination_file = os.path.join(destination_directory_train, data_id)
        shutil.copy(source_file, destination_file)

    for data_id in valid_data:

        source_file = os.path.join(source_directory, data_id)
        destination_file = os.path.join(destination_directory_valid, data_id)
        shutil.copy(source_file, destination_file)
    
    