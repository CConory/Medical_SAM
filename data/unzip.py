import zipfile
import os
import argparse

def unzip_file(extract_folder):
    # 解压ZIP文件
    if extract_folder == "MoNuSAC":
        for zip_save_path in ["MoNuSAC/MoNuSAC_images_and_annotations.zip", "MoNuSAC/MoNuSAC.zip"]:
            with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
            # 删除ZIP文件（可选）
            os.remove(zip_save_path)
    elif extract_folder == "MoNuSeg":
        for zip_save_path in ["MoNuSeg/MoNuSeg_train_Dataset.zip", "MoNuSeg/MoNuSeg_test_Dataset.zip"]:
            with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
            # 删除ZIP文件（可选）
            os.remove(zip_save_path)
    elif extract_folder == "SegPC-2021":
        for zip_save_path in ["SegPC-2021/valid.zip", "SegPC-2021/train.zip"]:
            with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
            # 删除ZIP文件（可选）
            os.remove(zip_save_path)
    else:
        zip_save_path = extract_folder + "/" + extract_folder + ".zip"
        with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        # 删除ZIP文件（可选）
        os.remove(zip_save_path)


# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Unzip the dataset zip file in a dataset folder')
parser.add_argument('--dataset_name', type=str, default='CryoNuSeg', help='Name of the dataset')
args = parser.parse_args()

dataset_name = args.dataset_name
unzip_file(dataset_name)
