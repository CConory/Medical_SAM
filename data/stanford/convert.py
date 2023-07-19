import pandas as pd
import os
from PIL import Image
import numpy as np
from skimage import measure
from tqdm import tqdm
import shutil
class_dict = pd.read_csv(os.path.join("Stanford Background Dataset", 'labels_class_dict.csv'))
# Get class names
class_names = class_dict['class_names'].tolist()
# Get class RGB values
class_rgb_values = class_dict[['r','g','b']].values.tolist()

# 将 class_names 和 class_rgb_values 转换为 class_mapping 字典
class_mapping = {}
for index, class_name in enumerate(class_names):
    class_rgb = class_rgb_values[index]
    class_mapping[index+1] = class_rgb

save_img_path = "./images/"
save_mask_path = "./masks/"
if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)
if not os.path.exists(save_mask_path):
    os.makedirs(save_mask_path)

def read_masks_from_folder(folder_path, class_mapping):
    # 获取文件夹中所有的文件名
    file_names = os.listdir(folder_path)

    for file_name in tqdm(file_names):
        file_path = os.path.join(folder_path, file_name)
        # 使用 Pillow 库读取图像
        image = Image.open(file_path)

        # 将图像转换为 NumPy 数组
        mask_np = np.array(image)

        # 初始化一个与图像大小相同的全零矩阵
        targets = np.zeros((mask_np.shape[0], mask_np.shape[1], 2))
        class_mask = np.zeros_like(mask_np[:, :, 0], dtype=np.uint8)

        # 将 RGB 值转换为对应的种类编号
        for class_name, class_value in class_mapping.items():
            class_pixels = np.all(mask_np == class_value, axis=-1)
            class_mask[class_pixels] = class_name

        # 对 class_mask 进行连通域分析
        labeled_mask = measure.label(class_mask)

        # 获取连通域的数量
        num_instances = np.max(labeled_mask)

        # 初始化一个与 class_mask 大小相同的全零矩阵，用于存储分配后的实例 ID
        instance_mask = np.zeros_like(class_mask, dtype=np.uint16)

        # 分配不同的实例 ID 给每个连通域
        for instance_id in range(1, num_instances + 1):
            instance_pixels = labeled_mask == instance_id
            instance_mask[instance_pixels] = instance_id

        targets[..., 0] = instance_mask
        targets[..., 1] = class_mask

        np.save(os.path.join(save_mask_path, "stanford_" + os.path.splitext(file_name)[0] + ".npy"), targets)


def copy_jpg_images(source_folder, destination_folder):
    # 获取源文件夹中所有的文件名
    file_names = os.listdir(source_folder)

    # 筛选出所有 JPG 图像文件
    jpg_files = [file_name for file_name in file_names if file_name.lower().endswith('.jpg')]

    # 复制每个 JPG 图像文件到目标文件夹中
    for jpg_file in jpg_files:
        source_path = os.path.join(source_folder, jpg_file)
        destination_path = os.path.join(destination_folder, "stanford_"+jpg_file)
        shutil.copyfile(source_path, destination_path)


# 定义源文件夹和目标文件夹的路径
source_folder_path = os.path.join("Stanford Background Dataset", 'images')
destination_folder_path = save_img_path

# # 调用函数进行复制
# copy_jpg_images(source_folder_path, destination_folder_path)

# 读取文件夹中的所有 mask 图像
folder_path = os.path.join("Stanford Background Dataset", 'labels_colored')
read_masks_from_folder(folder_path, class_mapping)


