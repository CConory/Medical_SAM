import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Analyze the data distribution for each dataset')
parser.add_argument('--dataset_name', type=str, default='bowl_2018', help='Name of the dataset')
args = parser.parse_args()

dataset = args.dataset_name

images_folder = os.path.join(dataset, 'images')
masks_folder = os.path.join(dataset, 'masks')
mask_channel_instance = 0
mask_channel_semantic = 1

object_sizes = []
object_density = []
image_categories = []
image_sizes = []
category_counts = {}

for filename in os.listdir(images_folder):
    # 读取图像
    image_path = os.path.join(images_folder, filename)
    image = cv2.imread(image_path)

    # 获取图像尺寸
    image_size = image.shape[:2]  # (height, width)
    image_sizes.append(image_size)

    # 获取对应的掩码文件名
    image_name, image_ext = os.path.splitext(filename)
    mask_filename = image_name + '.npy'
    mask_path = os.path.join(masks_folder, mask_filename)

    # 读取掩码
    mask = np.load(mask_path)

    # 分析实例掩码
    instance_mask = mask[..., mask_channel_instance]
    # 计算总像素数量
    total_pixels = instance_mask.shape[0] * instance_mask.shape[1]
    unique_ids, counts = np.unique(instance_mask, return_counts=True)
    unique_ids = unique_ids[1:]
    counts = counts[1:]
    object_sizes.extend(counts/total_pixels)

    # 计算实例掩码的密集程度
    # 统计实例掩码中被标记的像素数量
    labeled_pixels = np.sum(instance_mask > 0)
    # 计算密集度
    density = labeled_pixels / total_pixels
    object_density.append(density)

    # 分析语义掩码
    semantic_mask = mask[..., mask_channel_semantic]
    unique_categories, classcounts = np.unique(semantic_mask, return_counts=True)

    # 更新类别数量字典
    for category, classcount in zip(unique_categories, classcounts):
        if category not in category_counts:
            category_counts[category] = classcount
        else:
            category_counts[category] += classcount

# 检查并删除键为0的键值对
if 0 in category_counts:
    del category_counts[0]

# 将图像尺寸转换为（width, height）的格式
image_sizes_pair = np.array([(height, width) for width, height in image_sizes])

# 统计图像尺寸的频数
unique_sizes, size_counts = np.unique(image_sizes_pair, axis=0, return_counts=True)

image_sizes = np.array(image_sizes)
# 创建一个包含4个子图的图像窗口
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# 对象大小分布
counts, bins, patches = axs[0, 0].hist(object_sizes, bins=20)
axs[0, 0].set_xlabel('Object Size')
axs[0, 0].set_ylabel('Count')
axs[0, 0].set_title('Object Size Distribution')
# 在每个柱子上添加计数值
for count, patch in zip(counts, patches):
    x = patch.get_x() + patch.get_width() / 2
    y = patch.get_height()
    axs[0, 0].text(x, y, int(count), ha='center', va='bottom', fontsize=4)
# 设置x轴刻度字体大小
axs[0, 1].tick_params(axis='x', labelsize=7)
# 添加刻度线和刻度值
plt.xticks(bins)

# 对象密集程度分布
counts, bins, patches = axs[0, 1].hist(object_density, bins=20)
axs[0, 1].set_xlabel('Object Density')
axs[0, 1].set_ylabel('Count')
axs[0, 1].set_title('Object Density Distribution')
# 在每个柱子上添加计数值
for count, patch in zip(counts, patches):
    x = patch.get_x() + patch.get_width() / 2
    y = patch.get_height()
    axs[0, 1].text(x, y, int(count), ha='center', va='bottom', fontsize=4)
# 设置x轴刻度字体大小
axs[0, 1].tick_params(axis='x', labelsize=7)
# 添加刻度线和刻度值
plt.xticks(bins)

# 图像类别分布
categories = [category for category in category_counts]
class_counts = [category_counts[category] for category in category_counts]
axs[1, 0].bar(range(len(category_counts)), class_counts)
axs[1, 0].set_xlabel('Category')
axs[1, 0].set_ylabel('Count')
axs[1, 0].set_title('Category Distribution')
axs[1, 0].set_xticks(range(len(category_counts)))
axs[1, 0].set_xticklabels(categories, rotation=90)
# 在每个柱子上添加计数值
for i, count in enumerate(class_counts):
    x = i
    y = count
    axs[1, 0].text(x, y, str(count), ha='center', va='bottom', fontsize=4)

if 1<len(unique_sizes)<20:
    # 图像尺寸分布
    axs[1, 1].bar(range(len(unique_sizes)), size_counts)
    # 设置刻度标签
    axs[1, 1].set_xticks(range(len(unique_sizes)))
    axs[1, 1].set_xticklabels([f"{size[0]}x{size[1]}" for size in unique_sizes], rotation=90, fontsize=8)
    axs[1, 1].set_xlabel('Image Size')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].set_title('Image Size Distribution')
else:
    # 图像尺寸分布
    axs[1, 1].scatter(image_sizes[:, 1], image_sizes[:, 0], marker='.')
    axs[1, 1].set_xlabel('Image Width')
    axs[1, 1].set_ylabel('Image Height')
    axs[1, 1].set_title('Image Size Distribution')

    # 设置x轴刻度和标签
    x_ticks = axs[1, 1].get_xticks()
    axs[1, 1].set_xticks(x_ticks)
    axs[1, 1].set_xticklabels([int(x) for x in x_ticks])

# 调整子图之间的间距
plt.tight_layout()
# 保存图像
fig.savefig('./data_analysis_result/'+dataset+'_analysis.png', dpi=300)
# 显示图像
plt.show()