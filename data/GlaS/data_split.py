import os
import json

# 定义文件夹路径
folder_path = "./images"

# 创建用于存储结果的字典
data = {
    "train": [],
    "valid": [],
    "test": []
}

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.endswith(".bmp"):
        # 提取图像文件名（不包含文件扩展名）
        image_name = os.path.splitext(filename)[0]

        if image_name.startswith("train"):
            data["train"].append(image_name)
        elif image_name.startswith("testA"):
            data["valid"].append(image_name)
        elif image_name.startswith("testB"):
            data["test"].append(image_name)

# 写入JSON文件
json_path = "./data_split.json"
with open(json_path, "w") as json_file:
    json.dump(data, json_file, indent=4)

print("JSON文件已生成：", json_path)
