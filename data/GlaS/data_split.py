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
        if filename.startswith("train"):
            data["train"].append(filename)
        elif filename.startswith("testA"):
            data["valid"].append(filename)
        elif filename.startswith("testB"):
            data["test"].append(filename)

# 写入JSON文件
json_path = "./data_split.json"
with open(json_path, "w") as json_file:
    json.dump(data, json_file, indent=4)

print("JSON文件已生成：", json_path)
