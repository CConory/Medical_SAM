import json
import numpy as np
import os
from tqdm import tqdm

test_dir = "./MoNuSAC Testing Data and Annotations"
train_val_dir = "./MoNuSAC_images_and_annotations"

test_ids = next(os.walk(test_dir))[1]
test_set = []
for file_name in tqdm(test_ids):
    file_path = os.path.join(test_dir, file_name)
    # 获取指定路径下的所有文件名
    f_names = os.listdir(file_path)
    test_set += ["MoNuSAC_"+tmp for tmp in f_names if os.path.splitext(tmp)[1] == ".tif"]

train_val_ids = next(os.walk(train_val_dir))[1]
train_set = []
val_set = []
temp_set = []
for file_name in tqdm(train_val_ids):
    file_path = os.path.join(train_val_dir, file_name)
    # 获取指定路径下的所有文件名
    f_names = os.listdir(file_path)
    temp_set += ["MoNuSAC_"+tmp for tmp in f_names if os.path.splitext(tmp)[1] == ".tif"]

# 计算划分的索引位置
np.random.seed(42)
val_size = len(test_set)
val_set = np.random.choice(temp_set,val_size,replace=False)
train_set = [ tmp for tmp in temp_set if tmp not in val_set]

ds_dict = {'train':list(train_set),
            'valid':list(val_set),
            'test': test_set
    }

with open("./data_split.json", 'w') as f:
    json.dump(ds_dict, f)
print('Number of train sample: {}'.format(len(train_set)))
print('Number of validation sample: {}'.format(len(val_set)))
print('Number of test sample: {}'.format(len(test_set)))