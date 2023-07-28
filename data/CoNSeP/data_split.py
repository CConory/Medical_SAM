import json
import numpy as np
import os
from tqdm import tqdm

test_dir = "./CoNSeP/Test"
train_val_dir = "./CoNSeP/Train"

train_val_img_dir = train_val_dir + "/Images"
test_img_dir = test_dir + "/Images"

train_val_img_names = os.listdir(train_val_img_dir)
test_img_names = os.listdir(test_img_dir)

test_set = []
test_set += ["CoNSeP_" + file_name for file_name in test_img_names if os.path.splitext(file_name)[1] == ".png"]

temp_set = []
temp_set += ["CoNSeP_" + file_name for file_name in train_val_img_names if os.path.splitext(file_name)[1] == ".png"]

# 计算划分的索引位置
np.random.seed(42)
val_size = len(test_set)
val_set = np.random.choice(temp_set, val_size, replace=False)
train_set = [tmp for tmp in temp_set if tmp not in val_set]

ds_dict = {'train': list(train_set),
           'valid': list(val_set),
           'test': test_set
           }

with open("./data_split.json", 'w') as f:
    json.dump(ds_dict, f)
print('Number of train sample: {}'.format(len(train_set)))
print('Number of validation sample: {}'.format(len(val_set)))
print('Number of test sample: {}'.format(len(test_set)))
