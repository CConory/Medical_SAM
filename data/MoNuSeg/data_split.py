import json
import numpy as np
import os

# # check test name is in train_val or not
# test_dir = "/userhome/cs2/kuangww/medical_sam/datasets/MoNuSeg/MoNuSegTestData"

# img_names = os.listdir(test_dir)
# test_set = [tmp for tmp in img_names if os.path.splitext(tmp)[1] == ".tif"]

# train_val_dir = "/userhome/cs2/kuangww/medical_sam/datasets/MoNuSeg/MoNuSeg_2018_Training_Data/Images"
# train_val_set = os.listdir(train_val_dir)

# print([tmp for tmp in test_set if tmp in train_val_set])
# exit(0)

test_dir = "MoNuSegTestData"

img_names = os.listdir(test_dir)
test_set = [tmp for tmp in img_names if os.path.splitext(tmp)[1] == ".tif"]

train_val_dir = "MoNuSeg_2018_Training_Data/Images"
train_val_set = os.listdir(train_val_dir)


np.random.seed(42)
val_size = len(test_set)

val_set = np.random.choice(train_val_set,val_size,replace=False)

train_set = [ tmp  for tmp in train_val_set if tmp not in val_set] 

ds_dict = {'train':list(train_set),
            'valid':list(val_set),
            'test': test_set
    }

with open("./data_split.json", 'w') as f:
    json.dump(ds_dict, f)
print('Number of train sample: {}'.format(len(train_set)))
print('Number of validation sample: {}'.format(len(val_set)))
print('Number of test sample: {}'.format(len(test_set)))
