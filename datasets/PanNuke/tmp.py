import os
import json

with open("./data_split.json", 'r') as f:
    ds_dict = json.load(f)

ds_dict['train'] = [tmp+".png" for tmp in ds_dict['train']]
ds_dict['valid'] = [tmp+".png" for tmp in ds_dict['valid']]
ds_dict['test'] = [tmp+".png" for tmp in ds_dict['test']]

with open("./data_split_1.json", 'w') as f:
    json.dump(ds_dict, f)