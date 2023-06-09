import json
import pandas as pd

df = pd.read_csv('./test_train_data.csv')

train_set = list(df[df.category=='train']['image_id'])
val_set = list(df[df.category=='validation']['image_id'])
test_set = list(df[df.category=='test']['image_id'])

ds_dict = {'train':train_set,
            'valid':val_set,
            'test': test_set
    }

with open("./data_split.json", 'w') as f:
    json.dump(ds_dict, f)
    
print('Number of train sample: {}'.format(len(train_set)))
print('Number of validation sample: {}'.format(len(val_set)))
print('Number of test sample: {}'.format(len(test_set)))