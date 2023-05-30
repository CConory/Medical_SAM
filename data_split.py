import json
import numpy as np
import os
import argparse


def pre_csv(data_path,frac):
    np.random.seed(42)
    image_ids = os.listdir(data_path)
    image_ids = [os.path.splitext(tmp)[0] for tmp in image_ids]
    data_size = len(image_ids)

    train_size = int(round(len(image_ids) * frac, 0))
    val_size = int((data_size-train_size)//2)

    train_set = np.random.choice(image_ids,train_size,replace=False)
    val_test_set = [tmp for tmp in image_ids if tmp not in train_set]
    val_set = np.random.choice(val_test_set,val_size,replace=False)

    test_set = [ tmp  for tmp in val_test_set if tmp not in val_set] 

    ds_dict = {'train':list(train_set),
               'valid':list(val_set),
               'test': test_set
        }
    
    with open(os.path.join(os.path.dirname(data_path),"data_split.json"), 'w') as f:
        json.dump(ds_dict, f)
    print('Number of train sample: {}'.format(len(train_set)))
    print('Number of validation sample: {}'.format(len(val_set)))
    print('Number of test sample: {}'.format(len(test_set)))




    # val_size = int(round(len(train_set) * 0.2 / 0.9, 0))
    # val_set = np.random.choice(train_set,val_size,replace=False)

    # import pdb;pdb.set_trace()
    # ds_split = []
    # for img_id in image_ids:
    #     if img_id in val_set:
    #         ds_split.append('validation')
    #     elif img_id in train_set:
    #         ds_split.append('train')
    #     else:
    #         ds_split.append('test')
    
    # ds_dict = {'image_id':image_ids,
    #            'category':ds_split 
    #     }
    # df = pd.DataFrame(ds_dict)
    # df.to_csv('segpc2021/test_train_data.csv',index=False)
    # print('Number of train sample: {}'.format(len(train_set)-len(val_set)))
    # print('Number of validation sample: {}'.format(len(val_set)))
    # print('Number of test sample: {}'.format(data_size-train_size))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', type=str, default='data/', help='the path of dataset')
    parser.add_argument('--dataset', type=str, default='segpc2021/data/images/x', help='the path of images') # issue 16
    parser.add_argument('--size', type=float, default=0.8, help='the size of your train set')
    args = parser.parse_args()
    # os.makedirs('src/',exist_ok=True)
    pre_csv(args.dataset,args.size)