import json
import numpy as np
import os
import argparse


def pre_csv(data_path, frac):
    np.random.seed(42)
    image_ids = os.listdir(data_path)
    # image_ids = [os.path.splitext(tmp)[0] for tmp in image_ids]
    data_size = len(image_ids)

    val_size = int(round(data_size * frac, 0))

    val_set = np.random.choice(image_ids, val_size, replace=False)
    test_set = [tmp for tmp in image_ids if tmp not in val_set]

    ds_dict = {'train': None,
               'valid': list(val_set),
               'test': test_set
               }
    with open(os.path.join(os.path.dirname(data_path), "data_split.json"), 'w') as f:
        json.dump(ds_dict, f)

    print('Number of validation sample: {}'.format(len(val_set)))
    print('Number of test sample: {}'.format(len(test_set)))


pre_csv("./images", 0.5)
