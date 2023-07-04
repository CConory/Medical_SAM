import cv2
import os
import numpy as np
from tqdm import tqdm


'''
0: background (non tissue) or unknown
1~5 : 上皮细胞核、淋巴细胞核、结缔组织细胞核、恶性细胞核和无法识别的细胞核
'''

def flat_for(a, f):
    a = a.reshape(-1)
    for i, v in enumerate(a):
        a[i] = f(v)


# A helper function to unique PanNuke instances indexes to [0..N] range where 0 is background
def map_inst(inst):
    seg_indexes = np.unique(inst)
    new_indexes = np.array(range(0, len(seg_indexes)))
    dict = {}
    for seg_index, new_index in zip(seg_indexes, new_indexes):
        dict[seg_index] = new_index

    flat_for(inst, lambda x: dict[x])


# A helper function to transform PanNuke format to HoverNet data format
def transform(images, masks, img_out_dir,mask_out_dir, count):

    for i in tqdm(range(len(images))):
        

        np_file = np.zeros((256,256,2), dtype='int16')
        image = images[i]
        # RGB 2 BGR and save
        image = image[...,::-1] 
        cv2.imwrite(os.path.join(img_out_dir, str(count)+".png"),image)

        # convert inst and type format for mask
        msk = masks[i]

        inst = np.zeros((256,256))
        for j in range(5):
            #copy value from new array if value is not equal 0
            inst = np.where(msk[:,:,j] != 0, msk[:,:,j], inst)
        map_inst(inst)

        types = np.zeros((256,256))
        for j in range(5):
            # write type index if mask is not equal 0 and value is still 0
            types = np.where((msk[:,:,j] != 0) & (types == 0), j+1, types) # 1,2,3,4,5 (category)

        # add padded inst and types to array
        np_file[:,:,0] = inst
        np_file[:,:,1] = types

        np.save(os.path.join(mask_out_dir, str(count)+".npy"), np_file) # instance_id + category_type_id
        count += 1
    return count
        


save_img_path = "./images/"
save_mask_path = "./masks/"
if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)
if not os.path.exists(save_mask_path):
    os.makedirs(save_mask_path)

count = 0
images = np.load('./Part1/images.npy',allow_pickle=True)
masks = np.load('./Part1/masks.npy',allow_pickle=True)
count = transform (images,masks,save_img_path,save_mask_path,count)

images = np.load('./Part2/Images/images.npy',allow_pickle=True)
masks = np.load('./Part2/Masks/masks.npy',allow_pickle=True)
count = transform (images,masks,save_img_path,save_mask_path,count)

images = np.load('./Part3/Images/images.npy',allow_pickle=True)
masks = np.load('./Part3/Masks/masks.npy',allow_pickle=True)
count = transform (images,masks,save_img_path,save_mask_path,count)