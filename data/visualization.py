import numpy as np
import cv2
import os
import random
import matplotlib.colors as mcolors

def random_color():
    r,g,b = random.uniform(0.8, 1)*255., random.uniform(0.6, 0.9)*255., random.uniform(0, 0.2)*255.
    return (r, g, b)

dataset_name="TNBC"
img_id = "TNBC_09_4.png"
category_color = {}
category_nums = 21

for i in range(1, 22):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    category_color[i] = (r, g, b)



img = cv2.imread(f"./{dataset_name}/images/{img_id}")
masks = np.load(f'./{dataset_name}/masks/{os.path.splitext(img_id)[0]}.npy',allow_pickle=True)
masks = masks.astype(int)

instances_mask = masks[...,0] # instance_id
semantic_mask = masks[...,1] # bg:0 , fg:1~5


# instance_example
instance_output_img = img.copy()
for i in range(1,np.max(instances_mask)+1):
    instance_id = instances_mask==i
    instance_output_img[instance_id] = (instance_output_img[instance_id]*0.7 + tuple(tmp*0.3 for tmp in random_color())).astype(np.uint8)
cv2.imwrite("./instance_example.jpg",instance_output_img)


# semantic_example
semantic_output_img = img.copy()
for i in range(1,category_nums):
    category_id = semantic_mask==i
    semantic_output_img[category_id] = (semantic_output_img[category_id]*0.4 + tuple(tmp*0.6 for tmp in category_color[i])).astype(np.uint8)
cv2.imwrite("./semantic_example.jpg",semantic_output_img)