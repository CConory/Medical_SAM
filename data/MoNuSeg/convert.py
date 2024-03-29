import cv2
import xml.etree.ElementTree as ET
import numpy as np
import os
from tqdm import tqdm

save_img_path = "./images/"
save_mask_path = "./masks/"
if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)
if not os.path.exists(save_mask_path):
    os.makedirs(save_mask_path)

img_dir = "MoNuSegTestData"
xml_dir = "MoNuSegTestData"

img_names = os.listdir(img_dir)
img_names = [tmp for tmp in img_names if os.path.splitext(tmp)[1] == ".tif"]
for img_name in tqdm(img_names):
    img_path = os.path.join(img_dir,img_name)
    img = cv2.imread(img_path)

    os.rename(img_path,os.path.join(save_img_path,img_name))

    xml_path=os.path.join(xml_dir,img_name).replace(".tif",".xml")

    tree = ET.ElementTree(file=xml_path)
    root = tree.getroot()
    annotation = root.find('Annotation')
    regions = annotation.find('Regions')
    region = regions.findall('Region')


    np_file = np.zeros((*img.shape[:2],2), dtype='int16')

    instance_id = 0
    for ins_id,region in enumerate(regions.findall('Region')):
        instance_id = instance_id + 1
        class_id = int(region.attrib['Type'])+1
        
        X = []
        Y = []
        obj_mask = np.zeros(img.shape[:2])
        for verticie in region.find('Vertices').findall('Vertex'):
            X.append(float(verticie.attrib['X']))
            Y.append(float(verticie.attrib['Y']))
        
        cor_xy = np.vstack((X, Y)).T
        cor_xy = cor_xy.astype(int)

        # edge_mask = cv2.polylines(obj_mask,[cor_xy],True,1,1)
        obj_mask = cv2.fillPoly(obj_mask, [cor_xy], 1)
        
        instance_mask = obj_mask != 0
        np_file[...,0][obj_mask!=0] = instance_id
        np_file[...,1][obj_mask!=0] = class_id
        
        num_labels, labels = cv2.connectedComponents(
            instance_mask.astype(np.uint8) * 255)  # 计算联通域，instance——label的误差，导致同一个instance_id对应多个obj
        if num_labels > 2:
            for i in range(2, num_labels):
                instance_id += 1
                np_file[..., 0][labels == i] = instance_id
                np_file[..., 1][labels == i] = class_id
    np.save(os.path.join(save_mask_path,os.path.splitext(img_name)[0]+".npy"), np_file) 
