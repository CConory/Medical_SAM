import cv2
import xml.etree.ElementTree as ET
import numpy as np
import os
from tqdm import tqdm

save_img_path = "./images"
save_mask_path = "./masks"
if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)
if not os.path.exists(save_mask_path):
    os.makedirs(save_mask_path)

def convert(root_path):
    train_ids = next(os.walk(root_path))[1]
    for file_name in tqdm(train_ids):
        file_path = os.path.join(root_path, file_name)
        # 获取指定路径下的所有文件名
        f_names = os.listdir(file_path)
        # 遍历文件名列表
        for file in f_names:
            # 检查文件名是否以 ".tif" 结尾
            if file.endswith(".tif"):
                # 构建对应的 xml 文件名
                xml_file_name = file.replace(".tif", ".xml")
                # 构建文件的完整路径
                tif_file_path = os.path.join(file_path, file)
                xml_file_path = os.path.join(file_path, xml_file_name)

                img = cv2.imread(tif_file_path)
                os.rename(tif_file_path, os.path.join(save_img_path, "MoNuSAC_" + file))

                tree = ET.ElementTree(file=xml_file_path)
                root = tree.getroot()
                annotation = root.find('Annotation')
                regions = annotation.find('Regions')
                region = regions.findall('Region')

                np_file = np.zeros((*img.shape[:2], 2), dtype='int16')

                instance_id = 0
                for ins_id, region in enumerate(regions.findall('Region')):
                    instance_id = instance_id + 1
                    class_id = int(region.attrib['Type']) + 1

                    X = []
                    Y = []
                    obj_mask = np.zeros(img.shape[:2])
                    for vertices in region.find('Vertices').findall('Vertex'):
                        X.append(float(vertices.attrib['X']))
                        Y.append(float(vertices.attrib['Y']))

                    cor_xy = np.vstack((X, Y)).T
                    cor_xy = cor_xy.astype(int)

                    # edge_mask = cv2.polylines(obj_mask,[cor_xy],True,1,1)
                    obj_mask = cv2.fillPoly(obj_mask, [cor_xy], 1)

                    instance_mask = obj_mask != 0
                    np_file[..., 0][instance_mask] = instance_id
                    np_file[..., 1][instance_mask] = class_id

                    num_labels, labels = cv2.connectedComponents(
                        instance_mask.astype(np.uint8) * 255)  # 计算联通域，instance——label的误差，导致同一个instance_id对应多个obj
                    if num_labels > 2:
                        for i in range(2, num_labels):
                            instance_id += 1
                            np_file[..., 0][labels == i] = instance_id
                            np_file[..., 1][labels == i] = class_id
                np.save(os.path.join(save_mask_path, "MoNuSAC_"+os.path.splitext(file)[0] + ".npy"), np_file)

if __name__ == '__main__':
    train_path = "MoNuSAC_images_and_annotations"
    test_path = "MoNuSAC Testing Data and Annotations"
    convert(train_path)
    convert(test_path)
