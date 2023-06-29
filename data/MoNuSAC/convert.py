import cv2
import xml.etree.ElementTree as ET
import numpy as np
import os
from tqdm import tqdm
from shapely.geometry import Polygon
from skimage import draw

save_img_path = "./images"
save_mask_path = "./masks"
if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)
if not os.path.exists(save_mask_path):
    os.makedirs(save_mask_path)
label_dict = {
    'Epithelial': 1,
    'Lymphocyte': 2,
    'Neutrophil': 3,
    'Macrophage': 4,
    'Ambiguous': 5
}
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

                # Generate binary mask for each cell-type
                instance_id = 0
                binary_mask = np.zeros((*img.shape[:2], 2), dtype='int16')
                for k in range(len(root)):
                    label = [x.attrib['Name'] for x in root[k][0]]
                    label = label[0]

                    for child in root[k]:
                        for x in child:
                            r = x.tag
                            if r == 'Attribute':
                                label = x.attrib['Name']
                            if r == 'Region':
                                instance_id += 1
                                regions = []
                                vertices = x[1]
                                coords = np.zeros((len(vertices), 2))
                                for i, vertex in enumerate(vertices):
                                    coords[i][0] = vertex.attrib['X']
                                    coords[i][1] = vertex.attrib['Y']
                                regions.append(coords)

                                coordinates_array = np.array(regions[0], dtype=np.int32)
                                # Create a blank image with the same shape as binary_mask[..., 0]
                                instance_blank_image = np.zeros_like(binary_mask[..., 0])
                                semantic_blank_image = np.zeros_like(binary_mask[..., 1])
                                # Use cv2.fillPoly to create polygon on the blank image
                                cv2.fillPoly(instance_blank_image, [coordinates_array], instance_id)
                                cv2.fillPoly(semantic_blank_image, [coordinates_array], label_dict[label])
                                # Assign the filled polygon to binary_mask[..., 0]
                                binary_mask[..., 0] = np.where(instance_blank_image > 0, instance_id, binary_mask[..., 0])
                                binary_mask[..., 1] = np.where(semantic_blank_image > 0, label_dict[label], binary_mask[..., 1])

                unique_values = np.unique(binary_mask[..., 0]).tolist()[1:]
                new_instance_id = 1
                for index, ins_id in enumerate(unique_values):
                    instance_mask = binary_mask[..., 0] == ins_id
                    binary_mask[..., 0][instance_mask] = ins_id
                    num_labels, labels = cv2.connectedComponents(
                        instance_mask.astype(np.uint8) * 255)  # 计算联通域，instance——label的误差，导致同一个instance_id对应多个obj
                    if num_labels > 2:
                        for i in range(2, num_labels):
                            binary_mask[..., 0][labels == i] = unique_values[-1] + new_instance_id
                            new_instance_id += 1

                np.save(os.path.join(save_mask_path, "MoNuSAC_" + os.path.splitext(file)[0] + ".npy"), binary_mask)


if __name__ == '__main__':
    # train_path = "MoNuSAC_images_and_annotations"
    test_path = "MoNuSAC Testing Data and Annotations"
    # convert(train_path)
    convert(test_path)
