# Step1: Conver data format
# Step2: Extract the image embedding
```
# e.g.
python extract_fm \
    --dataset PanNuke \
    --weight_path WEIGHT_DIR/sam_vit_h_4b8939.pth \
    --model_type vit_h

Argument:
dataset: the folder name in ./dataset
weight_path: the path of pretrained model weight, which is download from SAM
model_type: should correspone to the weight_path, which could be checked at SAM'github
Output:
The image embedding would be save as a torch.tensor in ./datasets/DATASET_NAME/features
```
