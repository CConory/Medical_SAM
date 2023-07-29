# Combine YOLO model With SAM
Get the weights of YOLO model from yolo training. Apply YOLO model to predict bounding boxes. Convey bounding boxes as box prompt for SAM to achieve segmentation. 

yolo_sam_vis.py add visualization on wandb comparing with yolo_sam.py.

Modify the YOLO model weights path in yolo_sam.py first.

```
python yolo_sam.py --dataset DATASET_NAME
python yolo_sam_vis.py --dataset DATASET_NAME
```

