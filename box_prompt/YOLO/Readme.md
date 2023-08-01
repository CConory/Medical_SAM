# Step1: Generate train & valid data
Generate YOLO training format images and labels  for train and valid dataset.

train data path: Medical_SAM/data/DATASET_NAME/train

vaild data path: Medical_SAM/data/DATASET_NAME/valid

```
python training_data.py --dataset DATASET_NAME
python training_label.py --dataset DATASET_NAME
```

# Step2: Write YAML file
Write YAML file for training data.

Change the contents for different datasets.

DATASET_NAME.yaml

```
python write_yaml.py
```

# Step2: Training YOLO model

```
# e.g.
python yolo_train.py \
			--data DATASET_NAME.yaml \
			--epochs 300 \
			--imgsz 640 \
			--batch 16 \
			--project pretrained \
			--name yolov8m_300e_imgsz640_dsbowl
			
Argument:
--data: the path of the data yaml
--epochs: the training epochs
--imgsz: the training image size
--batch: the training batch size
--project: the path of the training weights root directory
--name: the path of the training weights directory
```

Our pretrained weighs released here: https://drive.google.com/file/d/1cmu4aGJfCgoo1CFGHL15V9smaOzNa7PL/view?usp=sharing
