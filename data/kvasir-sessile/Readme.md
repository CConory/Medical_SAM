# Step1: Download the file 
```
bash download.sh
```

# Step2 : unzip the download files
```
python ../unzip.py --dataset_name Kvasir-SEG
```

# Step3: Convert the mask format
```
python convert.py
# all .jpg images will be moved into ./images/
# target would be a numpy matrix e.g. XXXX.npy; shape (w,h,2)
# [...,0] is the instance mask
# [...,1] is the semantic mask
```

# Step4: split the train, val, test dataset 
```
# Run the script 
python ../data_split.py --dataset Kvasir-SEG
```

# Step5: visualization the masks
```
# The dataset you want to check
dataset_name="Kvasir-SEG" 
# The image file name you want to check
img_id = "xxx"
# Enter the /PROJECT_PATH/datasets direction
cd ..  
# Run the script 
python ./visualization.py

```
Class values:
1. polyps

description: 
1. Multiclass image dataset for gastrointestinal disease detection
