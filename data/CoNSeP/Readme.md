# Step1: Download the file 
```
bash download.sh
```

# Step2 : unzip the download files
```
python ../unzip.py --dataset_name CoNSeP
```

# Step3: split the train, val, test dataset 
```
# Run the script 
python ./data_split.py
```

# Step4: Convert the mask format
```
python convert.py
# all .png images will be moved into ./images/
# target would be a numpy matrix e.g. XXXX.npy; shape (w,h,2)
# [...,0] is the instance mask
# [...,1] is the semantic mask
```

# Step5: visualization the masks
```
# The dataset you want to check
dataset_name="CoNSeP" 
# The image file name you want to check
img_id = "xxx"
# Enter the /PROJECT_PATH/datasets direction
cd ..  
# Run the script 
python ./visualization.py

```
Class values:
1. other
2. inflammatory
3. healthy epithelial
4. dysplastic/malignant epithelial
5. fibroblast
6. muscle
7. endothelial

description:
* The images were extracted from 16 colorectal adenocarcinoma (CRA) WSIs, each belonging to an individual patient,
