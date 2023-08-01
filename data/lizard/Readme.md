# Step1: Download the file 
```
bash download.sh
```

# Step2 : unzip the download files
```
python ../unzip.py --dataset_name lizard
```

# Step3: split the train, val, test dataset 


# Step4: Convert mask format
```
python convert.py
# all .png images will be moved into ./images/
# all .mat would generate to a numpy matrix e.g. XXXX.npy; shape (w,h,2)
# [...,0] is the instance mask
# [...,1] is the semantic mask
```

# Step5: Visualize the converted mask
```
# The dataset you want to check
dataset_name="lizard" 
# The image file name you want to check
img_id = "xxx"
# Enter the /PROJECT_PATH/datasets direction
cd ..  
# Run the script 
python ./visualization.py
```
classes:
1. Neutrophil
2. Epithelial
3. Lymphocyte
4. Plasma
5. Neutrophil
6. Connective tissue

description:
1. colonic nuclear instance segmentation and classification
