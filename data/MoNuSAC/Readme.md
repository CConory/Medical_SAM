# Step1: Download the file 
```
bash download_test.sh
bash download_train.sh
```

# Step2 : unzip the download files
```
python ../unzip.py --dataset_name MoNuSAC
```

# Step3: split the train, val, test dataset 
```
python ./data_split.py
```

# Step4: Convert mask format
```
python convert.py
# all images will be moved into ./images/
# all mask files would generate to a numpy matrix e.g. XXXX.npy; shape (w,h,2)
# [...,0] is the instance mask
# [...,1] is the semantic mask
```

# Step5: Visualize the converted mask
```
# The dataset you want to check
dataset_name="MoNuSAC" 
# The image file name you want to check
img_id = "xxx"
# Enter the /PROJECT_PATH/datasets direction
cd ..  
# Run the script 
python ./visualization.py
```
class:
1. Epithelial
2. Lymphocyte
3. Neutrophil
4. Macrophage
5. Ambiguous