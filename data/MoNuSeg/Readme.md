# Step1: Download the file 
## Notice to change the fileid and name for different set and run repeatedly
```
sh download.sh
```

# Step2 : unzip the download files
```
python ../unzip.py --dataset_name MoNuSeg
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
dataset_name="MoNuSeg" 
# The image file name you want to check
img_id = "xxx"
# Enter the /PROJECT_PATH/datasets direction
cd ..  
# Run the script 
python ./visualization.py
```