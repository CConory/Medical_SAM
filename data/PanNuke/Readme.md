# Step1: Download the file 
```
bash download.sh

```

# Step2 : unzip the download files
```
python ../unzip.py --dataset_name PanNuke
```
# Step3: Get the split json file from the csv file
```
python ./csv2json.py
```

# Step4: Convert the mask format
```
python convert.py
# all image files images will be moved into ./images/
# all mask files would generate to a numpy matrix e.g. XXXX.npy; shape (w,h,2)
# [...,0] is the instance mask, cell+nuclear is one instance
# [...,1] is the semantic mask Core is 1 ; cell is 2
```

# Step5: visualization the masks
```
# The dataset you want to check
dataset_name="SegPC-2021" 
# The image file name you want to check
img_id = "xxx"
# Enter the /PROJECT_PATH/datasets direction
cd ..  
# Run the script 
python ./visualization.py

```
class:
1. PCa

Description:
1. detecting PCa on images of prostate tissue samples