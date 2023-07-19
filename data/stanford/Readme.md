# Step1: Download the file 
```
bash download.sh

```

# Step2 : unzip the download files
```
unzip stanford.zip -d "Stanford Background Dataset"
```

# Step3: Convert the mask format
```
python convert.py
# all image files images will be moved into ./images/
# all mask files would generate to a numpy matrix e.g. XXXX.npy; shape (w,h,2)
# [...,0] is the instance mask, cell+nuclear is one instance
# [...,1] is the semantic mask Core is 1 ; cell is 2
```

# Step4: visualization the masks
```
# The dataset you want to check
dataset_name="stanford" 
# The image file name you want to check
img_id = "xxx"
# Enter the /PROJECT_PATH/datasets direction
cd ..  
# Run the script 
python ./visualization.py

```
class:
1. sky
2. tree
3. road
4. grass
5. water
6. building
7. mountain
8. foreground
9. unknown