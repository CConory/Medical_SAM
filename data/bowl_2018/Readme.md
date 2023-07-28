# Step1: Download the file 
```
bash download.sh
```

# Step2 : unzip the download files

```
python ../unzip.py --dataset_name bowl_2018
```

# Step3: Change annotaion style from XML 2 mask
```
python convert.py
# all .png images will be moved into ./images/
# target would be a numpy matrix e.g. XXXX.npy; shape (w,h,2)
# [...,0] is the instance mask
# [...,1] is the semantic mask
```

# Step4: visualization the masks
```
# The dataset you want to check
dataset_name="bowl_2018" 
# The image file name you want to check
img_id = "f4b7c24baf69b8752c49d0eb5db4b7b5e1524945d48e54925bff401d5658045d"
# Enter the /PROJECT_PATH/datasets direction
cd ..  
# Run the script 
python ./visualization.py

```

# Step5: split the train, val, test dataset 
```
# Run the script 
python ../data_split.py --dataset xxx
```

class:
1. nuclei

description:
* It includes cell types, magnifications and imaging modalities (bright field or fluorescence) from different conditions and variations.