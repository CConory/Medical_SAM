# Step1: Download the file 
```
bash download.sh
```

# Step2 : unzip the download files
```
python ../unzip.py --dataset_name NuCLS
```

# Step3: Convert the mask format
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
dataset_name="NuCLS" 
# The image file name you want to check
img_id = "xxx"
# Enter the /PROJECT_PATH/datasets direction
cd ..  
# Run the script 
python ./visualization.py

```

# Step5: split the train, val, test dataset 
```
# Run the script 
python ../data_split.py --dataset NuCLS
```
classes:
0. outside_roi
1. tumor
2. stroma
3. lymphocytic_infiltrate
4. necrosis_or_debris
5. glandular_secretions
6. blood
7. exclude
8. metaplasia_NOS
9. fat
10. plasma_cells
11. other_immune_infiltrate
12. mucoid_material
13. normal_acinus_or_duct
14. lymphatics
15. undetermined
16. nerve
17. skin_adnexa
18. blood_vessel
19. angioinvasion
20. dcis
21. other

description: 
1. The NuCLS datasets contain over 220,000 labeled nuclei from breast cancer images from TCGA. 