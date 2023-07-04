# Step1: Download the file 
```
wget from kaggle # https://www.kaggle.com/datasets/aadimator/lizard-dataset
```

# Step2 : unzip the download files

# Step3: split the train, val, test dataset 


# Step4: Change annotaion style from XML 2 mask
```
python convert.py
# all .png images will be moved into ./images/
# all .mat would generate to a numpy matrix e.g. XXXX.npy; shape (w,h,2)
# [...,0] is the instance mask
# [...,1] is the semantic mask
```
classes:
1. Neutrophil
2. Epithelial
3. Lymphocyte
4. Plasma
5. Neutrophil
6. Connective tissue