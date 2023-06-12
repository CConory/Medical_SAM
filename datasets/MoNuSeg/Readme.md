# Step1: Download the file 
## Notice to change the fileid and name for different set and run repeatedly
```
sh download.sh
```

# Step2 : unzip the download files

# Step3: split the train, val, test dataset 
```
python ./data_split.py
```

# Step4: Change annotaion style from XML 2 mask
```
python convert.py
# all .tif images will be moved into ./images/
# all .xml would generate to a numpy matrix e.g. XXXX.npy; shape (w,h,2)
# [...,0] is the instance mask
# [...,1] is the semantic mask
```