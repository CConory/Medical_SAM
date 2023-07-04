# Step1: Download the file 
```
# notice to change the fileid and name for different set
sh download.sh
```

# Step2 : unzip the download files

# Step3: Get the split json file from the csv file
```
python ./csv2json.py
```

# Step4: Change annotaion style 
```
python convert.py
# all .tif images will be moved into ./images/
# all .xml would generate to a numpy matrix e.g. XXXX.npy; shape (w,h,2)
# [...,0] is the instance mask, cell+nuclear is one instance
# [...,1] is the semantic mask Core is 1 ; cell is 2
```