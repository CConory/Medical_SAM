# Step1: Follow Steps for each dataset.
Details can be seen in each dataset file. Follow the Readme instruction in each file. 

# Step2 (Optional) : Add prefix for each file
Considered that we have run each of the dataset to see the performance of SAM, we have already 
completed the data preprocessing for bowl_2018, lizard, MoNuSeg, PanNuke, SegPC-2021. 
In order to avoid cases where some files in different datasets have the same name, I added a 
prefix to the data in subsequent data processing. This prefix is the name of the dataset in 
which the data resides. You can add prefixes to previously processed datasets as needed. 
Processing ranges include data_split.json,./masks,./images, and./features.
```
python add_prefix.py --dataset_name bowl_2018
```

# Step3: See the data distribution of each dataset
```
python data_analysis.py --dataset_name bowl_2018
```
