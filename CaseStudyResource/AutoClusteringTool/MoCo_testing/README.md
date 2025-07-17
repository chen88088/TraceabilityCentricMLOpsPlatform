# MoCo_testing User Guide 
  
## Requirements
- python >= 3.6 (recommend 3.7)
- tensorflow >= 2.2 (recommend 2.5)
- scikit-learn
- matplotlib

Run the following instruction to install related packages.
```
pip install -r requirement.txt
```

## Input data
1.Place the preprocessing result folder(.npy parcel data and gt .txt) under
```
./data/new_data
```
(if you run NCU-RSS-1.5-preprocessing copy from result)

You can also change the data folder path in config.py as needed.

2.Set the saved_model_path to the model path that you want to use in config.py.

3.Set the clusters_amount in config.py.
## Testing
Run the following instruction
```
python .\parcel_moco_kmeans.py
```
## Result
png files of rice parcel in each group will save at
```
 ./data/new_data/cluster number
```
