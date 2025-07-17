# MoCo_revised User Guide 
  
## Requirements
- python >= 3.6 (recommend 3.7)
- tensorflow >= 2.2 (recommend 2.5)
- pyyaml
- numpy
- pandas
- tensorflow_addons

Run the following instruction to install related packages.
```
pip install -r requirement.txt
```

## Input data
Place the preprocessing result folder(only .npy parcel data) under
  ```
  ./data
  ```
(if you run NCU-RSS-1.5-preprocessing copy from result)
## Training
For training moco, please run the following instruction
```
python main.py --task v1 --weight_decay 0.0001 --brightness 0.4 --contrast 0.4 --saturation 0.4 --hue 0.4 --lr_mode exponential --lr_interval 120,160 --data_path ./data --gpus 0 --history --summary --resume
```
You can also change any parameters in common.py as needed.
## Result
The model will save at
```
./result/v1/timestamp/model/moco_model_batchsize_epoch_lr
```