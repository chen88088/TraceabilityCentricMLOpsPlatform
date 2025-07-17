""""Module prediction on generated inference dataset using trained model"""

import glob
import os
import logging
import time

import tensorflow as tf
import atexit
from configs.config import CFG
from src.models import driver
from src.utils.config import Config
from src.models import parcel_based_CNN

# this line disables the gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
_time_as_inference_starts = time.time()
config = Config.from_json(CFG)
if not os.path.exists(config.data_generation.log_path):
    os.mkdir(config.data_generation.log_path)

logging.basicConfig(
    filename=config.inference.inference_log_save_path, level=logging.INFO)
Data_root_folder_path = config.inference.Inference_Data_root_folder_path

select_specific_parcels = False
# -----
# the shape of parcel image
data_shape = config.testing.data_shape
EPOCH_N = config.testing.EPOCH_N
# optimizer_learning_rate
optimizer_learning_rate = config.testing.optimizer_learning_rate
# batch_size
batch_size = config.testing.batch_size
# the root dirname saving random_sampling_result txt file(須設定成存放著要載入的坵塊資料txt的資料夾路徑)

random_sampling_result_dirname = config.testing.random_sampling_result_dirname
# 存放model的資料夾路徑檔名前段(可以任意命名)
saving_model_dir_path_front_part = config.inference.inference_saving_model_dir_path_front_part
round_number = config.testing.round_number  # affect the loaded training dataset and testing dataset
save_prediction = True
saved_model_folder = config.inference.saved_model_folder

inference_NRG_png_path=config.inference.inference_NRG_png_path
inference_parcel_mask_path=config.inference.inference_parcel_mask_path
logging.info("saved_model_folder:%s", str(config.inference.saved_model_folder))
# -----

random_sampling_result_txt_target_string = config.testing.random_sampling_result_dirname

if data_shape[2] == 3:
    masking = False
    data_shape_str = "NonMasked "
## 4 bands masked
# elif data_shape[2] == 4: 
#     masking = True 
#     data_shape_str = "Masked "
## 4 bands nonmasked
elif data_shape[2] == 4: 
    masking = False 
    data_shape_str = "NonMasked "

data_shape_str = data_shape_str + f"{data_shape[0]}x{data_shape[1]}"

Dataset_preparation = parcel_based_CNN.Dataset_preparation(
    kmeans_result_dirname=None,
    random_sampling_result_dirname=random_sampling_result_dirname
)

# load the random sampling result of training & validation dataset for one round into memory from txt
# training_ds_frame_dataset, validation_ds_frame_dataset = Dataset_preparation.load_one_round_random_sampling_result(
#    random_sampling_result_txt_target_string=random_sampling_result_txt_target_string,
#    Round_number=round_number
# )

# print(training_ds_frame_dataset["parcel_NIRRGA_path_list"][0])

arguments = parcel_based_CNN.Arguments(
    data_shape=data_shape,
    round_number=round_number,
    BATCH_SIZE=batch_size,
    masking=masking,
    split_up_fourth_channel=False,
    normalizing=False,
    per_image_standardization=False,
    preprocess_input=False,
    random_brightness_max_delta=0.3,
    specified_size_for_testing=None,
    test_only=True,
    inference=True,
    save_prediction=save_prediction
)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
    logging.info("device : %s"%str(physical_devices[i]))
if len(physical_devices) > 1:  # for multiple gpu on twcc
    logging.info("using %d gpus: "%len(physical_devices))
    strategy = tf.distribute.MirroredStrategy()  # this allows tensor flow to utilize all available gpus
    with strategy.scope():
        driver.main(
        EPOCH_N=EPOCH_N,
        arguments=arguments,
        data_shape=data_shape,
        round_number=round_number,
        saving_model_dir_path_front_part=saving_model_dir_path_front_part,
        optimizer_learning_rate=optimizer_learning_rate,
        lr_d=False,
        training_ds_frame_dataset=None,
        validation_ds_frame_dataset=None,
        Data_root_folder_path=Data_root_folder_path,
        select_specific_parcels=select_specific_parcels,
        saved_model_folder=saved_model_folder,
        inference_NRG_png_path=inference_NRG_png_path,
        inference_parcel_mask_path=inference_parcel_mask_path,
        training_parcel_mask_path=None
        )
    tf.keras.backend.clear_session()
    atexit.register(strategy._extended._collective_ops._pool.close) # type: ignore
else:
    driver.main(
        EPOCH_N=EPOCH_N,
        arguments=arguments,
        data_shape=data_shape,
        round_number=round_number,
        saving_model_dir_path_front_part=saving_model_dir_path_front_part,
        optimizer_learning_rate=optimizer_learning_rate,
        lr_d=False,
        training_ds_frame_dataset=None,
        validation_ds_frame_dataset=None,
        Data_root_folder_path=Data_root_folder_path,
        select_specific_parcels=select_specific_parcels,
        saved_model_folder=saved_model_folder,
        inference_NRG_png_path=inference_NRG_png_path,
        inference_parcel_mask_path=inference_parcel_mask_path,
        training_parcel_mask_path=None
        )
    tf.keras.backend.clear_session()
# rename predictions from [frame_num].png to FOLD1_TEST_[frame_num].png
h5_path = glob.glob(saved_model_folder + '/*.h5')
h5_name = os.path.basename(h5_path[0])
h5_name = h5_name.split('.')[0]
all_png_paths = glob.glob(os.path.join(saved_model_folder, h5_name, '*.png'))
for p in all_png_paths:
    _base_name = os.path.basename(p)
    new_name = p.replace(_base_name, 'FOLD1_TEST_' + _base_name)
    os.rename(p, new_name)
_time_as_inference_ends = time.time()
logging.info("time cost for prediction computation in seconds%d",
                         _time_as_inference_ends - _time_as_inference_starts)
logging.info("time cost for prediction computation in minutes%f",
                         (_time_as_inference_ends - _time_as_inference_starts) / 60)