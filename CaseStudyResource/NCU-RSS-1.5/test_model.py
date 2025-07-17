"""Module testing the model using generated testing dataset"""

from configs.config import CFG
from src.models import driver
from src.utils.config import Config
from src.models import parcel_based_CNN

config = Config.from_json(CFG)
Data_root_folder_path = config.testing.Data_root_folder_path

select_specific_parcels = config.testing.select_specific_parcels
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
saving_model_dir_path_front_part = config.testing.saving_model_dir_path_front_part
round_number = config.testing.round_number  # affect the loaded training dataset and testing dataset
save_prediction = config.testing.save_prediction
# -----

random_sampling_result_txt_target_string = config.testing.random_sampling_result_dirname

if data_shape[2] == 3:
    masking = True
    data_shape_str = "Masked "
elif data_shape[2] == 4:
    masking = True
    data_shape_str = "Masked "
# elif data_shape[2] == 4:
#     masking = False
#     data_shape_str = "NonMasked "
data_shape_str = data_shape_str + f"{data_shape[0]}x{data_shape[1]}"

Dataset_preparation = parcel_based_CNN.Dataset_preparation(
    kmeans_result_dirname=None,
    random_sampling_result_dirname=random_sampling_result_dirname
)

# load the random sampling result of training & validation dataset for one round into memory from txt
training_ds_frame_dataset, validation_ds_frame_dataset = Dataset_preparation.load_one_round_random_sampling_result(
    random_sampling_result_txt_target_string=random_sampling_result_txt_target_string,
    Round_number=round_number
)

training_parcel_mask_path=config.data_generation.training_parcel_mask_path

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
    save_prediction=save_prediction
)
driver.main(
    EPOCH_N=EPOCH_N,
    arguments=arguments,
    data_shape=data_shape,
    round_number=round_number,
    saving_model_dir_path_front_part=saving_model_dir_path_front_part,
    optimizer_learning_rate=optimizer_learning_rate,
    lr_d=False,
    training_ds_frame_dataset=training_ds_frame_dataset,
    validation_ds_frame_dataset=validation_ds_frame_dataset,
    Data_root_folder_path=Data_root_folder_path,
    select_specific_parcels=select_specific_parcels,
    training_parcel_mask_path=training_parcel_mask_path
)
