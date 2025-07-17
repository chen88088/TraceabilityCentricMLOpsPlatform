"""Module training and validating the model using generated dataset"""

import os
import logging
from logging import config as log_config
import tensorflow as tf
import atexit
from src.models import parcel_based_CNN
from src.models import driver
from src.utils.config import Config
from configs.config import CFG
from configs.logger_config import LOGGING_CONFIG
import argparse
import pandas as pd  # 用於輸出 Excel
from pathlib import Path

# 輸出 Excel 的檔案路徑
excel_file_path = "training_results.xlsx"

# Initialize configuration
config = Config.from_json(CFG)
if not os.path.exists(config.data_generation.log_path):
    os.mkdir(config.data_generation.log_path)

log_config.dictConfig(LOGGING_CONFIG)
print(LOGGING_CONFIG)
info_logger = logging.getLogger("train_info_log")
debug_logger = logging.getLogger("train_debug_log")

Data_root_folder_path = config.train_and_val.Data_root_folder_path

data_shape = config.train_and_val.data_shape

EPOCH_N = config.train_and_val.EPOCH_N

debug_logger.debug("EPOCH_N: %d", EPOCH_N)
# optimizer_learning_rate
optimizer_learning_rate = config.train_and_val.optimizer_learning_rate
debug_logger.debug("optimizer_learning_rate: %f", optimizer_learning_rate)

# batch_size
batch_size = config.train_and_val.batch_size
debug_logger.debug("batch_size: %d", batch_size)

random_sampling_result_dirname = config.train_and_val.random_sampling_result_dirname
print(random_sampling_result_dirname)
# 存放model的資料夾路徑檔名前段(可以任意命名)
saving_model_dir_path_front_part = config.train_and_val.saving_model_dir_path_front_part
# affect the loaded training dataset and testing dataset
round_number = config.train_and_val.round_number
debug_logger.debug("round_number: %d", round_number)
# -----
training_parcel_mask_path=config.data_generation.training_parcel_mask_path
random_sampling_result_txt_target_string = config.train_and_val.random_sampling_result_dirname

if data_shape[2] == 3:
    MASKING = False
    DATA_SHAPE_STR = "NonMasked "
#4 band masked
# elif data_shape[2] == 4:
#     MASKING = True
#     DATA_SHAPE_STR = "Masked "
# 4 bands non masked
elif data_shape[2] == 4:
    MASKING = False
    DATA_SHAPE_STR = "NonMasked "
# elif data_shape[2] == 4:
#     MASKING = True
#     DATA_SHAPE_STR = "Masked "
DATA_SHAPE_STR = DATA_SHAPE_STR + \
                 f"{data_shape[0]}x{data_shape[1]}"

Dataset_preparation = parcel_based_CNN.Dataset_preparation(
    kmeans_result_dirname=None,
    random_sampling_result_dirname=random_sampling_result_dirname
)

# load the random sampling result of training & validation dataset for one round into memory from txt
training_ds_frame_dataset, validation_ds_frame_dataset = Dataset_preparation.load_one_round_random_sampling_result(
    random_sampling_result_txt_target_string=random_sampling_result_txt_target_string,
    Round_number=round_number
)

info_logger.info(training_ds_frame_dataset["parcel_NIRRGA_path_list"][0])

arguments = parcel_based_CNN.Arguments(
    data_shape=data_shape,
    round_number=round_number,
    BATCH_SIZE=batch_size,
    masking=MASKING,
    split_up_fourth_channel=False,
    normalizing=False,
    per_image_standardization=False,
    preprocess_input=False,
    random_brightness_max_delta=0.3,
    specified_size_for_testing=None,
    test_only=False,
    save_prediction=False
)

physical_devices = tf.config.experimental.list_physical_devices('GPU')

for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
    logging.info("device : %s"%physical_devices[i].name)
if len(physical_devices) > 1:  # for multiple gpu on twcc
    logging.info("using %d gpus: "%len(physical_devices))
    strategy = tf.distribute.MirroredStrategy()  # this allows tensor flow to utilize all available gpus
    with strategy.scope():
        model, history = driver.main(
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
            inference_NRG_png_path=None,
            training_parcel_mask_path=training_parcel_mask_path
        )
    tf.keras.backend.clear_session()
    atexit.register(strategy._extended._collective_ops._pool.close) # type: ignore
else:
    model, history = driver.main(
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
        inference_NRG_png_path=None,
        training_parcel_mask_path=training_parcel_mask_path
    )
    tf.keras.backend.clear_session()

# 將參數與結果儲存到 Excel
params = {
    "EPOCH_N": EPOCH_N,
    "optimizer_learning_rate": optimizer_learning_rate,
    "batch_size": batch_size,
    "round_number": round_number,
    "MASKING": MASKING,
    "tensorflow_version": tf.__version__,
}

# 訓練指標
if history is not None:
    metrics = {
        "train_accuracy": history.history.get('acc', []),
        "val_accuracy": history.history.get('val_acc', []),
        "train_loss": history.history.get('loss', []),
        "val_loss": history.history.get('val_loss', []),
    }
    if 'kappa' in history.history:
        metrics["train_kappa"] = history.history.get('kappa', [])
        metrics["val_kappa"] = history.history.get('val_kappa', [])
    metrics_df = pd.DataFrame(metrics)
else:
    metrics_df = pd.DataFrame()


# 設定 Excel 檔案要存放的相對路徑
# 確保程式在 Docker 內正確找到腳本的目錄
script_dir = Path(__file__).parent.resolve()  # 取得腳本所在的絕對目錄
excel_storage_path = script_dir / "data/train_test/For_training_testing/320x320/train_test/training_results.xlsx"

# 確保目錄存在
excel_storage_path.parent.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(excel_storage_path) as writer:
    pd.DataFrame([params]).to_excel(writer, sheet_name="Parameters", index=False)
    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

print(f"訓練結果與參數已保存到 {excel_storage_path}")
