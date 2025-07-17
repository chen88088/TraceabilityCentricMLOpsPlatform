"""Module generating parcel for train_test"""

import os
import logging
from configs.config import CFG
from src.utils.config import Config
from src.preprocessing.generate_parcel_dataset_based_on_complete_frame_aerial_image import generate_parcel_dataset

config = Config.from_json(CFG)
if not os.path.exists(config.data_generation.log_path):
    os.mkdir(config.data_generation.log_path)
if os.path.exists(config.data_generation.train_test_log_save_path):
    with open(config.data_generation.train_test_log_save_path, 'wb'):
        pass  # clear log

logging.basicConfig(
    filename=config.data_generation.train_test_log_save_path, level=logging.INFO, force=True)

Data_root_folder_path = config.data_generation.Data_root_folder_path

logging.info("Data_root_folder_path:%s", config.data_generation.Data_root_folder_path)
shape = config.data_generation.shape  # (H,W)
logging.info("shape:%dx%d", config.data_generation.shape[0], config.data_generation.shape[1])
bands = config.random_sampling.data_shape[2]
logging.info(f"Selected bands = {str(bands)}")
select_specific_parcels = config.data_generation.select_specific_parcels
training_NRG_png_path=config.data_generation.training_NRG_png_path
training_parcel_mask_path=config.data_generation.training_parcel_mask_path
training_selected_parcel_mask_path=config.data_generation.training_selected_parcel_mask_path
frame_dataset_list = generate_parcel_dataset(shape=shape, dataset_root_folder_path=Data_root_folder_path,
                                             image_type="npy",
                                             select_specific_parcels=select_specific_parcels,
                                             NRG_png_path=training_NRG_png_path,
                                             parcel_mask_path=training_parcel_mask_path,
                                             selected_parcel_mask_path=training_selected_parcel_mask_path, 
                                             band=bands   #modify the band
                                             )
