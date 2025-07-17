"""Module generating parcel for inference"""
import os
import logging
from configs.config import CFG
from src.utils.config import Config
from src.preprocessing.generate_parcel_dataset_based_on_complete_frame_aerial_image import generate_parcel_dataset

config = Config.from_json(CFG)
if not os.path.exists(config.inference.log_path):
    os.mkdir(config.inference.log_path)
if os.path.exists(config.inference.inference_log_save_path):
    with open(config.inference.inference_log_save_path, 'w', encoding="utf-8"):
        pass  # clear log
logging.basicConfig(
    filename=config.inference.inference_log_save_path, level=logging.INFO, force=True)
Inference_Data_root_folder_path = config.inference.Inference_Data_root_folder_path
shape = config.data_generation.shape  # (H,W)
bands = config.random_sampling.data_shape[2]

logging.info(f"Selected bands = {str(bands)}")
logging.info("Inference_Data_root_folder_path:%s", config.inference.Inference_Data_root_folder_path)
logging.info("shape: (%dx%d) ", shape[0], shape[1])

inference_NRG_png_path=config.inference.inference_NRG_png_path
inference_parcel_mask_path=config.inference.inference_parcel_mask_path

frame_dataset_list = generate_parcel_dataset(shape=shape, dataset_root_folder_path=Inference_Data_root_folder_path,
                                             image_type="npy",
                                             select_specific_parcels=False, inference=True,
                                             NRG_png_path=inference_NRG_png_path,
                                             parcel_mask_path=inference_parcel_mask_path,
                                             selected_parcel_mask_path=None, 
                                             band=bands
                                             )
