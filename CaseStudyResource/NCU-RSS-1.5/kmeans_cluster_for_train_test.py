"""Module clustering parcel"""

import os
import logging
from configs.config import CFG
from src.models import parcel_based_CNN
from src.preprocessing.parcel_preprocessing import DataManager
from src.preprocessing.kmeans_clustering_for_parcel_dataset import KMeans_clustering_and_save_result_txt
from src.utils.config import Config

data_manager = DataManager()
config = Config.from_json(CFG)
if not os.path.exists(config.data_generation.log_path):
    os.mkdir(config.data_generation.log_path)
logging.basicConfig(
    filename=config.data_generation.train_test_log_save_path, level=logging.INFO)
Data_root_folder_path = config.kmeans.Data_root_folder_path

train_on_all_frames = config.random_sampling.train_on_all_frames
# the size of parcel image
shape = config.kmeans.shape  # (H,W)
# the path of folder saving the parcel dataset based on complete frame aerial image
dataset_root_folder_path = config.kmeans.dataset_root_folder_path
# the number of cluster for kmeans
rice_cluster_n = config.kmeans.rice_cluster_n
logging.info("rice_cluster_n:%d", config.kmeans.rice_cluster_n)
non_rice_cluster_n = config.kmeans.non_rice_cluster_n
logging.info("non_rice_cluster_n:%d", config.kmeans.non_rice_cluster_n)
# the excel recorded the training frame and testing frame for each round
excel_path = config.kmeans.excel_path
# the path of the folder saving the Kmeans result txt for each round
Kmeans_result_root_dirname = config.kmeans.Kmeans_result_root_dirname
parcel_based_CNN.Create_Folder([Kmeans_result_root_dirname])

frame_dataset_list = data_manager.load_saved_dataset(dataset_root_path=dataset_root_folder_path)
parcel_partial_area_shape = config.kmeans.parcel_partial_area_shape  # (H,W)
training_NRG_png_path=config.kmeans.training_NRG_png_path

KMeans_clustering_and_save_result_txt(
    shape=shape,
    parcel_partial_area_shape=parcel_partial_area_shape,
    rice_cluster_n=rice_cluster_n,
    non_rice_cluster_n=non_rice_cluster_n,
    excel_path=excel_path,
    frame_dataset_list=frame_dataset_list,
    Kmeans_result_root_dirname=Kmeans_result_root_dirname,
    train_on_all_frames=train_on_all_frames,
    Data_root_folder_path=Data_root_folder_path,
    training_NRG_png_path=training_NRG_png_path
)
