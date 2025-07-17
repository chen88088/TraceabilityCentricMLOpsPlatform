"""Module random sampling from clustered parcels"""

import os
import logging
from configs.config import CFG
from src.models import parcel_based_CNN
from src.utils.config import Config

config = Config.from_json(CFG)
if not os.path.exists(config.data_generation.log_path):
    os.mkdir(config.data_generation.log_path)
logging.basicConfig(
    filename=config.data_generation.train_test_log_save_path, level=logging.INFO, force=True)
Data_root_folder_path = config.random_sampling.Data_root_folder_path
# the path of the folder saving the Kmeans result txt for each round
Kmeans_result_root_dirname = config.random_sampling.Kmeans_result_root_dirname
# the shape of the parcel image in the parcel dataset which has the Kmeans clustering result txt
data_shape = config.random_sampling.data_shape
# the number of the cluster for the Kmeans clustering result txt
rice_cluster_n = config.kmeans.rice_cluster_n
non_rice_cluster_n = config.kmeans.non_rice_cluster_n
# the amount of parcel data that will be sampled
total_parcel_amount = config.random_sampling.total_parcel_amount

logging.info("total_parcel_amount:%d", config.random_sampling.total_parcel_amount)
training_data_amount_ratio = config.random_sampling.training_data_amount_ratio

train_on_all_frames = config.random_sampling.train_on_all_frames
rice_ratio = config.random_sampling.rice_ratio
non_rice_ratio = 1 - rice_ratio

data_shape_str = f"{data_shape[0]}x{data_shape[1]}"

# 計算理想狀況(每個cluster的資料量足夠)下，每個cluster會抽樣幾個坵塊資料
parcel_data_amount_per_rice_cluster = int(
    total_parcel_amount * rice_ratio / rice_cluster_n)
parcel_data_amount_per_non_rice_cluster = int(
    total_parcel_amount * non_rice_ratio / non_rice_cluster_n)

logging.info("parcel_data_amount_per_rice_cluster:%d", parcel_data_amount_per_rice_cluster)
logging.info("parcel_data_amount_per_non_rice_cluster:%d", parcel_data_amount_per_non_rice_cluster)

kmeans_clustering_result_txt_target_string = \
    f"R{rice_cluster_n}NR{non_rice_cluster_n}_{data_shape[0]}x{data_shape[1]}"
random_sampling_result_dirname = config.random_sampling.random_sampling_result_dirname
Dataset_preparation0416_do_random_sampling = parcel_based_CNN.Dataset_preparation(
    kmeans_result_dirname=Kmeans_result_root_dirname,
    random_sampling_result_dirname=random_sampling_result_dirname,
    train_on_all_frames=train_on_all_frames  # this sets self.train_on_all_frames in this class
)

# do the random sampling for Round1~Round5
for round_number in range(1, 5 + 1):
    print(f"round {round_number}")
    training_ds_frame_dataset, validation_ds_frame_dataset = \
        Dataset_preparation0416_do_random_sampling.do_one_round_random_sampling_processing(
            kmeans_clustering_result_txt_target_string=kmeans_clustering_result_txt_target_string,
            Round_number=round_number,
            output_root_folder_path=random_sampling_result_dirname,
            parcel_data_amount_per_rice_cluster=parcel_data_amount_per_rice_cluster,
            parcel_data_amount_per_non_rice_cluster=parcel_data_amount_per_non_rice_cluster,
            training_data_amount_ratio=training_data_amount_ratio

        )
    if train_on_all_frames:
        break
