# experiment settings
select_specific_parcels = False  # set false to use all parcels
train_on_all_frames = True  # when true, the following will be enforced:
# - ignore mapping.xlsx in training.
# - only do kmeans for 1 round
# - testing is not supported
# nrg_shape=(18688,18720) #(H,W)(12260, 11460)
nrg_shape=(12260, 11460)
train_on_all_rice_parcel =True

# Deep learning params
shape = (320, 320)  # (H,W), usually the bigger the better
# shape = (80, 80)
# we could set this to INF, if we want to sample as much training data as we can
# total_parcel_amount = 4000
# total_parcel_amount= 3000
# total_parcel_amount= 2000
total_parcel_amount= 10000

rice_cluster_n = 2
non_rice_cluster_n = 2
# used in sampling, this determines the percentage of rice parcels in the training data
rice_ratio = 0.5
round_number = 1 # the round number in mapping.xlsx
batch_size = 20
optimizer_learning_rate = 0.0001
EPOCH_N = 50  # 30
BANDS = 4
kmeans_batch_size = 2000
# training input path
training_NRG_png_path="./data/train_test/NRG_png"
training_parcel_mask_path="./data/train_test/parcel_mask"
training_selected_parcel_mask_path='./data/train_test/selected_parcel_mask'

# inference input path
inference_NRG_png_path="./data/inference/NRG_png"
inference_parcel_mask_path="./data/inference/parcel_mask"
inference_selected_parcel_mask_path="./data/inference/selected_parcel_mask"

# storage path
Data_root_folder_path = "./data/train_test"
Inference_Data_root_folder_path = "./data/inference"
saved_model_folder = "./data/inference/saved_model_and_prediction"
log_path = './data/logs/'
train_test_log_save_path = './data/logs/train_test_log.txt'
inference_log_save_path = './data/logs/inference_log.txt'
# --------------------------# Don't touch anything below this line!
data_shape = (shape[0], shape[1], BANDS) # modify to (shape[0], shape[1],4) if u want to use 4 bands
dataset_root_folder_path = Data_root_folder_path + \
                           "/For_training_testing/{}x{}".format(shape[0], shape[1])
random_sampling_result_dirname = Data_root_folder_path + \
                                 "/For_training_testing/Kmeans_results/Random_sampling_results/" \
                                 "{}x{}_T{}R{}NR{}_Rratio{}".format(
                                     data_shape[0], data_shape[1], total_parcel_amount, rice_cluster_n,
                                     non_rice_cluster_n, rice_ratio)
saving_model_dir_path_front_part = Data_root_folder_path + \
                                   "/For_training_testing/{}x{}".format(data_shape[0], data_shape[1])
inference_saving_model_dir_path_front_part = Inference_Data_root_folder_path + \
                                             "/For_training_testing/{}x{}".format(data_shape[0], data_shape[1])
save_prediction = True
# this determines the ratio of (training data :val data)
training_data_amount_ratio = 0.8
CFG = {
    "data_generation": {
        "training_NRG_png_path":training_NRG_png_path,
        "training_parcel_mask_path":training_parcel_mask_path,
        "training_selected_parcel_mask_path":training_selected_parcel_mask_path,
        "log_path": log_path,
        "train_test_log_save_path": train_test_log_save_path,
        "select_specific_parcels": select_specific_parcels,
        "shape": shape,
        "Data_root_folder_path": Data_root_folder_path,
    },
    "kmeans": {
        "training_NRG_png_path":training_NRG_png_path,
        "train_on_all_frames": train_on_all_frames,
        "Data_root_folder_path": Data_root_folder_path,
        "shape": shape,
        "dataset_root_folder_path": dataset_root_folder_path,
        "rice_cluster_n": rice_cluster_n,
        "non_rice_cluster_n": non_rice_cluster_n,
        "excel_path": Data_root_folder_path + "/mapping.xlsx",
        "Kmeans_result_root_dirname": Data_root_folder_path + "/For_training_testing/Kmeans_results",
        "parcel_partial_area_shape": (64, 64), 
        "batch_size": kmeans_batch_size,
    },
    "random_sampling": {
        "train_on_all_frames": train_on_all_frames,
        "training_data_amount_ratio": training_data_amount_ratio,
        "rice_ratio": rice_ratio,
        "Data_root_folder_path": Data_root_folder_path,
        "Kmeans_result_root_dirname": Data_root_folder_path + "/For_training_testing/Kmeans_results",
        "random_sampling_result_dirname": random_sampling_result_dirname,
        "data_shape": data_shape,
        "total_parcel_amount": total_parcel_amount,
        "train_all_rice" : train_on_all_rice_parcel
    },
    "train_and_val": {
        "EPOCH_N": EPOCH_N,
        "optimizer_learning_rate": optimizer_learning_rate,
        "batch_size": batch_size,
        "Data_root_folder_path": Data_root_folder_path,
        "random_sampling_result_dirname": random_sampling_result_dirname,
        "saving_model_dir_path_front_part": saving_model_dir_path_front_part,
        "data_shape": data_shape,
        "round_number": round_number
    },
    "testing": {
        "EPOCH_N": EPOCH_N,
        "optimizer_learning_rate": optimizer_learning_rate,
        "batch_size": batch_size,
        "Data_root_folder_path": Data_root_folder_path,
        "random_sampling_result_dirname": random_sampling_result_dirname,
        "saving_model_dir_path_front_part": saving_model_dir_path_front_part,
        "data_shape": data_shape,
        "round_number": round_number,
        "save_prediction": save_prediction,
        "select_specific_parcels": select_specific_parcels
    },
    "inference": {
        "inference_NRG_png_path":inference_NRG_png_path,
        "inference_parcel_mask_path":inference_parcel_mask_path,
        "inference_selected_parcel_mask_path" :inference_selected_parcel_mask_path,
        "log_path": log_path,
        "inference_log_save_path": inference_log_save_path,
        "Inference_Data_root_folder_path": Inference_Data_root_folder_path,
        "inference_saving_model_dir_path_front_part": inference_saving_model_dir_path_front_part,
        "saved_model_folder": saved_model_folder,
    }

}
