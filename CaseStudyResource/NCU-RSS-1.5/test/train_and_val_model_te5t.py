import glob
import tensorflow as tf
import os
import src.models.parcel_based_CNN as parcel_based_CNN
from src.models import driver
class Training(tf.test.TestCase):
    def setUp(self):
        self.Data_root_folder_path = "./test/data/train_test"
        self.Kmeans_result_root_dirname=self.Data_root_folder_path+"/For_training_testing/Kmeans_results"
        self.shape= (80, 80)
        self.data_shape= (80, 80,3)
    def train(self,rice_cluster_n,non_rice_cluster_n,rice_ratio):
        
        Data_root_folder_path =self.Data_root_folder_path 

        data_shape = self.data_shape

        EPOCH_N=2

       
        optimizer_learning_rate=0.01
        batch_size=1

        random_sampling_result_dirname = self.Data_root_folder_path+"/For_training_testing/Kmeans_results/Random_sampling_results/{}x{}_T{}R{}NR{}_Rratio{}".format(
    data_shape[0], data_shape[1], 800, rice_cluster_n, non_rice_cluster_n, rice_ratio)
        # 存放model的資料夾路徑檔名前段(可以任意命名)
        saving_model_dir_path_front_part =  Data_root_folder_path + \
    "/For_training_testing/{}x{}".format(data_shape[0], data_shape[1])
        # affect the loaded training dataset and testing dataset
        round_number = 1
        # -----

        random_sampling_result_txt_target_string = random_sampling_result_dirname
        if data_shape[2] == 3:
            masking = True
            data_shape_str = "Masked "
        elif data_shape[2] == 4:
            masking = False
            data_shape_str = "NonMasked "
        data_shape_str = data_shape_str + \
            "{}x{}".format(str(data_shape[0]), str(data_shape[1]))

        Dataset_preparation = parcel_based_CNN.Dataset_preparation(
            kmeans_result_dirname=None,
            random_sampling_result_dirname=random_sampling_result_dirname
        )

        # load the random sampling result of training & validation dataset for one round into memory from txt
        training_ds_frame_dataset, validation_ds_frame_dataset = Dataset_preparation.load_one_round_random_sampling_result(
            random_sampling_result_txt_target_string=random_sampling_result_txt_target_string,
            Round_number=round_number
        )

        print(training_ds_frame_dataset["parcel_NIRRGA_path_list"][0])

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
            test_only=False,
            save_prediction=False
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
            select_specific_parcels= False,
            inference_NRG_png_path = "./test/data/inference/NRG_png",
            inference_parcel_mask_path = "./test/data/inference/parcel_mask",
            training_parcel_mask_path= './test/data/train_test/parcel_mask'
        )
        _trained_h5=glob.glob(saving_model_dir_path_front_part+'/train_test'+'/*.h5')
        self.assertLess(0,len(_trained_h5))
        _training_record_png=glob.glob(saving_model_dir_path_front_part+'/train_test'+'/*.png')
        self.assertLess(0,len(_training_record_png))