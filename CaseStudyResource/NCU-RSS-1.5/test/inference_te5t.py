import glob
import tensorflow as tf
import os
import src.models. parcel_based_CNN as parcel_based_CNN
from src.models import driver
class Inference(tf.test.TestCase):
    def setUp(self):
        self.Data_root_folder_path = "./test/data/inference"
        self.Kmeans_result_root_dirname=self.Data_root_folder_path+"/For_training_testing/Kmeans_results"
        self.shape= (80, 80)
        self.data_shape= (80, 80,3)
    def infer(self):
        
        Data_root_folder_path =self.Data_root_folder_path

        select_specific_parcels=False
        #----- 
        # the shape of parcel image
        data_shape = self.data_shape
        EPOCH_N=1
        #optimizer_learning_rate
        optimizer_learning_rate=0.001
        #batch_size
        batch_size=10
        # the root dirname saving random_sampling_result txt file(須設定成存放著要載入的坵塊資料txt的資料夾路徑)

        random_sampling_result_dirname = self.Data_root_folder_path+"/For_training_testing/Kmeans_results/Random_sampling_results/{}x{}_T{}R{}NR{}_Rratio{}".format(
    data_shape[0], data_shape[1], 800, 2, 2, .5)
        # 存放model的資料夾路徑檔名前段(可以任意命名)
        saving_model_dir_path_front_part =  Data_root_folder_path + \
    "/For_training_testing/{}x{}".format(data_shape[0], data_shape[1])
        round_number = 1
        save_prediction=True
        saved_model_folder= "./test/data/inference/saved_model_and_prediction"

        
        #-----

        random_sampling_result_txt_target_string = Data_root_folder_path+"/For_training_testing/Kmeans_results/Random_sampling_results/{}x{}_T{}R{}NR{}_Rratio{}".format(
    data_shape[0], data_shape[1], 800, 2, 2, .5)
        if data_shape[2]==3:
            masking=True
            data_shape_str = "Masked "
        elif data_shape[2]==4:
            masking=False
            data_shape_str = "NonMasked " 
        data_shape_str = data_shape_str + "{}x{}".format(str(data_shape[0]),str(data_shape[1])) 

        Dataset_preparation = parcel_based_CNN.Dataset_preparation( 
            kmeans_result_dirname=None, 
            random_sampling_result_dirname=random_sampling_result_dirname
        )

        # load the random sampling result of training & validation dataset for one round into memory from txt
        #training_ds_frame_dataset, validation_ds_frame_dataset = Dataset_preparation.load_one_round_random_sampling_result(
        #    random_sampling_result_txt_target_string=random_sampling_result_txt_target_string,
        #    Round_number=round_number
        #)

        #print(training_ds_frame_dataset["parcel_NIRRGA_path_list"][0])

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
                save_prediction =save_prediction
            )
        driver.main(
            EPOCH_N=EPOCH_N,
            arguments=arguments,
            data_shape=data_shape,
            round_number=round_number,
            saving_model_dir_path_front_part=saving_model_dir_path_front_part,
            optimizer_learning_rate = optimizer_learning_rate  ,
            lr_d = False,
            training_ds_frame_dataset=None,
            validation_ds_frame_dataset=None,
            Data_root_folder_path=Data_root_folder_path,
            select_specific_parcels=select_specific_parcels,
            saved_model_folder=saved_model_folder,
            inference_NRG_png_path = "./test/data/inference/NRG_png",
            inference_parcel_mask_path = "./test/data/inference/parcel_mask",
            training_parcel_mask_path = './test/data/train_test/parcel_mask'
        )
        _segmentation_masks=glob.glob("test/data/inference/saved_model_and_prediction/model_val_acc/*.png")
        self.assertLess(0,len(_segmentation_masks))