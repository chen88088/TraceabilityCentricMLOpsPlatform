
import glob
import tensorflow as tf
from src.models import parcel_based_CNN
class Sampling(tf.test.TestCase):
    def setUp(self):
        self.Data_root_folder_path = "./test/data/train_test"
        self.Kmeans_result_root_dirname=self.Data_root_folder_path+"/For_training_testing/Kmeans_results"
        self.shape= (80, 80)
        
    def sample(self,rice_cluster_n,non_rice_cluster_n,train_on_all_frames,rice_ratio):
        training_data_amount_ratio=0.8
        total_parcel_amount=800
        non_rice_ratio = 1 - rice_ratio
        self.random_sampling_result_dirname=self.Data_root_folder_path+"/For_training_testing/Kmeans_results/Random_sampling_results/{}x{}_T{}R{}NR{}_Rratio{}".format(
    self.shape [0], self.shape [1], 800, rice_cluster_n, non_rice_cluster_n, rice_ratio)
        data_shape=self.shape
        data_shape_str = "{}x{}".format(data_shape[0], data_shape[1])

        # 計算理想狀況(每個cluster的資料量足夠)下，每個cluster會抽樣幾個坵塊資料
        parcel_data_amount_per_rice_cluster = int(
            total_parcel_amount*rice_ratio/rice_cluster_n)
        parcel_data_amount_per_non_rice_cluster = int(
            total_parcel_amount*non_rice_ratio/non_rice_cluster_n)
        print("parcel_data_amount_per_rice_cluster:",
              parcel_data_amount_per_rice_cluster)
        print("parcel_data_amount_per_non_rice_cluster:",
              parcel_data_amount_per_non_rice_cluster)

        kmeans_clustering_result_txt_target_string = r"R{}NR{}_{}x{}".format(
            rice_cluster_n, non_rice_cluster_n, data_shape[0], data_shape[1])


        
        Dataset_preparation0416_do_random_sampling = parcel_based_CNN.Dataset_preparation(
            kmeans_result_dirname=self.Kmeans_result_root_dirname,
            random_sampling_result_dirname=self.random_sampling_result_dirname,
            train_on_all_frames=train_on_all_frames#this sets self.train_on_all_frames in this class
        )

        # do the random sampling for Round1~Round5
        for round_number in range(1, 5+1):
            training_ds_frame_dataset, validation_ds_frame_dataset = Dataset_preparation0416_do_random_sampling.do_one_round_random_sampling_processing(
                kmeans_clustering_result_txt_target_string=kmeans_clustering_result_txt_target_string,
                Round_number=round_number,
                output_root_folder_path=self.random_sampling_result_dirname,
                parcel_data_amount_per_rice_cluster=parcel_data_amount_per_rice_cluster,
                parcel_data_amount_per_non_rice_cluster=parcel_data_amount_per_non_rice_cluster,
                training_data_amount_ratio=training_data_amount_ratio

            )
            if train_on_all_frames:break
        _sampling_result_txt=glob.glob(self.random_sampling_result_dirname+'/*.txt')
        self.assertLess(0,len(_sampling_result_txt))