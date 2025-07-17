
import tensorflow as tf
from src. preprocessing.kmeans_clustering_for_parcel_dataset import KMeans_clustering_and_save_result_txt
from src.models import parcel_based_CNN
from src.preprocessing.parcel_preprocessing import DataManager
import os
import glob
class Clustering(tf.test.TestCase):
    def setUp(self):
        self.Data_root_folder_path = "./test/data/train_test"
        self.shape = (80, 80)
        self.excel_path = "./test/data/train_test/mapping.xlsx"
        self.data_manager = DataManager()

    def cluster(  # change these
        self, train_on_all_frames=True,
            rice_cluster_n=2,
            non_rice_cluster_n=2):

        # and don't change stuff below
        Data_root_folder_path = self.Data_root_folder_path
        shape = self.shape
        dataset_root_folder_path = Data_root_folder_path + "/For_training_testing/{}x{}".format(shape[0], shape[1])
        excel_path = self.excel_path
        Kmeans_result_root_dirname = Data_root_folder_path+"/For_training_testing/Kmeans_results"
        parcel_based_CNN.Create_Folder([Kmeans_result_root_dirname])

        frame_dataset_list = self.data_manager.load_saved_dataset(
            dataset_root_path=dataset_root_folder_path)
        parcel_partial_area_shape = (64, 64)
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
            training_NRG_png_path = "./test/data/train_test/NRG_png"
        )
        _kmeans_result_txt=glob.glob(Kmeans_result_root_dirname+"/*.txt")
        self.assertLess(0,len(_kmeans_result_txt))
