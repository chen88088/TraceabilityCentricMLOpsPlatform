# -*- coding: utf-8 -*-
# +

import shutil
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras.layers import GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras import layers, models
from src.preprocessing.parcel_preprocessing import DataManager
from sklearn.metrics import confusion_matrix
import sys
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten
from tensorflow.keras.models import Sequential
import math
from tensorflow import float32
from skimage.measure import label, regionprops
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import random
import random as rd
from tensorflow.keras.models import load_model
from skimage.transform import resize

import time

import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import tensorflow as tf
import os
from configs.logger_config import LOGGING_CONFIG
from logging import config
import logging


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
# from keras.models 改成 from tensorflow.keras.models !!


config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("train_info_log")
debug_logger = logging.getLogger("train_debug_log")
DATA_ROOT_FOLDER_PATH = "./data/train_test"
data_manager = DataManager()


def Create_Folder(created_folder_path_list):
    """
    created_folder_path_list:
        a list saving the path of folder that need to be created
    """

    for f in created_folder_path_list:
        try:
            os.makedirs(f)
            logger.info(f + " is not exist => create folder")
        except OSError:
            # shutil.rmtree(f) 刪除資料夾
            # os.makedirs(f)
            logger.info(f + " is already exist ")


# +
class Dataset_preparation:
    def __init__(self, kmeans_result_dirname=None,
                 random_sampling_result_dirname=None, train_on_all_frames=False):
        """
        txt 寫入和讀取

        load_one_round_kmeans_clustering_result(self, target_string, Round_number):
            載入一個回合的分群結果txt

        kmeans_result_dirname:
            kmeans clustering result txt存檔位置
        random_sampling_result_dirname:
            保存著要載入的隨機抽樣結果txt的目錄名稱
        """

        self.kmeans_result_dirname = kmeans_result_dirname
        self.kmeans_result_txt_list = None

        self.random_sampling_result_dirname = random_sampling_result_dirname
        self.random_sampling_result_txt_list = None
        self.train_on_all_frames = train_on_all_frames

    def __load_kmeans_result_txt_list(self, kmeans_result_txt_target_string):
        """
        載入Kmeans clustering的分群結果txt路徑list

        target_string:

        """

        logger.info("__load_kmeans_result_txt_list...")

        self.kmeans_result_txt_list = self.get_target_kmeans_result_txt_list(
            folder_path=self.kmeans_result_dirname,
            kmeans_result_txt_target_string=kmeans_result_txt_target_string
        )

    def __load_random_sampling_result_txt_list(
            self,
            random_sampling_result_txt_target_string):
        """
        載入random_sampling的抽樣結果txt路徑list
        # 可能載入到一個或多個txt

        target_string: ????

        """
        logger.info("\t__load_random_sampling_result_txt_list() ")

        self.random_sampling_result_txt_list = self.get_target_random_sampling_result_txt_list(
            folder_path=self.random_sampling_result_dirname,
            random_sampling_result_txt_target_string=random_sampling_result_txt_target_string
        )
        if self.random_sampling_result_txt_list is None:
            raise Exception("no sampling result txt found!")
        logger.info(f"\tself.random_sampling_result_txt_list: %s\n", " ".join(
            self.random_sampling_result_txt_list))

    def __get_corresponding_kmeans_result_txt_path(self, Round_number):
        logger.info("self.kmeans_result_txt_list: %s", " ".join(self.kmeans_result_txt_list))
        for txt_path in self.kmeans_result_txt_list:
            txt_round_number = int(os.path.basename(
                txt_path).split("Round")[1][0])
            if txt_round_number == Round_number:
                return txt_path

        logger.info(
            "*** No corresponding kmeans_result_txt_path of Round_number:{}".format(Round_number))
        sys.exit(1)

    def __get_corresponding_random_sampling_result_txt_path(
            self, Round_number):
        for txt_path in self.random_sampling_result_txt_list:
            txt_round_number = int(os.path.basename(
                txt_path).split("Round")[1][0])
            if txt_round_number == Round_number:
                return txt_path

        logger.info("*** No corresponding random_sampling_result_txt_path of Round_number:{}".format(Round_number))
        sys.exit(1)

    def load_one_round_kmeans_clustering_result(
            self,
            kmeans_result_txt_target_string,
            kmeans_result_txt_path=None,
            Round_number=None):
        """
        用法1:
            指定kmeans_result_txt_path
        用法2:
            指定Round_number

        載入一個回合的分群結果txt
        回傳對應的loaded_clusters_storage、kmeans_result_txt_path

        loaded_clusters_storage:
            已經過Kmeans分群的資料集
        """
        if kmeans_result_txt_path is not None:
            loaded_clusters_storage = self.load_txt_storage(
                kmeans_result_txt_path)
            return loaded_clusters_storage

        if Round_number is not None:
            self.__load_kmeans_result_txt_list(
                kmeans_result_txt_target_string)  # 可能載入到一個或多個txt
            kmeans_result_txt_path = self.__get_corresponding_kmeans_result_txt_path(
                Round_number)
            logger.info("Load  kmeans_result_txt_path: %s", kmeans_result_txt_path)
            loaded_clusters_storage = self.load_txt_storage(
                kmeans_result_txt_path)

        return loaded_clusters_storage, kmeans_result_txt_path

    def load_one_round_random_sampling_result(
            self, random_sampling_result_txt_target_string,
            random_sampling_result_txt_path=None, Round_number=None):
        """
        用法1:
            指定random_sampling_result_txt_path
        用法2:
            指定Round_number

        載入對應的loaded_sampling_storage並拆成training_ds_frame_dataset、validation_ds_frame_dataset回傳(用於training)

        loaded_sampling_storage:
            已經過隨機取樣的資料集

        executing_location:
            do this function at where
            => "local" or "TWCC"

        """
        
        if random_sampling_result_txt_path is not None:
            loaded_sampling_storage = self.load_txt_storage(
                random_sampling_result_txt_path)
        elif Round_number is not None:
            self.__load_random_sampling_result_txt_list(
                random_sampling_result_txt_target_string)
            random_sampling_result_txt_path = self.__get_corresponding_random_sampling_result_txt_path(
                Round_number)
            logger.info("Load random_sampling_result_txt_path: %s",
                        random_sampling_result_txt_path)
            loaded_sampling_storage = self.load_txt_storage(
                random_sampling_result_txt_path)
            
        random_training_ds_list = loaded_sampling_storage["random_training_ds_list"]
        random_validation_ds_list = loaded_sampling_storage["random_validation_ds_list"]
        n_random_training_ds_list = len(random_training_ds_list)
        n_random_validation_ds_list = len(random_validation_ds_list)
        logger.info(f"  len(random_training_ds_list): {n_random_training_ds_list}")
        logger.info(f"  len(random_validation_ds_list): {n_random_validation_ds_list}")

        # 轉換成 以dictionary存放list 的形式
        training_ds_frame_dataset = self.converting_parcel_data_list_to_list_of_dictionary(
            random_training_ds_list)
        validation_ds_frame_dataset = self.converting_parcel_data_list_to_list_of_dictionary(
            random_validation_ds_list)

        return training_ds_frame_dataset, validation_ds_frame_dataset

    def do_one_round_random_sampling_processing(
            self, kmeans_clustering_result_txt_target_string, Round_number,
            output_root_folder_path,
            parcel_data_amount_per_rice_cluster, parcel_data_amount_per_non_rice_cluster,
            training_data_amount_ratio
    ):
        """
        回傳經過抽樣的 training_ds_frame_dataset, validation_ds_frame_dataset

        parcel_data_amount_per_cluster:
            從每個cluster隨機挑選的parcel data數量
        output_root_folder_path:
            存放多個round的隨機挑選結果的目錄路徑

        """

        if os.path.isdir(output_root_folder_path) == False:
            Create_Folder([output_root_folder_path])

        one_round_kmeans_clustering_result, kmeans_result_txt_path = self.load_one_round_kmeans_clustering_result(
            kmeans_clustering_result_txt_target_string, Round_number=Round_number)

        # 從各個cluster中進行隨機挑選
        random_training_ds_list = []
        random_validation_ds_list = []

        # 進行隨機挑選 寫檔紀錄隨機挑選的結果
        rice_remaining_amount = 0
        non_rice_remaining_amount = 0

        # Random sampling
        cluster_count = 0
        cluster_name_list = []

        class cluster_name_len:
            def __init__(self, cluster_name, data_amount) -> None:
                self.cluster_name = cluster_name
                self.data_amount = data_amount

        cluster_name_len_list = []
        for cluster_name in one_round_kmeans_clustering_result:
            cluster_name_len_list.append(cluster_name_len(
                cluster_name, len(one_round_kmeans_clustering_result[cluster_name])))

        # 根據數量由小到大排序
        # 假設在取水稻時，若數量較少的cluster全取還有剩餘要取的量，會在數量較多的cluster繼續取
        cluster_name_len_list.sort(
            key=lambda cluster_name_len_obj: cluster_name_len_obj.data_amount)
        for ele in cluster_name_len_list:
            logger.info("cluster name:{}   cluster len:{}".format(
                ele.cluster_name, ele.data_amount))
            cluster_name_list.append(ele.cluster_name)

        training_data_amount_ratio = round(
            training_data_amount_ratio, 4)  # 0.9

        logger.info(" training_data_amount_ratio:{}".format(
            training_data_amount_ratio))

        tr_rice_amount = 0
        tr_non_rice_amount = 0

        # sample
        # one_round_kmeans_clustering_result[cluster_name],
        # parcel_data_amount_per_rice_cluster
        # parcel_data_amount_per_non_rice_cluster,
        # training_data_amount_ratio
        for cluster_name in cluster_name_list:

            if "NonRice_cluster" in cluster_name:  # 非水稻
                if parcel_data_amount_per_non_rice_cluster > len(
                        one_round_kmeans_clustering_result[cluster_name]):
                    parcel_data_amount_per_non_rice_cluster = len(
                        one_round_kmeans_clustering_result[cluster_name])
                _sample_size = parcel_data_amount_per_non_rice_cluster

            else:  # 水稻
                if parcel_data_amount_per_rice_cluster > len(
                        one_round_kmeans_clustering_result[cluster_name]):
                    parcel_data_amount_per_rice_cluster = len(
                        one_round_kmeans_clustering_result[cluster_name])
                _sample_size = parcel_data_amount_per_rice_cluster

            list_of_this_cluster: list = one_round_kmeans_clustering_result[cluster_name]

            if self.train_on_all_frames:
                train_list_of_this_cluster = random.sample(
                    list_of_this_cluster, _sample_size)
                val_list_of_this_cluster = random.sample(list_of_this_cluster, int(
                    _sample_size * round((1 - training_data_amount_ratio), 4)))
                random_training_ds_list.extend(train_list_of_this_cluster)
                random_validation_ds_list.extend(val_list_of_this_cluster)
            else:
                train_list_of_this_cluster = random.sample(
                    list_of_this_cluster, int(_sample_size * training_data_amount_ratio))
                val_list_of_this_cluster = [
                    elem for elem in list_of_this_cluster if elem not in train_list_of_this_cluster]
                val_list_of_this_cluster = random.sample(val_list_of_this_cluster, int(
                    _sample_size * round((1 - training_data_amount_ratio), 4)))
                random_training_ds_list.extend(train_list_of_this_cluster)
                random_validation_ds_list.extend(val_list_of_this_cluster)
        # logger.info("------抽樣結束------")

        # 對訓練資料集打亂順序
        random.shuffle(random_training_ds_list)
        random.shuffle(random_validation_ds_list)

        training_ds_amount = len(random_training_ds_list)
        validation_ds_amount = len(random_validation_ds_list)
        logger.info(f"training_ds_amount:{training_ds_amount}")
        logger.info(f"validation_ds_amount: {validation_ds_amount}")

        # 轉換成 以dictionary存放list 的形式
        training_ds_frame_dataset = self.converting_parcel_data_list_to_list_of_dictionary(
            random_training_ds_list)
        validation_ds_frame_dataset = self.converting_parcel_data_list_to_list_of_dictionary(
            random_validation_ds_list)

        # 寫檔 ------------------------
        # 設定此round的抽樣結果txt路徑

        #  ===========================

        # random_sampling_result_txt_path = os.path.join(
        # output_root_folder_path,
        # os.path.basename(kmeans_result_txt_path)).replace(".txt", "") +
        # "_random_sampling_tr_T{}_r{}_nr{}.txt".format(tr_rice_amount +
        # tr_non_rice_amount, tr_rice_amount, tr_non_rice_amount)
        kmeans_result_txt_path_base_name_split = os.path.basename(
            kmeans_result_txt_path).split("_")
        random_sampling_result_txt_name = "Round{}_{}.txt".format(
            Round_number, kmeans_result_txt_path_base_name_split[2])
        random_sampling_result_txt_path = os.path.join(
            output_root_folder_path, random_sampling_result_txt_name)
        logger.info("\t output_root_folder_path:{}".format(output_root_folder_path))
        logger.info("\t os.path.basename(kmeans_result_txt_path):{}".format(
            os.path.basename(kmeans_result_txt_path)))
        logger.info("\t random_sampling_result_txt_path:{}".format(
            random_sampling_result_txt_path))

        # clear existing random_sampling_result_txt
        with open(random_sampling_result_txt_path, 'w'):
            pass
        # training dataset寫入txt
        self.write_2_txt(
            random_sampling_result_txt_path,
            usage="training",
            random_sampling_ds_list=random_training_ds_list)

        # validation dataset寫入txt
        self.write_2_txt(
            random_sampling_result_txt_path,
            usage="validation",
            random_sampling_ds_list=random_validation_ds_list)

        return training_ds_frame_dataset, validation_ds_frame_dataset

    def write_2_txt(
            self,
            random_sampling_result_txt_path,
            usage, random_sampling_ds_list):
        """
        被使用時機
        1.從完整航照產生坵塊資料後, 要進行隨機抽樣時會透過do_one_round_random_sampling_processing
            呼叫這個function
        2.從framelet產生坵塊資料時會直接呼叫這個function

        usage:
            "training" or "validation"
            表示此次的寫入是寫入訓練資料集或驗證資料集

        """

        # 用於紀錄隨機抽樣結果的txt
        random_sampling_result_f = open(random_sampling_result_txt_path, 'a')

        # 寫入用於分隔訓練、驗證資料集的字串到txt
        if usage == "training":
            random_sampling_result_f.write(
                "-" + r"random_training_ds_list" + "-" + "\n")
        elif usage == "validation":
            random_sampling_result_f.write(
                "-" + r"random_validation_ds_list" + "-" + "\n")

        # 依序將list中的每一筆坵塊資料寫入txt
        for parcel_data in random_sampling_ds_list:
            """
            each parcel_data here is a string recording the path of parcel image and corresponding GT label
            the format is "{the path of parcel image} , {GT label}"
            """
            # "{the path of parcel image} , {GT label}"
            write_parcel_data = parcel_data.parcel_NIRRGA_path + \
                                ", " + str(parcel_data.parcel_GT_label)

            random_sampling_result_f.write(write_parcel_data + "\n")

    def converting_parcel_data_list_to_list_of_dictionary(
            self, parcel_data_list):
        """
        return statement structure
        class parcel_data =
            parcel_NIRRGA_path
            parcel_GT_label

        dictionary={
            "parcel_NIRRGA_path_list":[],
            "parcel_GT_label_list":[]
        }

        """

        dictionary = {
            "parcel_NIRRGA_path_list": [],
            "parcel_GT_label_list": []
        }
        for parcel_data in parcel_data_list:
            dictionary["parcel_NIRRGA_path_list"].append(
                parcel_data.parcel_NIRRGA_path)
            dictionary["parcel_GT_label_list"].append(
                parcel_data.parcel_GT_label)
        return dictionary

    def converting_list_of_dictionary_to_parcel_data_list(self,
                                                          list_of_dictionary,
                                                          new_source_folder_path,
                                                          replaced_str_front_part
                                                          ):
        """
        0711 取得不同大小 但是parcel編號與0619 24F 110x110 crop all內的相同
        進行 改變shape的比較
        餵入 寫檔紀錄+複製 funciton前 list_of_dictionary要先轉成parcel data list
        """
        copy_same_parcel_data_list = []
        for NIRRGA_path, GT_label in zip(
                list_of_dictionary["parcel_NIRRGA_path_list"], list_of_dictionary["parcel_GT_label_list"]):
            # 改變路徑
            new_source_folder_dirname = os.path.basename(
                new_source_folder_path)
            # logger.info("new_source_folder_dirname:", new_source_folder_dirname)
            NIRRGA_path = NIRRGA_path.replace("{}".format(
                replaced_str_front_part), new_source_folder_dirname)

            copy_same_parcel_data_list.append(
                self.Parcel_data(NIRRGA_path, GT_label))

        return copy_same_parcel_data_list

    class Parcel_data:
        def __init__(self, parcel_NIRRGA_path, parcel_GT_label):
            self.parcel_NIRRGA_path = parcel_NIRRGA_path
            self.parcel_GT_label = parcel_GT_label

    def load_txt_storage(self, result_txt):
        """
        載入已經過Kmeans clustering或ramdom sampling以txt紀錄的結果

        loaded_result_storage:
        {
            "set_name":[
                parcel_data,
                ...
            ],

            "set_name":[
                parcel_data,
                ...
            ]
        }
        """
        load_result_txt_f = open(result_txt, 'r')

        loaded_result_storage = {}
        for line in load_result_txt_f.readlines():
            line = line.strip()

            if line[0] == '-' and line[-1] == '-':  # set_name
                set_name = line.strip('-')  # name of set
                loaded_result_storage[set_name] = []
            else:
                NIRRGA_path, GT_label = self.restore_one_parcel_data_from_txt(
                    line)
                loaded_result_storage[set_name].append(
                    self.Parcel_data(NIRRGA_path, GT_label))
        return loaded_result_storage

    def restore_one_parcel_data_from_txt(self, one_parcel_data):
        """
        取出保存於txt中的一筆parcel data
        e.g. "F:\\Code review  Winnie research\0406 Crop fixed shape 600x800/parcel_NIRRGA\0\f0_18_parcel_NIRRGA.npy, 0"

        """
        one_parcel_data_split = one_parcel_data.split(',')
        NIRRGA_path = one_parcel_data_split[0]
        GT_label = int(one_parcel_data_split[1].strip())
        # logger.info("NIRRGA_path   ,   GT_label:",NIRRGA_path, GT_label)
        return NIRRGA_path, GT_label

    def get_target_kmeans_result_txt_list(
            self, folder_path, kmeans_result_txt_target_string):
        """
        取得kmeans_result_dirname中, 檔名包含target_string的所有kmeans_result txt路徑(不含random sampling的結果),
        並根據Round編號排序
        """
        txt_list = glob.glob(folder_path + "/*.txt")
        target_txt_list = [
            txt_path for txt_path in txt_list if kmeans_result_txt_target_string in txt_path]

        target_txt_list = [
            txt_path for txt_path in target_txt_list if "Calibrated" not in txt_path]

        target_txt_list = [
            txt_path for txt_path in target_txt_list if "random_sampling" not in txt_path]
        target_txt_list.sort(
            key=lambda txt_path: os.path.basename(txt_path).split("_")[2])

        logger.info("get_target_kmeans_result_txt_list   =>  target_txt_list: %s",
                    " ".join(target_txt_list))

        if target_txt_list == []:
            return None
        return target_txt_list

    def get_target_random_sampling_result_txt_list(
            self, folder_path, random_sampling_result_txt_target_string):
        """
        取得kmeans_result_dirname中, 檔名包含target_string的所有random sampling的結果txt路徑, 並根據Round編號排序
        """
        txt_list = glob.glob(folder_path + "/*.txt")
        logger.info("\t---get_target_random_sampling_result_txt_list---")
        logger.info("txt_list: %s", ",".join(txt_list))
        target_txt_list = [
            txt_path for txt_path in txt_list if random_sampling_result_txt_target_string in txt_path]
        target_txt_list = sorted([
            txt_path for txt_path in target_txt_list if "TWCC" not in txt_path])

        if target_txt_list == []:
            return None
        return target_txt_list


# -

# ---------------------


def combine_frame_dataset_from_each_frame(
        frame_codename_list, frame_dataset_list):
    """
    根據要合併的frame_codename, 將先前從數個frame產生並保存的dictionary型態dataset(e.g. parcel image...)

    合併到一個dictionary型態dataset

    e.g. 若frame_codename_list==[0,2,3], 就將frame_codename為0,2,3的dictionary型態dataset合併

    combined_frame_dataset:
        {
            "0":[
                "path, GT label",
                "path, GT label",
                ...
            ],

            "2":[
                "path, GT label",
                "path, GT label",
                ...
            ],

            "3":[
                "path, GT label",
                "path, GT label",
                ...
            ]
        }

    frame_codename_list:
        以list存放要合併的frame_codename

    frame_dataset_list:
        所有經過前處理產生的dictionary型態dataset

    """

    combined_frame_dataset = {
        "frame_codename": [],
        "parcel_NIRRGA_saving_dirname": [],
        "parcel_NIRRGA_path_list": [],
        "parcel_GT_label_list": []
    }

    for frame_dataset_dic in frame_dataset_list:
        if frame_dataset_dic["frame_codename"] in frame_codename_list:
            combined_frame_dataset["frame_codename"].append(
                frame_dataset_dic["frame_codename"])
            combined_frame_dataset["parcel_NIRRGA_saving_dirname"].append(
                frame_dataset_dic["parcel_NIRRGA_saving_dirname"])
            combined_frame_dataset["parcel_NIRRGA_path_list"].extend(
                frame_dataset_dic["parcel_NIRRGA_path_list"])
            combined_frame_dataset["parcel_GT_label_list"].extend(
                frame_dataset_dic["parcel_GT_label_list"])
    return combined_frame_dataset


# +


def kappa(y_true, y_pred, argm=True, weight=False):
    if argm:
        y_pred2 = tf.reshape(tf.argmax(y_pred, axis=-1), [-1])
    else:
        y_pred2 = tf.reshape(y_pred, [-1])
    y_true2 = tf.reshape(y_true, [-1])

    # WEIGHT FOR CONFUSION MATRIX
    if weight == False:
        conf = tf.cast(tf.math.confusion_matrix(
            y_true2, y_pred2, num_classes=2), float32)
    #     else:
    #         weight = tf.where(y_true2==0, 0, 1)
    #         conf = tf.cast(tf.math.confusion_matrix(y_true2, y_pred2,num_classes=3, weights=weight),float32)

    # CALCULATE KAPPA
    actual_ratings_hist = tf.reduce_sum(conf, axis=1)
    pred_ratings_hist = tf.reduce_sum(conf, axis=0)

    nb_ratings = tf.shape(conf)[0]
    weight_mtx = tf.zeros([nb_ratings, nb_ratings], dtype=tf.float32)
    diagonal = tf.ones([nb_ratings], dtype=tf.float32)
    weight_mtx = tf.linalg.set_diag(weight_mtx, diagonal=diagonal)
    gc = actual_ratings_hist * pred_ratings_hist
    conf = tf.cast(conf, float32)
    totaln = tf.cast(tf.shape(y_true2)[0], float32)
    up = tf.cast(totaln * tf.reduce_sum(conf * weight_mtx),
                 float32) - tf.cast(tf.reduce_sum(gc), float32)
    down = tf.cast(totaln ** 2 - tf.reduce_sum(gc), float32)
    # logger.info(weight_mtx,gc,conf,up,down)

    if tf.math.is_nan(up / down):
        return 0.
    return up / down


# +

def generate_ds(
        split_up_fourth_channel,
        masking,
        normalizing,
        per_image_standardization,
        random_brightness_max_delta,
        image_path_list,
        GT_label_list,
        aug,
        bands,
        change_NirRGA_to_RGNirA=False):
    """
    generator
    """
    if random_brightness_max_delta != 0:
        threshold_of_random_value_for_BR = 0.6  # 與1.3相同
        logger.info("The probability of triggering random brightness needs to be greater than {}".format(
            threshold_of_random_value_for_BR))

    if per_image_standardization == True:
        logger.info("0714 New per_image_standardization")

    if change_NirRGA_to_RGNirA == True:
        logger.info("四通道影像順序 NirRGA 改為 RGNirA")
    else:
        logger.info("四通道影像順序 NirRGA")

    if masking == False:
        # 有做Masking的話，不須考慮Annotation mask
        # 0716以前不做遮罩時，四通道影像中的annoatation mask的值為0和1，可能對凸顯focus parcel的效果不明顯
        # 嘗試將Annotation mask中的1改成255
        set_1_of_A_as_255 = True

        if set_1_of_A_as_255 == True:
            logger.info("不進行遮罩 且 Annotation mask的focus parcel像素值為255")

    i = 0  # index of parcel data
    while i < len(image_path_list):  # stop = len(labels)
        # logger.info("\n image_path_list[i]:", image_path_list[i].decode('utf-8'))

        # npy file為四通道{NIR, R, G, Annotation}
        # decode  # np.load用於載入npy檔
        img = np.load(image_path_list[i].decode('utf-8'))

        # 改變影像通道順序 NIRRG => RGNIR 因為VGG16的preprocess_input layers的輸入為RGB影像
        if change_NirRGA_to_RGNirA == True:
            nir_band = img[:, :, 0].copy()
            r_band = img[:, :, 1].copy()
            g_band = img[:, :, 2].copy()
            img[:, :, 0] = r_band
            img[:, :, 1] = g_band
            img[:, :, 2] = nir_band

        # 影像處理 per_image_standardization
        if per_image_standardization == True:
            normalizing = False

            NIRRG_img = img[:, :, 0:3]
            A_img = img[:, :, 3]

            NIRRG_img = tf.image.per_image_standardization(NIRRG_img)
            img = np.stack((NIRRG_img[:, :, 0],
                            NIRRG_img[:, :, 1],
                            NIRRG_img[:, :, 2],
                            A_img), axis=2)

        # ----output----

        # npy為四通道影像的資料，目前只有要加上preprocess_input layer時，
        # 需要將NIRRG image和Annotation image分開才用到split_up_fourth_channel == True
        if split_up_fourth_channel == True:
            # 拆分成NIRRG和Annotation mask
            nirrg = img[:, :, 0:3]

            if set_1_of_A_as_255 == True:
                img[:, :, 3] = img[:, :, 3] * 255
            annotation_mask = img[:, :, 3]
            annotation_mask = tf.expand_dims(annotation_mask, -1)

            # numpy array 轉成 tensor
            nirrg = tf.convert_to_tensor(nirrg)
            annotation_mask = tf.convert_to_tensor(annotation_mask)

            # constant 轉成 tensor
            GT_label = tf.convert_to_tensor([GT_label_list[i]])
            GT_label = tf.cast(GT_label, tf.float32)

            if aug == True:
                if rd.random() > 0.5:
                    nirrg = tf.image.flip_left_right(nirrg)
                    annotation_mask = tf.image.flip_left_right(annotation_mask)
                if rd.random() > 0.5:
                    nirrg = tf.image.flip_up_down(nirrg)
                    annotation_mask = tf.image.flip_up_down(annotation_mask)

            i += 1
            yield (nirrg, annotation_mask), GT_label

        if split_up_fourth_channel == False:
            # 先做亮度變化 才做masking
            if aug == True:  # 訓練驗證集
                aug_img = img.copy()  # 四通道
                # 隨機變化影像亮度 (Ida有做)
                if random_brightness_max_delta != 0:
                    if rd.random() > threshold_of_random_value_for_BR:
                        # 只對NIRRG三通道進行隨機亮度變化
                        aug_img[:, :, 0:3] = tf.image.random_brightness(
                            aug_img[:, :, 0:3], random_brightness_max_delta)
                        img = aug_img

            img = img.astype("float32")  # 資料型態轉換成float32

            if masking == True:  # 要產生三通道NIRRG影像，使用annotation mask進行遮罩後, 捨棄annotation mask, 輸出三通道Masked NIRRG
                if bands==3:
                    # Masking
                    img[:, :, 0] = img[:, :, 0] * img[:, :, 3]
                    img[:, :, 1] = img[:, :, 1] * img[:, :, 3]
                    img[:, :, 2] = img[:, :, 2] * img[:, :, 3]
                    img = img[:, :, 0:3]  # 只取 Masked NIRRG三通道
                if bands==4:
                    img[:, :, 0] = img[:, :, 0] * img[:, :, 4]
                    img[:, :, 1] = img[:, :, 1] * img[:, :, 4] 
                    img[:, :, 2] = img[:, :, 2]  * img[:, :, 4]
                    img[:, :, 3] = img[:, :, 3]  * img[:, :, 4]
                    img = img[:, :, 0:4]
                if normalizing == True:
                    if bands==3:                        
                        img = img.astype("float32")
                        img[:, :, 0:3] = img[:, :, 0:3] / 255  # NIRRG三通道除以255
                    if bands==4:                        
                        img = img.astype("float32")
                        img[:, :, 0:4] = img[:, :, 0:4] / 255  # NIRRG三通道除以255
            elif masking == False:  # 要產生四通道NIRRGA影像

                if set_1_of_A_as_255 == True:
                    img[:, :, 3] = img[:, :, 3] * 255

                if normalizing == True:
                    img = img.astype("float32")
                    img[:, :, 0:4] = img[:, :, 0:4] / 255

            # numpy array 轉成 tensor
            img = tf.convert_to_tensor(img)

            # constant 轉成 tensor
            GT_label = tf.convert_to_tensor([GT_label_list[i]])
            GT_label = tf.cast(GT_label, tf.float32)

            if aug == True:
                if rd.random() > 0.5:
                    img = tf.image.flip_left_right(img)
                if rd.random() > 0.5:
                    img = tf.image.flip_up_down(img)

            i += 1
            yield img, GT_label


def create_dataset(
        arguments,
        image_path_list,
        GT_label_list,
        usage=str("test")):
    """
    使用 tf.data.Dataset.from_generator 將Python的generator包裝成tf Dataset物件

    """

    # output_shapes参数不是必须的，但是极力推荐指定该参数。因为很多TensorFlow operations不支持unknown rank的Tensor。
    # 如果某一个axis的长度是未知或者可变的，可以在output_shapes参数中将其置为None。

    if usage == "train":
        aug = True  # data augmentation
    else:
        aug = False

    if arguments.masking == True:  # 產生三通道Masked NIRRG
        if arguments.data_shape[2]==3:
            ds = tf.data.Dataset.from_generator(generate_ds,
                                                (tf.float32, tf.float32),
                                                output_shapes=(
                                                    (None, None, 3), (1)),
                                                args=[
                                                    arguments.split_up_fourth_channel,
                                                    arguments.masking,
                                                    arguments.normalizing,
                                                    arguments.per_image_standardization,
                                                    arguments.random_brightness_max_delta,
                                                    image_path_list,
                                                    GT_label_list,
                                                    aug,arguments.data_shape[2]]
                                                )
        if arguments.data_shape[2]==4:
            ds = tf.data.Dataset.from_generator(generate_ds,
                                    (tf.float32, tf.float32),
                                    output_shapes=(
                                        (None, None, 4), (1)),
                                    args=[
                                        arguments.split_up_fourth_channel,
                                        arguments.masking,
                                        arguments.normalizing,
                                        arguments.per_image_standardization,
                                        arguments.random_brightness_max_delta,
                                        image_path_list,
                                        GT_label_list,
                                        aug,arguments.data_shape[2]]
                                    )
    if arguments.masking == False:  # 產生 "四通道Non-masked NIRRGA" 或 "三通道Non-masked NIRRG+單通道Annotation mask"
        if arguments.split_up_fourth_channel == False:  # 產生 "四通道Non-masked NIRRGA"
            ds = tf.data.Dataset.from_generator(generate_ds,
                                                (tf.float32, tf.float32),
                                                output_shapes=(
                                                    (None, None, 4), (1)),
                                                args=[
                                                    arguments.split_up_fourth_channel,
                                                    arguments.masking,
                                                    arguments.normalizing,
                                                    arguments.per_image_standardization,
                                                    arguments.random_brightness_max_delta,
                                                    image_path_list,
                                                    GT_label_list,
                                                    aug,arguments.data_shape[2]]
                                                )
        elif arguments.split_up_fourth_channel == True:  # 產生 "三通道Non-masked NIRRG+單通道Annotation mask"
            ds = tf.data.Dataset.from_generator(generate_ds,
                                                output_types=(
                                                    (tf.float32, tf.float32), tf.float32),
                                                output_shapes=(
                                                    ([None, None, 3], [None, None, 1]), [1]),
                                                args=[
                                                    arguments.split_up_fourth_channel,
                                                    arguments.masking,
                                                    arguments.normalizing,
                                                    arguments.per_image_standardization,
                                                    arguments.random_brightness_max_delta,
                                                    image_path_list,
                                                    GT_label_list,
                                                    aug,arguments.data_shape[2]]
                                                )
    parcel_count = len(GT_label_list)

    if usage == "train":
        logger.info(" create_dataset()...  for Training")
        ds = ds.cache()

        # logger.info("tf_dataset  cache & shuffle_buffer20000 & repeat & batch & prefetch")
        # ds = ds.shuffle(20000).repeat().batch(arguments.BATCH_SIZE) # shuffle

        logger.info("tf_dataset  cache & No_shuffle & repeat & batch & prefetch")
        ds = ds.repeat().batch(arguments.BATCH_SIZE)  # no shuffle

        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        if arguments.random_brightness_max_delta != 0:
            logger.info(" random image brightness arguments.random_brightness_max_delta:{}  (1.3 Unet done)".format(
                arguments.random_brightness_max_delta))

        return ds, parcel_count
    elif usage == "test":
        logger.info(" create_dataset()...  for Testing")
        ds = ds.batch(arguments.BATCH_SIZE)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds, parcel_count


# -


# +
class Training:
    def __init__(
            self,
            saving_model_dir_path,  # ./data/For_training_testing/70x70/model
            data_shape,
            Round_number,
            EPOCH_N,
            BATCH_SIZE,
            optimizer_learning_rate,
            training_ds_frame_dataset, validation_ds_frame_dataset):
        """
        用法:
            t = Training(saving_model_dir_path, Round_number, EPOCH_N, BATCH_SIZE, optimizer_learning_rate, training_ds_frame_dataset, validation_ds_frame_dataset)
            create_tensorflow_dataset(self, normalizing=False)
            model = parcel_based_CNN.build()
            t.train(model)

        """
        self.saving_model_dir_path = saving_model_dir_path
        self.EPOCH_N = EPOCH_N
        self.BATCH_SIZE = BATCH_SIZE
        self.optimizer_learning_rate = optimizer_learning_rate
        self.training_ds_frame_dataset = training_ds_frame_dataset
        self.validation_ds_frame_dataset = validation_ds_frame_dataset
        self.data_shape = data_shape
        self.Round_number = Round_number

    def create_tensorflow_dataset(self, arguments):
        """
        建立用於訓練和驗證的tensorflow dataset
        """
        logger.info("\tcreate_tensorflow_dataset()...")
        logger.info("masking: %s", str(arguments.masking))
        logger.info("split_up_fourth_channel: %s", str(arguments.split_up_fourth_channel))

        self.training_ds, self.training_data_count = create_dataset(
            arguments,
            image_path_list=self.training_ds_frame_dataset["parcel_NIRRGA_path_list"],
            GT_label_list=self.training_ds_frame_dataset["parcel_GT_label_list"],
            usage="train"
        )

        self.validation_ds, self.validation_data_count = create_dataset(
            arguments,
            image_path_list=self.validation_ds_frame_dataset["parcel_NIRRGA_path_list"],
            GT_label_list=self.validation_ds_frame_dataset["parcel_GT_label_list"],
            usage="train"
        )

        # training dataset會拆分成幾個batch用於訓練
        # data_count/BATCH_SIZE 後向上取整數，目的為取完每一筆資料

        # self.STEPS_PER_EPOCH: 每個epoch進行訓練的step數(每個step會使用一個batch的資料)
        self.STEPS_PER_EPOCH = math.ceil(
            self.training_data_count / self.BATCH_SIZE)

        # self.VALIDATION_STEPS: 每個epoch進行驗證的step數(每個step會使用一個batch的資料)
        self.VALIDATION_STEPS = math.ceil(
            self.validation_data_count / self.BATCH_SIZE)

    def model_saving_setting(
            self, model_checkpoint_monitor="val_loss", model_checkpoint_mode="min"):
        """
        在model_compile時設定保存model的check point，根據self.test_only決定是否刪除已存在的相同名稱目錄

        """
        # self.saving_model_dir_path = self.saving_model_dir_path + "_" + self.model_codename
        if self.test_only == False:
            if os.path.isdir(self.saving_model_dir_path):
                logger.info("存放model的dir已存在=>刪除 重建存放資料夾")
                shutil.rmtree(self.saving_model_dir_path)
                os.mkdir(self.saving_model_dir_path)
            else:
                os.mkdir(self.saving_model_dir_path)
        else:
            # test_only == True => 不刪除已存在的model
            pass

        # 每個epoch保存model
        # self.mc = tf.keras.callbacks.ModelCheckpoint((self.saving_model_dir_path + r"/" + self.model_codename + " e{epoch:02d}--{val_loss:.2f}.h5"), monitor="val_loss", mode=model_checkpoint_mode, verbose=1, save_best_only=False)

        # self.mc_path = self.saving_model_dir_path + r"/" + self.model_codename + ".h5"
        # self.mc_valacc_path = self.saving_model_dir_path + r"/" + self.model_codename + "_val_acc.h5"
        self.mc_path = self.saving_model_dir_path + r"/" + "model.h5"
        self.mc_valacc_path = self.saving_model_dir_path + r"/" + "model_val_acc.h5"
        # self.mc_valkappa_path = self.saving_model_dir_path + r"/" + self.model_codename + "_val_kappa.h5"
        self.mc = tf.keras.callbacks.ModelCheckpoint(
            (self.mc_path), monitor=model_checkpoint_monitor, mode=model_checkpoint_mode, verbose=1,
            save_best_only=True)  # save the model on best training accuracy
        self.mc_valacc = tf.keras.callbacks.ModelCheckpoint(
            (self.mc_valacc_path), monitor="val_acc", mode="max", verbose=1,
            save_best_only=True)  # save model when best validation accuracy
        # self.mc_valkappa = tf.keras.callbacks.ModelCheckpoint((self.mc_valkappa_path), monitor="val_kappa", mode="max", verbose=1, save_best_only=True)

    def model_compile(self, model, model_name,
                      test_only=False,
                      optimizer_name="Adam", lr=0.001, m=0.5,
                      SGD_nesterov=True,
                      warmup_epoch=0,
                      sgd_momentum=0.5,
                      focal_loss=False):
        """
        在此做self.model_saving_setting
        """

        self.test_only = test_only

        # loss function 預設使用SparseCategoricalCrossentropy
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        if focal_loss == True:
            logger.info("Use SparseCategoricalFocalLoss as Loss function")
            from focal_loss import SparseCategoricalFocalLoss
            loss_function = SparseCategoricalFocalLoss(gamma=2)

        if warmup_epoch != 0:
            self.warmup_epoch = warmup_epoch  # Number of warmup epochs.
            # Compute the number of warmup batches.
            warmup_batches = warmup_epoch * self.STEPS_PER_EPOCH
            # Create the Learning rate scheduler.
            self.warm_up_lr_Scheduler = WarmUpLearningRateScheduler(
                warmup_batches, init_lr=lr, verbose=0)
        else:
            self.warmup_epoch = 0



        if optimizer_name == "Adam":
            opt=keras.optimizers.Adam(learning_rate=lr)
            #opt=tf.train.Optimizer.Adam(learning_rate=lr)
        elif optimizer_name == "SGDm":
            opt=keras.optimizers.SGD(
                              learning_rate=lr,
                              momentum=sgd_momentum,
                              decay=0.0,
                              nesterov=SGD_nesterov)
        try: 
            opt =    tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite(
            opt, loss_scale='dynamic'
        )
        except:
            pass
        model.compile(loss=loss_function,
                          optimizer=opt,
                          metrics=['acc', kappa])
        # 建立存放model的資料夾，若已存在資料夾會先刪除
        self.model_saving_setting(
            model_checkpoint_monitor="val_loss", model_checkpoint_mode="min")
        self.model = model  # 經過compile

        return model, model_name

    def train(self, lr_d=False):
        t1 = time.time()

        self.callbacks = [self.mc, self.mc_valacc]

        if self.warmup_epoch != 0:
            self.callbacks.append(self.warm_up_lr_Scheduler)

        # learning rate decrease
        if lr_d == True:
            # 若經過15epoch, val_loss沒有更低, 就將learning rate*0.6, 最小只會將lr設為0.0001
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.6,
                                                             patience=15, min_lr=0.0001)
            self.callbacks.append(reduce_lr)

        history_obj = self.model.fit(
            self.training_ds,
            epochs=self.EPOCH_N,
            steps_per_epoch=self.STEPS_PER_EPOCH,
            validation_steps=self.VALIDATION_STEPS,
            validation_data=self.validation_ds,
            callbacks=self.callbacks
        )
        
        save_path = "data/train_test/For_training_testing/%dx%d/train_test/final_weight.h5" % (
                                                        self.data_shape[0], self.data_shape[1])
        self.model.save(save_path)
        t2 = time.time()
        total_sec = t2 - t1
        logger.info("\n***********************************")
        logger.info("EPOCH_N= {}    model.fit() takes:  {} min\n".format(
            self.EPOCH_N, round(total_sec / 60, 2)))
        logger.info("***********************************")
        total_min = round(total_sec / 60, 0)
        return history_obj, total_min


# -

class WarmUpLearningRateScheduler(keras.callbacks.Callback):
    """Warmup learning rate scheduler
    """

    def __init__(self, warmup_batches, init_lr, verbose=0):
        """Constructor for warmup learning rate scheduler
        Arguments:
            warmup_batches {int} -- Number of batch for warmup.
            init_lr {float} -- Learning rate after warmup.
        Keyword Arguments:
            verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count * self.init_lr / self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                logger.info('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
                            'rate to %s.' % (self.batch_count + 1, lr))



def show_history_acc_loss(saving_png_dir_path, history_list,
                          target_monitor="val_acc", network_name="network_name", time_cost=0):
    """
    target_monitor:
        "val_acc", "val_loss", "val_kappa"
    limit_y:
        是否限制畫布中的y值, accuracy在0~1, loss值在0~5

    """
    crilist = list(history_list[0].keys())
    if "acc" in target_monitor:
        target_monitor_mode = "max"
    elif "loss" in target_monitor:
        target_monitor_mode = "min"
    elif "kappa" in target_monitor:
        target_monitor_mode = "max"

    plt.rc('font', size=12)
    for n, r in enumerate(history_list):  # r為history_list內的內容 dictionary
        # logger.info("\nn:{} , r:{}".format(n,r))
        if target_monitor_mode == "max":
            target_value = max(r[target_monitor])
        else:
            target_value = min(r[target_monitor])
        target_value_epoch_number = r[target_monitor].index(target_value)

        target_monitor_val_loss = "val_loss"
        target_value_val_loss = min(r[target_monitor_val_loss])
        target_value_val_loss_epoch_number = r[target_monitor_val_loss].index(
            target_value_val_loss)

        logger.info("\n************** ")
        logger.info(
            "target  %s epoch: %s \nval_loss: %s , val_acc: %s \ntr_loss: %s , tr_acc: %s \nval_kappa: %s , tr_kappa: %s \n",
            target_monitor_val_loss,
            str(target_value_val_loss_epoch_number + 1),
            str(round(r["val_loss"][target_value_val_loss_epoch_number], 4)),
            str(round(r["val_acc"][target_value_val_loss_epoch_number], 4)),
            # as in training loss
            str(round(r["loss"][target_value_val_loss_epoch_number], 4)),
            # as in training acc
            str(round(r["acc"][target_value_val_loss_epoch_number], 4)),
            str(round(r["val_kappa"][target_value_val_loss_epoch_number], 4)),
            # as in training kappa
            str(round(r["kappa"][target_value_val_loss_epoch_number], 4))
        )

        logger.info("\n**********************************************************************")
        logger.info("Time Cost: %s min", str(int(round(time_cost, 0))))
        debug_logger.debug("Time Cost: %s min", str(int(round(time_cost, 0))))
        logger.info("target monitor  %s epoch: %s \nval_loss: %s , val_acc: %s \ntr_loss: %s , tr_acc: %s \n",
                    target_monitor,
                    str(target_value_epoch_number + 1),
                    str(round(r["val_loss"][target_value_epoch_number], 4)),
                    str(round(r["val_acc"][target_value_epoch_number], 4)),
                    str(round(r["loss"][target_value_epoch_number], 4)),
                    str(round(r["acc"][target_value_epoch_number], 4))
                    )
        max_val_acc = round(r["val_acc"][target_value_epoch_number], 4)

        # logger.info("最小{}所在epoch:{}\nval_loss:{} , val_acc:{}\ntr_loss:{} , tr_acc:{}\n".format(
        #    target_monitor_val_loss,
        #    target_value_val_loss_epoch_number+1,
        #    round(r["val_loss"][target_value_val_loss_epoch_number],4),
        #    round(r["val_acc"][target_value_val_loss_epoch_number],4),
        #    round(r["loss"][target_value_val_loss_epoch_number],4),
        #    round(r["acc"][target_value_val_loss_epoch_number],4)
        # ))

    plt.figure(figsize=(16, 8))
    # accuracy畫布
    # fig, ax = plt.subplots(2, 3, figsize=(16, 8))

    axes = plt.subplot(2, 3, 1)
    axes.set_title(network_name, fontsize=10)
    # plt.suptitle('{}'.format(network_name),fontsize=10)
    line_name_list = []
    for i in crilist:
        if "acc" in i:
            line_name_list.append(i)
            try:
                plt.plot(np.array(r[i]), linewidth=3)
            except BaseException:
                debug_logger.warning(f"exception found in {r.history[i]}")
                logger.info(r.history[i])

                plt.plot(np.array(r.history[i]), linewidth=3)

    plt.legend(line_name_list, fontsize=10)

    # accuracy畫布 0~1
    # plt.subplot(2,3,4)
    # plt.suptitle('CV%d'%(n+1),fontsize=10)
    # line_name_list = []
    # for i in crilist:
    #  if "acc" in i:
    #    line_name_list.append(i)
    #    try:
    #      plt.plot(np.array(r[i]),linewidth=3)
    #    except:
    #      logger.info("except")
    #      logger.info(r)
    #      logger.info(r.history[i])
    #
    #      plt.plot(np.array(r.history[i]),linewidth=3)
    #
    # plt.legend(line_name_list,fontsize=10)
    # plt.ylim(0.7,1)
    #

    # loss畫布
    plt.subplot(2, 3, 2)
    # plt.suptitle('CV%d'%(n+1),fontsize=25)
    line_name_list = []
    for i in crilist:
        if "loss" in i:
            line_name_list.append(i)
            try:
                plt.plot(np.array(r[i]), linewidth=3)
            except BaseException:
                logger.info("except")
                logger.info(r)
                logger.info(r.history[i])

                plt.plot(np.array(r.history[i]), linewidth=3)

    plt.legend(line_name_list, fontsize=10)

    # # loss畫布 0~2
    # plt.subplot(2,3,5)
    # # plt.suptitle('CV%d'%(n+1),fontsize=25)
    # line_name_list = []
    # for i in crilist:
    #   if "loss" in i:
    #     line_name_list.append(i)
    #     try:
    #       plt.plot(np.array(r[i]),linewidth=3)
    #     except:
    #       logger.info("except")
    #       logger.info(r)
    #       logger.info(r.history[i])
    #
    #       plt.plot(np.array(r.history[i]),linewidth=3)
    #
    # plt.legend(line_name_list,fontsize=10)
    # plt.ylim(0,1.5)
    #
    # kappa畫布
    plt.subplot(2, 3, 3)
    # plt.suptitle('CV%d'%(n+1),fontsize=25)
    line_name_list = []
    for i in crilist:
        if "kappa" in i:
            line_name_list.append(i)
            try:
                plt.plot(np.array(r[i]), linewidth=3)
            except BaseException:
                debug_logger.warning("exception found in r.history[i]")
                logger.info(r)
                logger.info(r.history[i])

                plt.plot(np.array(r.history[i]), linewidth=3)

    plt.legend(line_name_list, fontsize=10)

    # # kappa畫布 0~2
    # plt.subplot(2,3,6)
    # # plt.suptitle('CV%d'%(n+1),fontsize=25)
    # line_name_list = []
    # for i in crilist:
    #   if "kappa" in i:
    #     line_name_list.append(i)
    #     try:
    #       plt.plot(np.array(r[i]),linewidth=3)
    #     except:
    #       logger.info("except")
    #       logger.info(r)
    #       logger.info(r.history[i])
    #
    #       plt.plot(np.array(r.history[i]),linewidth=3)
    #
    # plt.legend(line_name_list,fontsize=10)
    # plt.ylim(0.4,1)

    plt.savefig(saving_png_dir_path)
    debug_logger.debug("max_val_acc:%f" % max_val_acc)

    return max_val_acc


# -


# +

class Boundary_box:
    def __init__(self, min_row, max_row, min_col, max_col) -> None:
        self.min_row = min_row  # 高 y軸值
        self.max_row = max_row
        self.min_col = min_col  # 寬 x軸值
        self.max_col = max_col
        self.max_row_plus = max_row + 1
        self.max_col_plus = max_col + 1
        self.width = self.max_col_plus - self.min_col
        self.height = self.max_row_plus - self.min_row


def get_GT_label_of_focus_parcel(
        target_parcel_region, whole_frame_parcel_GT_mask):
    """
    焦點坵塊是否為水稻的label, 1 or 0
    """
    # target_parcel_region.coords 目標坵塊內像素 在原圖上的點座標
    # 1=non-rice, 2=rice
    parcel_GT_3class_label = whole_frame_parcel_GT_mask[target_parcel_region.coords[0][0],
    target_parcel_region.coords[0][1]]
    if parcel_GT_3class_label == 2:  # 該坵塊為水稻坵塊
        parcel_GT_label = 1
    else:
        parcel_GT_label = 0

    return parcel_GT_label


def model_testing(
        testing_argument,
        testing_ds_frame_codename,
        frame_name,
        min_valid_parcel_area_size=7000,  # 7000 for testing, 0 for inference
        path_to_parcels_taken_hashmap="data\inference\For_training_testing\320x320\parcel_NIRRGA\0\parcels_that_are_actually_taken.txt"
):
    """

    min_valid_parcel_area_size:
        最小的有效坵塊面積大小
        需要與 "從完整航照產生坵塊資料" 時設定的面積threshold大小相同, 才能正確地對應GT label

    """
    logger.info("\t----model_testing()----")

    # frame_number = splitted_parcel_mask_path_basename[-3]+"_"+splitted_parcel_mask_path_basename[-2]
    frame_number = frame_name
    logger.info("\nFrame_number: %s", str(frame_number))

    # 設定要用於測試的frame codename
    testing_ds_frame_codename_list = [testing_ds_frame_codename]

    testing_ds_frame_dataset = combine_frame_dataset_from_each_frame(frame_codename_list=testing_ds_frame_codename_list,
                                                                     frame_dataset_list=testing_argument.frame_dataset_list)
    """
    testing_ds_frame_codename_list = {
        "testing_ds_frame_codename":[
            "path, GT label", "path, GT label", ...
        ]
    }
    """

    logger.info(r"testing_ds_frame_dataset[\"parcel_NIRRGA_path_list\"][0]: %s",
                testing_ds_frame_dataset["parcel_NIRRGA_path_list"][0])

    # 準備用於測試的tf dataset

    testing_ds, testing_data_count = create_dataset(
        testing_argument,
        testing_ds_frame_dataset["parcel_NIRRGA_path_list"],
        testing_ds_frame_dataset["parcel_GT_label_list"],
        usage=str("test")
    )

    number_of_batch = math.ceil(
        testing_data_count / testing_argument.BATCH_SIZE)  # batch數量
    ds = testing_ds
    model = testing_argument.model

    ## ``` Grad Cam Test code start```
    # print('generate grad cam')
    # from grad_cam import plot_gram_cam

    # save_folder = os.path.join('./data/inference/saved_model_and_prediction/grad_cam/', frame_number)
    # if os.path.isdir(save_folder):
    #     shutil.rmtree(save_folder)

    # os.makedirs(save_folder)

    # for parcel in testing_ds_frame_dataset["parcel_NIRRGA_path_list"]:
    #     # print(parcel)

    #     result_name = parcel.split('/')[-1] + ".png"
    #     result = plot_gram_cam(model, 'conv2d_12', 4, parcel)
    #     # print('saving %s' %result_name)
    #     result.save(os.path.join(save_folder, result_name))
    # return frame_number, 0, 0
    ## ``` Grad Cam Test code end```
    ## ``` Grad Cam Test2 code start```
    # print('generate grad cam')
    # from test_gradcam import save_and_display_gradcam, make_gradcam_heatmap

    # save_folder = os.path.join('./data/inference/saved_model_and_prediction/grad_cam/', frame_number)
    # if os.path.isdir(save_folder):
    #     shutil.rmtree(save_folder)

    # os.makedirs(save_folder)

    # for parcel in testing_ds_frame_dataset["parcel_NIRRGA_path_list"]:
    #     # print(parcel)

    #     result_name = parcel.split('/')[-1] + ".png"

    #     img = np.load(parcel)
    #     img_array = np.expand_dims(img, axis=0)[:,:,:,:4]
    #     heatmap = make_gradcam_heatmap(img_array, model, 'conv2d_12')
    #     result = save_and_display_gradcam(parcel, heatmap)
    #     # print('saving %s' %result_name)
    #     result.save(os.path.join(save_folder, result_name))
    # return frame_number, 0, 0
    ## ``` Grad Cam Test2 code end```
    result = model.evaluate(ds, steps=number_of_batch)  # loss value
    # verbose: 0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行。
    pred = model.predict(ds, verbose=0, steps=number_of_batch)

    # logger.info('fnn pred shape:', pred.shape) # (parcel number, class number)
    # logger.info('predictions:\n', pred) # 各個坵塊的各分類的預測機率
    # for elem in pred[0:10]:
    #     logger.info(elem)

    output_confidence_score = tf.reduce_max(pred, axis=-1)

    output = tf.argmax(pred, axis=-1)
    # p1 1
    # p2 0

    labels = []  # GT labels
    for parcel, value in ds.take(number_of_batch):
        labels.extend(value.numpy())
    # CALCULATE KAPPA BASED ON PARCEL
    ytflat = np.array(labels).astype(
        'uint8').flatten()  # y_true  將labels攤平成一維陣列
    woflat = output.numpy().astype('uint8').flatten()  # output
    # logger.info('ytflat.shape:', ytflat.shape)
    # logger.info('ytflat:', ytflat)
    # logger.info('woflat:', woflat)
    
    parcel_based_kappa = cohen_kappa_score(ytflat, woflat)
    # ----confusion matrix----
    c_matrix = confusion_matrix(ytflat, woflat)
    logger.info("confusion matrix:")
    logger.info(" ".join([str(x) for x in c_matrix.ravel()]))
    logger.info("GT\\prediction")
    c_matrix_sum = c_matrix.sum()
    logger.info("   {}  {}".format(
        str(round(c_matrix[0, 0] / c_matrix_sum, 3)), str(round(c_matrix[0, 1] / c_matrix_sum, 3))))
    logger.info("   {}  {}".format(
        str(round(c_matrix[1, 0] / c_matrix_sum, 3)), str(round(c_matrix[1, 1] / c_matrix_sum, 3))))

    logger.info('\n************************************************************')
    logger.info('  ****** loss & acc & 不考慮面積權parcel based kappa %s , %s , %s',
                str(round(result[0], 3)), str(round(result[1], 3)), str(round(parcel_based_kappa, 3)))
    logger.info('************************************************************')

    if testing_argument.save_prediction == True:
        print("save result to", testing_argument.saved_model_result_folder_path + "/Pred_" + frame_number + ".txt")
        fid_name_list = [os.path.basename(i).split('.')[0].split('_')[-1] for i in testing_ds_frame_dataset["parcel_NIRRGA_path_list"] if i.endswith('.npy')]
        assert len(fid_name_list) == woflat.size
        with open(testing_argument.saved_model_result_folder_path + "/Pred_" + frame_number + ".txt", 'w') as file:
            for fid, number, confidence_score in zip(fid_name_list, woflat, output_confidence_score):
                file.write(f"{fid} {number} {confidence_score}\n")
        # testing_argument.saved_model_result_folder_path
        # im.save(testing_argument.saved_model_result_folder_path +
        #         "/" + separate_basename[0] + "_" + separate_basename[1])
    return frame_number, parcel_based_kappa, parcel_based_kappa
    # ----計算考慮面積大小的parcel based分類結果----
    prediction_result_list = woflat
    GT_list = ytflat
    TP_amount = 0  # 11
    TN_amount = 0  # 10
    FP_amount = 0  # 01
    FN_amount = 0  # 00

    parcel_mask = np.array(Image.open(parcel_mask_path))

    labeled_component_GT_mask, labeled_component_count = label(
        parcel_mask, background=0, return_num=True, connectivity=1)
    parcel_count = -1
    region_list = regionprops(labeled_component_GT_mask)
    # 任何大小的連通區域數量(包含誤標、polygon重疊處)
    logger.info("len(region_list):%d"% len(region_list))

    # determine which parcels are actually taken according to a txt file
    with open(path_to_parcels_taken_hashmap, 'r') as _file_reader:
        parcels_taken_from_region_props = dict()
        num_parcels_taken_from_region_props = 0
        _lines = _file_reader.readlines()
        for _l in _lines:
            _key = _l.split(' ')
            _key = _key[0]

            _value = _l.split(' ')[1]
            _value = _value.rstrip()
            if _value == 'taken':
                num_parcels_taken_from_region_props += 1
            parcels_taken_from_region_props[_key] = _value

        logger.info("number of parcels that are actually taken: (some can be too small so we ignore them) %d" %
                    num_parcels_taken_from_region_props)

    for region in region_list:  # 遍歷每個連通區域
        focus_parcel_region = region
        current_parcel_joined_calculation = False  # 初始化 => 每個坵塊預設為尚未參與權重計算
        # check if region is true in the hashmap
        _key0 = str(region.bbox[0])
        _key1 = str(region.bbox[1])
        _key2 = str(region.bbox[2])
        _key3 = str(region.bbox[3])
        _key = _key0 + '_' + _key1 + '_' + _key2 + '_' + _key3

        # 忽略面積 <min_valid_parcel_area_size 的連通區域
        if region.area > min_valid_parcel_area_size and parcels_taken_from_region_props[_key] == "taken":
            boundary_box = focus_parcel_region.bbox
            min_row = boundary_box[0]
            max_row_plus1 = boundary_box[2]
            max_row = max_row_plus1 - 1
            min_col = boundary_box[1]
            max_col_plus1 = boundary_box[3]
            max_col = max_col_plus1 - 1
            minimum_parcel_image_boundary = Boundary_box(
                min_row, max_row, min_col, max_col)

            # 篩選 若目標坵塊過於細小=>視為無效坵塊
            # 無視掉某邊低於18個像素寬的坵塊(應該都是polygon重疊導致的錯誤坵塊)
            # 寬高皆為18個以上像素寬才是有效坵塊
            if minimum_parcel_image_boundary.height >= 18 and minimum_parcel_image_boundary.width >= 18:
                parcel_count = parcel_count + 1  # 目前正在處理第幾個(有效的)parcel

                # focus_parcel_region.area = 面積 = 像素數量 = 權重
                if GT_list[parcel_count] == 1 and prediction_result_list[parcel_count] == 1:
                    TP_amount = TP_amount + focus_parcel_region.area
                elif GT_list[parcel_count] == 1 and prediction_result_list[parcel_count] == 0:
                    TN_amount = TN_amount + focus_parcel_region.area
                elif GT_list[parcel_count] == 0 and prediction_result_list[parcel_count] == 1:
                    FP_amount = FP_amount + focus_parcel_region.area
                elif GT_list[parcel_count] == 0 and prediction_result_list[parcel_count] == 0:
                    FN_amount = FN_amount + focus_parcel_region.area

                current_parcel_joined_calculation = True

        # 在上面沒有參與到權重的計算, 代表是面積太小或是寬高不滿足有效條件的"無效"坵塊,
        # 將會作為model判為非水稻的結果參與計算(實際上沒有給model判斷)
        if current_parcel_joined_calculation == False:
            # "無效"坵塊在GT_list中沒有對應的GT(因為沒有紀錄),
            # "無效"坵塊的GT label = "無效"坵塊在parcel mask上的pixel value(1 or 2)
            # "無效"坵塊不會給model分類, 視為model判為非水稻即可

            invalid_parcel_GT_label = get_GT_label_of_focus_parcel(
                focus_parcel_region, whole_frame_parcel_GT_mask=parcel_mask)  # 此"無效"的目標坵塊的GT label
            if invalid_parcel_GT_label == 0:  # FN
                FN_amount = FN_amount + focus_parcel_region.area
            elif invalid_parcel_GT_label == 1:  # TN
                TN_amount = TN_amount + focus_parcel_region.area
        else:
            # 作為"有效"坵塊已經參與前面的計算
            pass

    # ----考慮面積大小的parcel based kappa----
    c_m_with_area_weights = np.array([
        [FN_amount, FP_amount],
        [TN_amount, TP_amount]
    ])
    logger.info("\nconfusion matrix with area weights:\n ")
    logger.info(" ".join([str(x) for x in c_m_with_area_weights.ravel()]))
    # divide each of FN_amount, FP_amount, TN_amount, TP_amount by 10^5 to avoid overflow
    FN_amount/=100000
    FP_amount/=100000
    TN_amount/=100000
    TP_amount/=100000
    
    total_number_of_items = FN_amount + FP_amount + TN_amount + TP_amount

    number_of_items_predicted_as_P = FP_amount + TP_amount
    number_of_items_predicted_as_N = FN_amount + TN_amount
    number_of_items_labeled_as_T = TN_amount + TP_amount
    number_of_items_labeled_as_F = FN_amount + FP_amount
    sigma_GiCi = number_of_items_predicted_as_N * number_of_items_labeled_as_F + \
                 number_of_items_predicted_as_P * number_of_items_labeled_as_T
    up_formula = total_number_of_items * (FN_amount + TP_amount) - sigma_GiCi
    down_formula = total_number_of_items * total_number_of_items - sigma_GiCi
    P_kappa_W = up_formula / down_formula
    logger.info("P-kappa_W:{}".format(P_kappa_W))

    # ----將分類結果轉換成parcel-based語意分割結果----
    if testing_argument.save_prediction == True:
        
        # FILL IN THE PREDICT RESULT WITH LABEL IMAGE
        t1 = time.time()

        # 取得各個坵塊的4連通區域，輸出一個有連通區域標記的影像陣列
        labeled_component_GT_mask, labeled_component_count = label(
            parcel_mask, background=0, return_num=True, connectivity=1)

        parcel_count = 0
        region_list = regionprops(labeled_component_GT_mask)
        # 任何大小的連通區域數量(包含誤標、polygon重疊處)
        logger.info("len(region_list):%d"% len(region_list))
        for region in region_list:  # 遍歷每個連通區域(parcel)
            if region.area > min_valid_parcel_area_size:  # 忽略面積<=min_valid_parcel_area_size的parcel
                focus_parcel_region = region

                # Region邊界外框("真正的行列index") : min_row, min_col, max_row, max_col
                # 由region.bbox取得的行列"下限"，min_row、min_col為"真正的行列index"
                # 由region.bbox取得的行列"上限"，max_row_plus1、max_col_plus1為"真正的行列index"再加一，
                # 方便使用array[min_row:max_row_plus1,
                # min_col:max_col_plus1]取得該region所涵蓋的所有像素
                boundary_box = focus_parcel_region.bbox
                min_row = boundary_box[0]
                max_row_plus1 = boundary_box[2]
                max_row = max_row_plus1 - 1
                min_col = boundary_box[1]
                max_col_plus1 = boundary_box[3]
                max_col = max_col_plus1 - 1
                minimum_parcel_image_boundary = Boundary_box(
                    min_row, max_row, min_col, max_col)

                # 篩選 若目標坵塊過細小=>視為無效坵塊
                # 無視掉某邊低於18個像素寬的坵塊(應該都是polygon重疊導致的錯誤坵塊)
                # 寬高皆為18個以上像素寬才是有效坵塊

                if minimum_parcel_image_boundary.height >= 18 and minimum_parcel_image_boundary.width >= 18:
                    parcel_count = parcel_count + 1  # 目前正在處理第幾個(有效的)parcel

                    # 在labeled_component_GT_mask上的parcel填入model分類結果作為parcel-based語意分割結果
                    # 非水稻邱塊像素為0，水稻邱塊像素
                    # labelimg[labelimg==prop.label] 標記陣列中，像素值等同於目前正在遍歷的連通區域的標記
                    # woflat從0開始對應到第1個有效坵塊
                    if parcel_count - 1 > len(output_confidence_score)-1:labeled_component_GT_mask[labeled_component_GT_mask ==region.label] = 0
                    else : labeled_component_GT_mask[labeled_component_GT_mask ==region.label] = output_confidence_score[parcel_count - 1] * woflat[parcel_count - 1] * 255
            else:
                # 在labeled_component_GT_mask上的parcel填入model分類結果作為parcel-based語意分割結果
                # 面積小於min_valid_parcel_area_size的坵塊視為(誤標)背景
                labeled_component_GT_mask[labeled_component_GT_mask ==
                                          region.label] = 0

        t2 = time.time()
        logger.info('\nfill in result: %f'% (t2 - t1))

        # 保存parcel-based語意分割結果png
        im = Image.fromarray(labeled_component_GT_mask.astype('uint8'))
        # im = im.resize((11460, 12260), Image.NEAREST)
        separate_basename = os.path.basename(parcel_mask_path).split("_")
        im.save(testing_argument.saved_model_result_folder_path +
                "/" + separate_basename[0] + "_" + separate_basename[1])

    parcel_based_kappa = round(parcel_based_kappa, 3)
    parcel_based_kappa_with_area_weights = round(P_kappa_W, 3)
    return frame_number, parcel_based_kappa, parcel_based_kappa_with_area_weights


def test_saved_models(training_inform_obj, arguments, max_val_acc, val_acc_threshold=0.85,
                      Data_root_folder_path=DATA_ROOT_FOLDER_PATH, select_specific_parcels=False,training_parcel_mask_path='data/train_test/parcel_mask'):
    """
    根據model訓練過程的最大的val_acc是否大於val_acc_threshold 決定要不要測試model
    """
    if max_val_acc > val_acc_threshold:
        saved_model_list = [
            training_inform_obj.mc_valacc_path,
            training_inform_obj.mc_path
        ]

        for saved_model_path in saved_model_list:
            if "val_acc" in saved_model_path:
                logger.info("\n== Max val_acc model Testing =================")
            else:
                logger.info("\n== Min val_loss model Testing =================")

            one_round_testing(
                saved_model_path=saved_model_path,
                arguments=arguments, Data_root_folder_path=Data_root_folder_path,
                select_specific_parcels=select_specific_parcels,
                training_parcel_mask_path=training_parcel_mask_path
            )
    else:
        logger.info("max_val_acc < {} => 不做testing".format(val_acc_threshold))


def inference(training_inform_obj, arguments, Data_root_folder_path=DATA_ROOT_FOLDER_PATH,
              saved_model_folder='./data/inference/saved_model_and_prediction',inference_NRG_png_path="./data/inference/NRG_png",
              inference_parcel_mask_path="./data/inference/parcel_mask"):
    logger.info("\n== Max val_acc model inference =================")
    # h5_paths = glob.glob(saved_model_folder + "/*.keras")
    h5_paths = glob.glob(saved_model_folder + "/*.h5")
    one_round_testing(
        saved_model_path=h5_paths[0],
        arguments=arguments, Data_root_folder_path=Data_root_folder_path, select_specific_parcels=False, inference=True,inference_NRG_png_path=inference_NRG_png_path,inference_parcel_mask_path=inference_parcel_mask_path,training_parcel_mask_path=None
    )


def get_parcel_mask_path(
        parcel_masks_folder="./data/train_test/parcel_mask", frame_name=None):
    """
    get the path of parcel mask according to the frame_name

    """
    img_path_list = glob.glob(parcel_masks_folder + "/*.png")

    for img_path in img_path_list:
        if frame_name in img_path:
            return img_path.replace('\\', '/')

    logger.info("*** parcel_mask.png not found")
    sys.exit(-1)


def one_round_testing(
        arguments,
        saved_model_path=None,
        Data_root_folder_path=DATA_ROOT_FOLDER_PATH, select_specific_parcels=False, inference=False,inference_NRG_png_path="./data/inference/NRG_png",inference_parcel_mask_path="./data/inference/parcel_mask",training_parcel_mask_path="./data/train_test/parcel_mask"
):
    # set the path of folder saving the parcel dataset generated from complete
    # frame aerial image

    saved_dataset_root_path = Data_root_folder_path + "/For_training_testing/{}x{}".format(
        arguments.data_shape[0], arguments.data_shape[1])
    if arguments.specified_size_for_testing is not None:
        saved_dataset_root_path = Data_root_folder_path + "/For_training_testing/{}x{}".format(
            arguments.specified_size_for_testing[0], arguments.specified_size_for_testing[1])

    # 載入已保存的model權重
    arguments.model.load_weights(saved_model_path)

    # 測試用的資料集
    arguments.frame_dataset_list = data_manager.load_saved_dataset(
        saved_dataset_root_path, select_specific_parcels)

    logger.info("  *** saved_model_path:%s"% str(saved_model_path))
    saved_model_folder_path = os.path.dirname(saved_model_path)
    saved_model_result_folder_path = saved_model_folder_path + \
                                     "/" + os.path.basename(saved_model_path).split(".h5")[0]
    arguments.saved_model_result_folder_path = saved_model_result_folder_path

    if os.path.isdir(saved_model_result_folder_path):
        logger.info("存放model的資料夾已存在=>刪除，並重建存放資料夾")
        shutil.rmtree(saved_model_result_folder_path)
        os.mkdir(saved_model_result_folder_path)
    else:
        os.mkdir(saved_model_result_folder_path)

    if inference:
        # find all frame_nums in folder
        # _all_frame_nums = glob.glob(inference_NRG_png_path+'/*.png')
        _all_frame_nums = glob.glob(saved_dataset_root_path+'/parcel_NIRRGA/*')
        if len(_all_frame_nums) == 0:
            debug_logger.error("no .png to be inferenced found in folder")
            raise Exception("no .png to be inferenced found in folder")
        # _all_frame_nums = sorted([os.path.basename(f).split(
        #     '.')[0] for f in _all_frame_nums])
        _all_frame_nums = sorted([os.path.basename(f) for f in _all_frame_nums])

        # i = 
        print('all frame:', _all_frame_nums)
        for _frame_num in _all_frame_nums:
            print('inference frame:', _frame_num)
            inference_on_one_frame(
                testing_argument=arguments, Data_root_folder_path=Data_root_folder_path, frame_name=_frame_num,
                frame_codename=_frame_num,inference_parcel_mask_path=inference_parcel_mask_path)
            # i += 1
    else:
        # if arguments.round_number == 1:
        Round_testing(arguments, Data_root_folder_path,
                      select_specific_parcels,training_parcel_mask_path)


# +


class Arguments:
    def __init__(
            self,
            data_shape,
            round_number,
            BATCH_SIZE,
            split_up_fourth_channel,
            masking,
            normalizing,
            per_image_standardization,
            preprocess_input,
            model=None,
            saved_model_result_folder_path=None,
            frame_dataset_list=None,
            save_prediction=False,
            random_brightness_max_delta=0,
            test_only=False,
            inference=False,
            specified_size_for_testing=None,
            resize_half=False):
        self.data_shape = data_shape
        self.round_number = round_number
        self.BATCH_SIZE = BATCH_SIZE
        self.split_up_fourth_channel = split_up_fourth_channel
        self.masking = masking
        self.normalizing = normalizing
        self.per_image_standardization = per_image_standardization
        self.preprocess_input = preprocess_input
        self.random_brightness_max_delta = random_brightness_max_delta
        self.test_only = test_only
        self.inference = inference
        self.specified_size_for_testing = specified_size_for_testing  # (H,W)
        self.resize_half = resize_half

        if self.per_image_standardization == True:
            self.normalizing = False  # 不多做一次

        self.save_prediction = save_prediction

        self.model = model
        self.saved_model_result_folder_path = saved_model_result_folder_path
        self.frame_dataset_list = frame_dataset_list


def inference_on_one_frame(testing_argument, Data_root_folder_path,
                           frame_name='94191004_181006z', frame_codename=0,inference_parcel_mask_path='data/inference/parcel_mask'):
    parcel_based_kappa_list = []
    _path_to_parcels_taken_hashmap = os.path.join(Data_root_folder_path, 'For_training_testing', '%dx%d' % (
        testing_argument.data_shape[0], testing_argument.data_shape[1]), 'parcel_NIRRGA', str(frame_codename),
                                                  'parcels_that_are_actually_taken.txt')
    frame_number, parcel_based_kappa, parcel_based_kappa_with_area_weights = model_testing(
        testing_argument,
        testing_ds_frame_codename=frame_codename,
        frame_name=frame_name,
        min_valid_parcel_area_size=0,
        path_to_parcels_taken_hashmap=_path_to_parcels_taken_hashmap

    )
    parcel_based_kappa_list.append(
        (frame_number, parcel_based_kappa, parcel_based_kappa_with_area_weights))

    total_parcel_based_kappa = 0
    total_parcel_based_kappa_with_area_weights = 0

    for ele in parcel_based_kappa_list:  # 圖框號
        logger.info("{}".format(ele[0]))

    for ele in parcel_based_kappa_list:  # parcel_based kappa
        logger.info("{}".format(ele[1]))
        total_parcel_based_kappa = total_parcel_based_kappa + ele[1]

    logger.info("Avg. P-kappa: %f"%round(total_parcel_based_kappa / len(parcel_based_kappa_list), 3))

  

    for ele in parcel_based_kappa_list:  # parcel_based kappa with_area_weights
        logger.info("%f"%ele[2])
        total_parcel_based_kappa_with_area_weights = total_parcel_based_kappa_with_area_weights + \
                                                     ele[2]

    logger.info("Avg. P-kappa_W: {}".format(
        round(total_parcel_based_kappa_with_area_weights /
              len(parcel_based_kappa_list), 3)
    ))


def Round_testing(testing_argument, Data_root_folder_path, select_specific_parcels,training_parcel_mask_path):
    parcel_based_kappa_list = []

    if select_specific_parcels:
        _parcel_root_dir = 'parcel_NIRRGA_backup'
    else:
        _parcel_root_dir = 'parcel_NIRRGA'

    # add your frames here
    # frame 0: (94201093_181104z)
    testing_ds_frame_codename = 0
    frame_name = "94191004_181006z"

    _path_to_parcels_taken_hashmap = os.path.join(Data_root_folder_path, 'For_training_testing',
                                                  '%dx%d' % (
                                                      testing_argument.data_shape[0], testing_argument.data_shape[1]),
                                                  _parcel_root_dir, str(testing_ds_frame_codename),
                                                  'parcels_that_are_actually_taken.txt')

    frame_number, parcel_based_kappa, parcel_based_kappa_with_area_weights = model_testing(
        testing_argument,
        testing_ds_frame_codename=testing_ds_frame_codename,
        parcel_mask_path=get_parcel_mask_path(
            frame_name=frame_name, parcel_masks_folder=training_parcel_mask_path),
        path_to_parcels_taken_hashmap=_path_to_parcels_taken_hashmap
    )
    parcel_based_kappa_list.append(
        (frame_number, parcel_based_kappa, parcel_based_kappa_with_area_weights))

    total_parcel_based_kappa = 0
    total_parcel_based_kappa_with_area_weights = 0

    for ele in parcel_based_kappa_list:  # 圖框號
        logger.info("{}".format(ele[0]))

    for ele in parcel_based_kappa_list:  # parcel_based kappa
        logger.info("{}".format(ele[1]))
        total_parcel_based_kappa = total_parcel_based_kappa + ele[1]

    logger.info("Avg. P-kappa: {}".format(
        round(total_parcel_based_kappa / len(parcel_based_kappa_list), 3)
    ))

    logger.info("")

    for ele in parcel_based_kappa_list:  # parcel_based kappa with_area_weights
        logger.info("{}".format(ele[2]))
        total_parcel_based_kappa_with_area_weights = total_parcel_based_kappa_with_area_weights + \
                                                     ele[2]

    logger.info("Avg. P-kappa_W: {}".format(
        round(total_parcel_based_kappa_with_area_weights /
              len(parcel_based_kappa_list), 3)
    ))
