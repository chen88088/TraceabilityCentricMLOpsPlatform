"""Generating parcel """
import math
import os
import glob
import shutil
import sys
from logging import config as log_config
from skimage.measure import label, regionprops
import logging
import numpy as np
from configs.logger_config import LOGGING_CONFIG
from PIL import Image

log_config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("parcel_info_log")

PARCEL_NIRRGA_PATH = "/parcel_NIRRGA/"
Image.MAX_IMAGE_PIXELS = None
PARCEL_NIRRGA_PATH = "/parcel_NIRRGA/"

class DataManager:
    """
    load_saved_dataset(dataset_root_path):
        載入已經過前處理產生的frame資料集 e.g. parcel image的資訊

    """

    def __init__(self):
        pass  # empty cause no need to setup local variable

    def sort_parcel_image_list_by_number_in_file_path(self, filepath_list: list):
        """
        以坵塊的編號排序list中的坵塊影像npy檔案路徑

        list中的元素為坵塊影像npy檔案的路徑
        e.g. ".\\Data\\For training testing\\Calibrated 70x70\\parcel_NIRRGA\0\f0_1_parcel_NIRRGA_area 9737_GT 1.npy"
            "95213081_O230502a_21_hr4_moa_101120009.npy"
        """
        # 以os.path.basename(file_path).split("_")[1] 得到坵塊的編號,
        # 並以坵塊的編號(int)作為排序的依據
        filepath_list.sort(key=lambda file_path: int(
            os.path.basename(file_path).split("_")[-1].split('.')[0]))
        return filepath_list

    def sort_file_path_list_by_number_in_file_path(self, filepath_list: list):
        """
        以os.path.basename(file_path).split("_")[0] 得到檔名的編號, 並以檔名的編號(int)作為排序的依據
        """
        # logger.info("  sort_file_path_list_by_number_in_file_path(list)  list: %s", filepath_list)
        if len(filepath_list) != 0:
            if "DONE_FIN_" in os.path.basename(filepath_list[0]):  # 校正後
                filepath_list.sort(key=lambda file_path: int(
                    os.path.basename(file_path).split("_")[2]))

            else:
                filepath_list.sort(key=lambda file_path: int(
                    os.path.basename(file_path).split("_")[0]))
        return filepath_list

    def __get_npy_file_path_in_folder(self, folder_path):
        """
        取得資料夾內所有的npy檔案路徑list, 並根據坵塊編號排序
        """
        
        file_path_list = glob.glob(folder_path + "/*.npy")
        file_path_list = self.sort_parcel_image_list_by_number_in_file_path(file_path_list)
        if len(file_path_list) == 0:
            raise Exception("no .npy files found!!! check the dataset!!!")
        return file_path_list

    def get_all_file_path_in_folder(self, folder_path):
        """
        取得資料夾內所有的檔案路徑list
        """
        file_path_list = glob.glob(folder_path + r"/*")

        logger.info("folder_path: %s",folder_path)
        logger.info("file_path_list: %s",file_path_list)
        file_path_list = self.sort_file_path_list_by_number_in_file_path(file_path_list)
        return file_path_list

    def get_png_file_path_in_folder(self, folder_path):
        """
        取得資料夾內.png結尾的檔案路徑list
        """
        file_path_list = glob.glob(folder_path + "/*.png")
        file_path_list = self.sort_file_path_list_by_number_in_file_path(file_path_list)
        return file_path_list

    def __read_GT_label_txt(self, GT_label_txt_path):
        """
        讀入存放某一frame中所有parcel對應的GT label的txt檔, 將GT label存放於list
        """
        GT_label_list = []
        with open(GT_label_txt_path, "r") as f:
            for line in f.readlines():
                GT_label_list.append(int(line.strip()))
        return GT_label_list

    def load_saved_dataset(self, dataset_root_path, select_specific_parcels=False):
        """
        讀取已經過資料前處理的資料集, 每個frame的parcel_NIRRG, parcel_GTmask, parcel_GT_label
        以dictionary型態儲存dataset, 回傳frame_dataset_list

        dict_n:
            {
                "frame_codename": frame_codename,
                "parcel_NIRRGA_saving_dirname": parcel_NIRRGA_saving_dirname,
                "parcel_NIRRGA_path_list": parcel_NIRRGA_path_list,
                "parcel_GT_label_list": parcel_GT_label_list
            }

        frame_dataset_list:
            [dict_1, dict_2, ...]

        dataset_root_path:
            存放parcel_NIRRG資料夾的資料夾路徑
        """
        frame_dataset_list = []
        if select_specific_parcels:
            parcel_NIRRGA_root_folder_path = dataset_root_path + "/parcel_NIRRGA_backup"
            # backuped original folder for testing
        else:
            parcel_NIRRGA_root_folder_path = dataset_root_path + "/parcel_NIRRGA"  # for training or infernence
        parcel_NIRRGA_folder_path_list = self.get_all_file_path_in_folder(
            folder_path=parcel_NIRRGA_root_folder_path)

        for parcel_NIRRGA_folder_path in parcel_NIRRGA_folder_path_list:
            dic = {}  # dictionary for temp
            dic["frame_codename"] = os.path.basename(parcel_NIRRGA_folder_path)
            dic["parcel_NIRRGA_saving_dirname"] = parcel_NIRRGA_folder_path
            dic["parcel_NIRRGA_path_list"] = self.__get_npy_file_path_in_folder(
                folder_path=parcel_NIRRGA_folder_path)
            GT_label_txt_path = glob.glob(
                parcel_NIRRGA_folder_path + "/*label.txt")[0]
            dic["parcel_GT_label_list"] = self.__read_GT_label_txt(
                GT_label_txt_path)
            frame_dataset_list.append(dic)

        return frame_dataset_list


def generate_int_arange_list(start_val, end_val):
    """
    產生元素為 start_val~end_val-1 之間整數的list
    [0 ,1 2, 3, ...]
    """
    nparray = np.arange(start_val, end_val)
    nparray_to_list = nparray.tolist()

    return nparray_to_list


class Data_preprocessing:
    def __init__(
            self,
            dataset_root_folder_path,
            NIRRG_folder_path,
            GTmask_folder_path,
            saved_image_type,
            fixed_shape=(None, None),
            target_frame_codename=(None, None),
            rare_case_annotation_mask_folder=None,
            select_specific_parcels=False,
            inference=False
    ):
        """
        dataset_root_folder_path:
            作為存放此次產生的坵塊資料的根目錄路徑

        NIRRG_folder_path:
            存放NIRRG png image的資料夾路徑

        GTmask_folder_path:
            存放annotation image(parcel mask)的資料夾路徑

        saved_image_type:
            要產生的坵塊資料是什麼型態的檔案 => png 或 npy

        fixed_shape:
            要產生的坵塊資料的寬高 => (高, 寬)

        target_frame_codename:
            (首index, 末index), 對frame_codename介於 首index~末index-1 的frame進行資料前處理(若不指定則為0~25全做)
        """
        data_manager = DataManager()
        self.inference = inference
        logger.info("NIRRG_folder_path: %s", NIRRG_folder_path)
        self.dataset_root_folder_path = dataset_root_folder_path
        self.NIRRG_path_list = data_manager.get_png_file_path_in_folder(
            folder_path=NIRRG_folder_path)
        logger.info("self.NIRRG_path_list: %s", " ".join(self.NIRRG_path_list))

        if self.NIRRG_path_list == []:
            file_path_list = glob.glob(NIRRG_folder_path + "/*.tif")
            self.NIRRG_path_list = data_manager.sort_file_path_list_by_number_in_file_path(file_path_list)

        self.GTmask_path_list = data_manager.get_png_file_path_in_folder(
            GTmask_folder_path)  # 使用的GTmask是將polygon和polyline疊加而產生的
        self.saved_image_type = saved_image_type

        # 設定要處理第幾個frame_codename對應的frame image
        self.frame_codename_list_for_data_preprocessing = generate_int_arange_list(target_frame_codename[0],
                                                                                   target_frame_codename[1])

        self.fixed_shape = fixed_shape

        self.rare_case_annotation_mask_folder = rare_case_annotation_mask_folder

        self.select_specific_parcels = select_specific_parcels

    def start_preprocessing(self, in_shape):
        """
        對指定的frame進行資料前處理, 產生parcel image等資料紀錄在frame_dataset_list並回傳
        """
        frame_dataset_list = []
        rare_case_regions = []
        logger.info("self.GTmask_path_list: %s", " ".join(self.GTmask_path_list))
        frame_codename = 0  # 控制目前處理第幾張frame(從0開始)
        for NIRRG_path, GTmask_path in zip(
                self.NIRRG_path_list, self.GTmask_path_list):
            if frame_codename in self.frame_codename_list_for_data_preprocessing:
                logger.info("\n-----------------------")
                logger.info("frame_codename: %s  ; NIRRG_path: %s ;  GTmask_path: %s",
                            frame_codename, NIRRG_path, GTmask_path)
                logger.info("self.rare_case_annotation_mask_folder  : %s",
                            self.rare_case_annotation_mask_folder)
                preprocessing_frame = Preprocessing_per_frame(
                    self.dataset_root_folder_path,
                    NIRRG_path,
                    GTmask_path,
                    frame_codename,
                    self.fixed_shape,
                    self.saved_image_type,
                    based_on_framelet_dataset=False,
                    parcel_count_for_framelet_dataset=0,
                    rare_case_annotation_mask_folder=self.rare_case_annotation_mask_folder,
                    select_specific_parcels=self.select_specific_parcels,
                    inference=self.inference,
                    in_shape =in_shape
                )

                rare_case_regions.append(preprocessing_frame.rare_case_regions)
                frame_dataset_dic = preprocessing_frame.get_frame_dataset_dic()
                frame_dataset_list.append(frame_dataset_dic)

            frame_codename = frame_codename + 1

        return frame_dataset_list, rare_case_regions

    def run_select_specific_parcels(self, frame_code):
        GT_parcel_label_txt_path = self.dataset_root_folder_path + \
                                   '/parcel_NIRRGA_selected_parcels/%d/%d_GT_label.txt' % (
                                       frame_code, frame_code)
        GT_rice_label_txt_path = self.dataset_root_folder_path + \
                                 '/parcel_NIRRGA/%d/%d_GT_label.txt' % (frame_code, frame_code)

        GT_rice_label_list = []
        GT_parcel_label_list = []
        rice_directory = dict()

        with open(GT_rice_label_txt_path, "r") as f:
            for line in f.readlines():
                GT_rice_label_list.append(int(line.strip()))
        with open(GT_parcel_label_txt_path, "r") as f:
            for line in f.readlines():
                GT_parcel_label_list.append(int(line.strip()))
        if len(GT_parcel_label_list)>len(GT_rice_label_list):
            logger.info("len(GT_parcel_label_list)>len(GT_rice_label_list), indicating every parcel was selected. We will skip the rest of run_select_specific_parcels()")
            return
        files = glob.glob(self.dataset_root_folder_path +
                          "/parcel_NIRRGA/%d/*.npy" % frame_code)
        for file in files:
            index = int(os.path.basename(file).split('_')[1])
            rice_directory[index] = os.path.basename(file)

        t = 1
        for index in GT_parcel_label_list:
            if index == 0:
                os.remove(self.dataset_root_folder_path +
                          PARCEL_NIRRGA_PATH + "%d/" % (frame_code) + rice_directory[t])
                del rice_directory[t]
            t += 1

        rice_directory_index = rice_directory
        rice_directory_index = sorted(rice_directory_index)
        i = 1
        os.remove(GT_rice_label_txt_path)
        f = open(GT_rice_label_txt_path, 'w')

        for index in rice_directory_index:
            converted_num = str(i)
            a = os.path.basename(rice_directory[index]).split(' ')[0]
            new_name = os.path.basename(a).split('_')[0] + "_" + converted_num + "_" + \
                       os.path.basename(a).split('_')[2] + "_" + os.path.basename(a).split('_')[3] + \
                       "_" + os.path.basename(a).split('_')[4] + " " + \
                       os.path.basename(rice_directory[index]).split(' ')[1] \
                       + " " + \
                       os.path.basename(rice_directory[index]).split(' ')[2]
            os.rename(self.dataset_root_folder_path + PARCEL_NIRRGA_PATH + "%d/" % (frame_code) + rice_directory[index],
                      self.dataset_root_folder_path + PARCEL_NIRRGA_PATH + "%d/" % (frame_code) + new_name)
            f.write(os.path.basename(rice_directory[index]).split(' ')[2][0])
            f.write("\n")

            i += 1

        f.close()

    def cleanup(self):
        shutil.rmtree(self.dataset_root_folder_path +
                      '/parcel_NIRRGA_selected_parcels')


def package_boundary_information(region_bbox):
    boundary_box = region_bbox
    min_row = boundary_box[0]
    max_row_plus1 = boundary_box[2]
    max_row = max_row_plus1 - 1
    min_col = boundary_box[1]
    max_col_plus1 = boundary_box[3]
    max_col = max_col_plus1 - 1
    parcel_image_boundary = BoundaryBox(min_row, max_row, min_col, max_col)
    return parcel_image_boundary


class Preprocessing_per_frame:
    def __init__(self, dataset_root_folder_path, NIRRG_path, GTmask_path,
                 frame_codename, fixed_shape, saved_image_type,
                 based_on_framelet_dataset=False, parcel_count_for_framelet_dataset=0,
                 rare_case_annotation_mask_folder=None,
                 select_specific_parcels=False,
                 inference=False,
                 in_shape=None
                 ):
        """
        使用的parcel mask具有水稻GT label
        將來實際應用時使用的parcel mask不具有水稻GT label, 只有標示每塊parcel

        based_on_framelet_dataset:
            True => 與1.3 1.4使用相同的framelet dataset
            (隨機抽樣10800張Framelet, 取前2160張當作validataion dataset)

        """
        self.dataset_root_folder_path = dataset_root_folder_path
        self.whole_frame_NIRRG = np.array(Image.open(NIRRG_path))
        self.whole_frame_parcel_GT_mask = np.array(Image.open(GTmask_path))
        self.frame_codename = frame_codename
        self.fixed_shape = fixed_shape
        image_basename_split = os.path.basename(NIRRG_path).split("_")
        self.frameNumber_and_shotDate = image_basename_split[0] + "_" + image_basename_split[1]
        self.based_on_framelet_dataset = based_on_framelet_dataset
        self.select_specific_parcels = select_specific_parcels
        # 若要從framelet擷取出parcel
        # image，用此對所有framelet中的parcel編號(不同的framelet內的parcel編號是通用連續的)
        self.parcel_count_for_framelet_dataset = parcel_count_for_framelet_dataset
        self.rare_case_annotation_mask_folder = rare_case_annotation_mask_folder

        # 裁切NIRRG、GTmask
        """
        要產生固定大小的影像時，才需考慮影像中心是連通區域的中心或是連通區域的質心???
        """

        """
        1019
        將crop_all_processing和crop_all_processing_on_framelet合併
        根據self.based_on_framelet_dataset做不同的處理需求
        """
        # if self.based_on_framelet_dataset == False: # 從完整航攝影像擷取坵塊影像
        #     # 若不取固定大小，取每個坵塊的bounding box
        #     parcel_NIRRGA_saving_dirname, parcel_NIRRGA_path_list, parcel_GT_label_list,
        #     parcel_size_list, rare_case_regions = self.crop_all_processing(saved_image_type=saved_image_type)
        #     self.rare_case_regions = rare_case_regions
        # else: # 從framelet影像擷取坵塊影像
        #     if fixed_shape != None:
        #         parcel_NIRRGA_saving_dirname, parcel_NIRRGA_path_list, parcel_GT_label_list,
        #         parcel_size_list = self.crop_all_processing_on_framelet(saved_image_type=saved_image_type)
        #     else:
        #         logger.info("fixed_shape需指定為小於framelet大小的坵塊影像大小")
        #         sys.exit(1)
        return_list = self.crop_all_processing(saved_image_type=saved_image_type, inference=inference, in_shape=in_shape)
        parcel_NIRRGA_saving_dirname = return_list[0]
        parcel_NIRRGA_path_list = return_list[1]
        parcel_GT_label_list = return_list[2]
        parcel_size_list = return_list[3]
        rare_case_regions = return_list[4]

        self.rare_case_regions = rare_case_regions

        # parcel_GTmask_path為parcel的annotation mask路徑
        # parcel_GT_label為parcel是否為水稻的label
        self.dic = {
            "frame_codename": frame_codename,
            "parcel_NIRRGA_saving_dirname": parcel_NIRRGA_saving_dirname,
            "parcel_NIRRGA_path_list": parcel_NIRRGA_path_list,
            "parcel_GT_label_list": parcel_GT_label_list,
            "parcel_size_list": parcel_size_list
        }

    def get_self_parcel_count_for_framelet_dataset(self):
        return self.parcel_count_for_framelet_dataset

    def get_frame_dataset_dic(self):
        return self.dic

    def get_GT_label_of_focus_parcel(self, focus_parcel_region):
        """
        焦點坵塊是否為水稻的label, 1 or 0
        """
        # focus_parcel_region.coords 焦點坵塊內像素 在原圖上的點座標
        # 1=non-rice, 2=rice
        parcel_GT_3class_label = self.whole_frame_parcel_GT_mask[
            focus_parcel_region.coords[0][0], focus_parcel_region.coords[0][1]]
        if parcel_GT_3class_label == 2:  # 該坵塊為水稻坵塊
            parcel_GT_label = 1
        else:
            parcel_GT_label = 0

        return parcel_GT_label

    def __save_GT_labels_to_txt(self, parcel_GT_label_list):
        """
        將每個parcel的GT label記錄到txt, 將來要使用時可載入
        """

        GT_label_txt_path = self.saving_dirname + r'/{}_GT_label.txt'.format(self.frame_codename)
        with open(GT_label_txt_path, 'w') as f:
            # logger.info(" 寫檔 len(parcel_GT_label_list):",len(parcel_GT_label_list))
            for s in parcel_GT_label_list:
                f.write(str(s) + '\n')

    def __save_GT_labels_to_txt_for_framelet_dataset(
            self, parcel_GT_label_list):
        """
        將每個parcel的GT label記錄到txt, 將來要使用時可載入
        """

        GT_label_txt_path = self.saving_dirname + r'/GT_label.txt'
        with open(GT_label_txt_path, 'a') as f:
            # logger.info(" 寫檔 len(parcel_GT_label_list):",len(parcel_GT_label_list))
            for s in parcel_GT_label_list:
                f.write(str(s) + '\n')

    def take_out_parcel_image(self, minimum_width, minimum_height,
                              parcel_image_min_col, parcel_image_max_col, parcel_image_min_row, parcel_image_max_row):

        MAX_col = 11460 - 1  # 最右側的行 index =>最右邊的行是第MAX_col行
        MAX_row = 12260 - 1  # 最下側的列 index =>最下面的列是第MAX_row列
        new_parcel_image_min_col, new_parcel_image_max_col = self.cal_new_boundary_of_RowOrColumn(
            minimum_width, parcel_image_min_col, parcel_image_max_col, aerial_image_MAX=MAX_col)
        new_parcel_image_min_row, new_parcel_image_max_row = self.cal_new_boundary_of_RowOrColumn(
            minimum_height, parcel_image_min_row, parcel_image_max_row, aerial_image_MAX=MAX_row)

        return new_parcel_image_min_col, new_parcel_image_max_col, new_parcel_image_min_row, new_parcel_image_max_row

    def cal_new_boundary_of_RowOrColumn(self, minimum_length,
                                        parcel_image_min, parcel_image_max,
                                        aerial_image_MAX, aerial_image_MIN=0):
        """
        以像素數量計算長度
        若某邊長度小於minimum_length, 擷取minimum_length的長度
        若某邊長度大於等於minimum_length, 按照該長度擷取(>=minimum_length)


        一次只用於計算其中一個軸, 計算X軸或Y軸, 一個軸上的parcel新邊界座標值
        假設minimum_length=32
        offset:
            parcel image的某個邊與minimum_length的長度差距
        offset_half:
            將parcel image在同一個軸上的最小、最大邊界向外移動的位移量
            若offset是偶數, offset_half會是offset//2, 最終擷取出來的長度會是32
            若offset是奇數, offset_half會是offset//2+1, 最終擷取出來的長度會是34
        """
        odd_offset = False
        parcel_image_length = parcel_image_max - parcel_image_min + 1  # 像素數量為間隔+1
        offset = minimum_length - parcel_image_length
        if offset > 0:  # 長度小於minimum_length 需要擷取較大的parcel image
            if offset % 2 == 0:  # 偶數
                offset_half = offset // 2
            else:
                offset_half = offset // 2 + 1  # 位移+1
                odd_offset = True
            new_min = parcel_image_min - offset_half
            new_max = parcel_image_max + offset_half
            # logger.info("offset",offset)
            # logger.info("offset_half",offset_half)
            # logger.info("parcel_image_min:{}, parcel_image_max:{}".format(parcel_image_min, parcel_image_max))
            # logger.info("new_min:{}, new_max:{}".format(new_min, new_max))

            # check 是否超出邊界
            if new_min < aerial_image_MIN:  # 超出最左/上的邊界=>右/下移 超出的像素長度
                over_offset = aerial_image_MIN - new_min
                new_min = new_min + over_offset
                new_max = new_max + over_offset
            if new_max > aerial_image_MAX:  # 超出最右/下的邊界=>左/上移 超出的像素長度
                over_offset = new_max - aerial_image_MAX
                new_min = new_min - over_offset
                new_max = new_max - over_offset

            if odd_offset is True:
                # logger.info("odd_offset == True")
                new_max = new_max - 1

            return new_min, new_max
        else:  # 長度沒有小於minimum_length 不需要擷取較大的parcel image
            if odd_offset is True:
                # logger.info("odd_offset == True")
                new_max = new_max - 1 # weird new_max since it never been assigned

            return parcel_image_min, parcel_image_max

    @staticmethod
    def calculate_pixel_ratio_of_focus_parcel_in_boundary_box(
            focus_parcel_region,
            labeled_connected_component_GT_mask,
            boundary_box):

        # 擷取焦點坵塊"固定大小"的annotation image, 也就是焦點坵塊的GT mask
        focus_parcel_image_annotation_mask = (
                labeled_connected_component_GT_mask[boundary_box.min_row:boundary_box.max_row_plus,
                boundary_box.min_col:boundary_box.max_col_plus] == focus_parcel_region.label)
        # GT mask上, 像素的像素值為True代表該像素屬於焦點坵塊內的像素
        pixel_count_of_focus_parcel = np.count_nonzero(
            focus_parcel_image_annotation_mask == 1)
        
        if focus_parcel_image_annotation_mask.size == 0:
            logger.info("   ----calculate_pixel_ratio_of_focus_parcel_in_boundary_box()----")
            logger.info("boundary_box.min_row :  %s, boundary_box.max_row: %s",
                        str(boundary_box.min_row), str(boundary_box.max_row))
            logger.info("boundary_box.min_col : %s, boundary_box.max_col: %s",
                        str(boundary_box.min_col), str(boundary_box.max_col))
            logger.info("pixel_count_of_focus_parcel: %s", str(pixel_count_of_focus_parcel))
            logger.info("focus_parcel_image_annotation_mask.size: %s",
                        " ".join(focus_parcel_image_annotation_mask.size))

        pixel_ratio_of_focus_parcel_in_boundary_box = pixel_count_of_focus_parcel / \
                                                      focus_parcel_image_annotation_mask.size
        return pixel_ratio_of_focus_parcel_in_boundary_box

    @staticmethod
    def calculate_pixel_ratio_of_focus_parcel_in_scanned_boundary_box(
            labeled_component_GT_mask,
            focus_parcel_region,
            minimum_parcel_image_boundary,
            fixed_height,
            fixed_width):
        """

        """
        # 擷取此焦點坵塊之 "minimum parcel image大小" 的mask，焦點坵塊的GT
        # mask，像素值為True代表該像素屬於焦點坵塊內的像素
        minimum_focus_parcel_image_mask = (
                labeled_component_GT_mask[
                minimum_parcel_image_boundary.min_row:minimum_parcel_image_boundary.max_row_plus,
                minimum_parcel_image_boundary.min_col:minimum_parcel_image_boundary.max_col_plus
                ] == focus_parcel_region.label
        )  # True=parcel area

        # Window scan
        # 在minimum_focus_parcel_image_mask中scan比起在整張航照中scan更快
        # 但傳出的座標值就會是對應minimum_focus_parcel_image_mask的座標值，而不是整張航照中的座標值
        # =>需要轉換成對應整張航照的座標值
        min_col, max_col, min_row, max_row = window_scan_for_searching_focus_parcel(
            image=minimum_focus_parcel_image_mask,
            window_height=fixed_height, window_width=fixed_width, target_value=1
        )

        # 轉換成對應整張航照的座標值
        min_col = min_col + minimum_parcel_image_boundary.min_col
        max_col = max_col + minimum_parcel_image_boundary.min_col
        min_row = min_row + minimum_parcel_image_boundary.min_row
        max_row = max_row + minimum_parcel_image_boundary.min_row
        logger.info("min_row: %s max_row: %s", str(min_row),str( max_row))
        logger.info("min_col: %s, max_col: %s", str(min_col), str(max_col))

        logger.info("\t -----calculate_pixel_ratio_of_focus_parcel_in_scanned_boundary_box")
        scanned_parcel_image_boundary = BoundaryBox(
            min_row, max_row, min_col, max_col)
        pixel_ratio_of_focus_parcel_in_scanned_boundary_box = \
            Preprocessing_per_frame.calculate_pixel_ratio_of_focus_parcel_in_boundary_box(
                focus_parcel_region,
                labeled_connected_component_GT_mask=labeled_component_GT_mask,
                boundary_box=scanned_parcel_image_boundary
            )

        return pixel_ratio_of_focus_parcel_in_scanned_boundary_box, scanned_parcel_image_boundary

    def rare_case_processing(self,rare_case_annotation_mask_path_list):
        for rare_case_annotation_mask_path in rare_case_annotation_mask_path_list:
            if self.frameNumber_and_shotDate in rare_case_annotation_mask_path:
                logger.info("  rare_case_annotation_mask_path: %s",
                            rare_case_annotation_mask_path)
                rare_case_annotation_mask = np.array(Image.open(
                    rare_case_annotation_mask_path).convert("L"))
                val, count = np.unique(
                    rare_case_annotation_mask, return_counts=True)
                logger.info(
                    "rare_case_annotation_mask  val %s, count %s", str(val), str(count))
                # regard "not rare care" and "non-farmland background" as non-farmland background(pixel value 0)
                # pixel value 0: not rare care, pixel value 15:
                # non-farmland background
                rare_case_annotation_mask[np.where(
                    rare_case_annotation_mask == 15)] = 0
                val, count = np.unique(
                    rare_case_annotation_mask, return_counts=True)
                logger.info(
                    "change 15 to 0,  rare_case_annotation_mask  val %s , count %s", str(val), str(count))

                rare_case_labeled_component_GT_mask, rare_case_labeled_component_count = label(
                    rare_case_annotation_mask, background=0, return_num=True, connectivity=1)
                # rare_case_parcel_count = 0
                rare_case_region_list = regionprops(
                    rare_case_labeled_component_GT_mask)
                logger.info("  len(rare_case_region_list): %s",
                            str(len(rare_case_region_list)))
                for rare_case_region in rare_case_region_list:
                    # region : 透過regionprops取得的各個連通區域物件 == 每個parcel的資訊
                    if rare_case_region.area > min_valid_parcel_area_size:
                        target_rare_case_parcel_region = rare_case_region
                        boundary_box = target_rare_case_parcel_region.bbox
                        min_row = boundary_box[0]
                        max_row_plus1 = boundary_box[2]
                        max_row = max_row_plus1 - 1
                        min_col = boundary_box[1]
                        max_col_plus1 = boundary_box[3]
                        max_col = max_col_plus1 - 1
                        minimum_parcel_image_boundary = BoundaryBox(
                            min_row, max_row, min_col, max_col)
                        if minimum_parcel_image_boundary.height >= 18 and \
                                minimum_parcel_image_boundary.width >= 18:
                            # rare_case_parcel_count =
                            # rare_case_parcel_count + 1 #
                            # 目前正在處理第幾個(有效的)parcel
                            rare_case_region_centroid = (
                                int(rare_case_region.centroid[0]), int(rare_case_region.centroid[1]))
                            one_pixel_loc_in_region = (
                                rare_case_region.coords[0][0], rare_case_region.coords[0][1])
                            rare_case_label = rare_case_annotation_mask[one_pixel_loc_in_region]
                            rare_case_regions.append(
                                RareCaseRegion(
                                    rare_case_region_centroid, rare_case=rare_case_label, parcel_count=None))

                            """
                            用質心座標來檢查是否是同一個坵塊

                            要使用polyline和polygon的疊合產生的parcel mask和rare case annotation mask
                            # rare_case_annotation_mask上的坵塊面積應該會是>=parcel mask上的坵塊面積
                            # (因為沒有去除邊緣像素) 若parcel mask上的坵塊的像素座標包含在rare_case_annotation_mask
                            # 上的坵塊像素座標中, 就代表是同一個坵塊
                            """
                            # sys.exit(1)
        logger.info("len(rare_case_regions): %s", str(len(rare_case_regions)))
    def crop_all_processing(self, saved_image_type="npy", inference=False, in_shape = None):
        """
        saved_image_type:
            "png" or "npy"

        self.whole_frame_NIRRG:

        self.whole_frame_parcel_GT_mask :
            完整frame大小的annnotation image, 具有坵塊資訊,
            用於將NIRRG image和Annotation image裁切成一個一個的parcel image(NIRRG、GT mask)
        whole_frame_parcel_GT_mask:
            完整frame大小的GT mask(已經結合polygon和polyline), 每個坵塊為分開不相連的, 用於透過label()取得
            每個坵塊的區域資訊
        rare_case_annotation_mask_path:
            使用ArcGIS以"rare_case"欄位數值輸出的image, 原本具有polygon的像素在image上的像素數值等同於rare_case的數值

        """
        logger.info("     -----crop_all_processing()-----")

        # ----   ----
        if inference:
            parcel_count = 0
            min_valid_parcel_area_size = 0
        elif self.based_on_framelet_dataset is False:
            # 每個frame中坵塊對應的坵塊編號parcel count不是唯一的, 每當為不同frame產生坵塊資料時從0開始編號
            parcel_count = 0  # 初始化
            min_valid_parcel_area_size = 7000
        else:
            # 每個framelet中坵塊對應的坵塊編號parcel count為唯一的, 不會從0重新編號
            parcel_count = self.parcel_count_for_framelet_dataset  # 接續先前完成的坵塊編號
            min_valid_parcel_area_size = 2000

        # ----Rare case???----  應該另外建一個function
        # Rare case的坵塊影像只能從完整圖幅產生坵塊資料
        rare_case_regions = None
        if self.based_on_framelet_dataset is False:
            rare_case_regions = []  # 初始化
            if self.rare_case_annotation_mask_folder is not None:
                logger.info("\n *** self.rare_case_annotation_mask_folder != None ***")
                rare_case_annotation_mask_path_list = glob.glob(
                    self.rare_case_annotation_mask_folder + "/*.png")
                self.rare_case_processing(rare_case_annotation_mask_path_list)


        parcel_size_list = []
        parcel_image_path_list = []
        parcel_GT_label_list = []
        self.labeled_component_GT_mask, labeled_component_count = label(
            self.whole_frame_parcel_GT_mask, background=0, return_num=True, connectivity=2)

        if self.fixed_shape is not None:
            fixed_height = self.fixed_shape[0]
            fixed_width = self.fixed_shape[1]

        # ----建立資料夾----
        image_type = "parcel_NIRRGA_selected_parcels" if self.select_specific_parcels else "parcel_NIRRGA"
        # 建立存放該image_type的parcel data的資料夾

        if os.path.isdir(self.dataset_root_folder_path + "/" + image_type) is False:
            os.mkdir(self.dataset_root_folder_path + "/" + image_type)

        # 建立用於存放從單一個frame所產生的parcel data的資料夾
        # 但是若是從framelet產生坵塊資料的話, 沒有區分不同來源frame的概念,
        # 所有frame的坵塊資料全部放在同一個資料夾(frame_codename為0)
        self.saving_dirname = self.dataset_root_folder_path + "/{}/{}".format(image_type, str(self.frame_codename))
        if os.path.isdir(self.saving_dirname) is False:
            os.mkdir(self.saving_dirname)

        region_list = regionprops(self.labeled_component_GT_mask)

        # from PIL import Image
        image_height, image_width = self.labeled_component_GT_mask.shape
        # label_id = np.zeros([height, width, 3], dtype=np.uint8)
        # from random import randint
        # colorMapping = dict()
        # smallCount = 0
        
        # print(self.labeled_component_GT_mask.shape)
        # print('labeled_component_count = ', labeled_component_count)
        # for i in range(labeled_component_count+1):
        #     colorMapping[i] = [randint(100, 255), randint(100, 255), randint(100, 255)]
        # for i in range(labeled_component_count):
        #     if len(region_list[i].coords) < 500:
        #         smallCount += 1
        #     for j in region_list[i].coords:
        #         label_id[j[0], j[1]] = colorMapping[i]
        # print('smallCount = ', smallCount)
        # image = Image.fromarray(label_id)
        # image.save("show_labeled_component.png")
        # n_region_list = len(region_list)
        # logger.info(f"Number of disconnected regions of arbitrary size: %s", str(n_region_list))
        # exit(0)

        # hashmap for remembering which parcels(regions) are actually taken from regionprops
        parcels_taken_from_region_props = dict()
        for region in region_list:
            parcels_taken_from_region_props[region.bbox] = False
        for region in region_list:  # region : 透過regionprops取得的各個連通區域物件, 可看作具有每個parcel的資訊
            # 忽略面積過小的連通區域(可能是因為polygon重疊產生的非坵塊連通區域)
            if region.area > min_valid_parcel_area_size:
                focus_parcel_region = region

                # ----設定坵塊的邊框index----
                # Region邊界外框("真正的行列index") : min_row, min_col, max_row, max_col
                # 由region.bbox取得的行列"下限"，min_row、min_col為"真正的行列index"
                # 由region.bbox取得的行列"上限"，max_row_plus1、max_col_plus1為"真正的行列index"再加一，
                # 應該是為了方便使用array[min_row:max_row_plus1,
                # min_col:max_col_plus1]取得該region所涵蓋的所有像素
                boundary_box = focus_parcel_region.bbox
                min_row = boundary_box[0]
                max_row_plus1 = boundary_box[2]
                max_row = max_row_plus1 - 1
                min_col = boundary_box[1]
                max_col_plus1 = boundary_box[3]
                max_col = max_col_plus1 - 1
                minimum_parcel_image_boundary = BoundaryBox(
                    min_row, max_row, min_col, max_col)

                if self.fixed_shape is None:
                    # 若沒有指定一個固定大小,
                    # 會以坵塊原始的邊界作為要產生的坵塊影像大小,
                    # 每個坵塊的坵塊影像大小不一致
                    fixed_height = minimum_parcel_image_boundary.height
                    fixed_width = minimum_parcel_image_boundary.width

                # 篩選 若焦點坵塊過細小=>視為無效坵塊
                # 無視掉某邊低於18個像素寬的坵塊(應該都是polygon重疊導致的錯誤坵塊)
                # 寬高皆為18個以上像素寬才是有效坵塊
                if minimum_parcel_image_boundary.height >= 18 and minimum_parcel_image_boundary.width >= 18:
                    parcel_count = parcel_count + 1  # 目前正在處理第幾個(有效的)parcel
                    # logger.info("parcel_count: {}    parcel area: {}".format(parcel_count, focus_parcel_region.area))

                    # ----檢查focus_parcel 是否是rare case的坵塊----
                    if self.rare_case_annotation_mask_folder is not None:
                        focus_parcel_region_rare_case = 0  # 預設為非rare case, case 0
                        focus_parcel_region_centroid = (
                            int(focus_parcel_region.centroid[0]),
                            int(focus_parcel_region.centroid[1]))

                        logger.info(
                            f"parcel_count: {parcel_count}    focus_parcel_region_centroid: {focus_parcel_region_centroid}")
                        for rare_case_region in rare_case_regions:
                            if focus_parcel_region_centroid == rare_case_region.centroid:
                                focus_parcel_region_rare_case = rare_case_region.rare_case
                                rare_case_region.parcel_count = parcel_count  # 在此航照上所有有效坵塊的坵塊編號

                    # ----計算要裁切的坵塊影像邊界----
                    """
                    self.based_on_framelet_dataset is True
                    若從framelet dataset產生256x256大小的坵塊資料, 直接設定邊界即可
                    min_col = 0
                    max_col = 255
                    min_row = 0
                    max_row = 255
                    self.based_on_framelet_dataset is False call get_partial_area_boundary
                    """
                    [min_col, max_col, min_row, max_row] = [0, 255, 0,
                                                            255] if self.based_on_framelet_dataset else get_partial_area_boundary(
                        min_col=minimum_parcel_image_boundary.min_col,
                        max_col=minimum_parcel_image_boundary.max_col,
                        min_row=minimum_parcel_image_boundary.min_row,
                        max_row=minimum_parcel_image_boundary.max_row,
                        partial_area_width=fixed_width,
                        partial_area_height=fixed_height,
                        parcel_image_min_col=0,
                        # parcel_image_max_col=11460 - 1,
                        parcel_image_max_col=image_width - 1,
                        parcel_image_min_row=0,
                        # parcel_image_max_row=12260 - 1
                        parcel_image_max_row=image_height - 1
                    )
                    # print(parcel_count, region.bbox, min_col, max_col, min_row, max_row)

                    # 以Boundary_box物件保存要裁切的邊界
                    cropped_parcel_image_boundary = BoundaryBox(
                        min_row, max_row, min_col, max_col)
                    final_parcel_image_boundary = cropped_parcel_image_boundary  # 暫存 若有進行window scan 有可能更改
                    # logger.info(
                    #     "cropped_parcel_image_boundary.height, cropped_parcel_image_boundary.width:",
                    # cropped_parcel_image_boundary.height,
                    # cropped_parcel_image_boundary.width)

                    # 計算focus parcel在裁切後的邊界框中的面積比例
                    pixel_ratio_of_focus_parcel_in_cropped_boundary_box = \
                        Preprocessing_per_frame.calculate_pixel_ratio_of_focus_parcel_in_boundary_box(
                            focus_parcel_region,
                            labeled_connected_component_GT_mask=self.labeled_component_GT_mask,
                            boundary_box=cropped_parcel_image_boundary
                        )

                    # 檢查裁切出的partial area中是否包含夠多focus parcel
                    if pixel_ratio_of_focus_parcel_in_cropped_boundary_box <= 0.6 and (
                            minimum_parcel_image_boundary.height >= fixed_height and
                            minimum_parcel_image_boundary.width >= fixed_width):

                        # 若裁切出來的partial area包含焦點坵塊的比例<60%, 可能是特殊形狀的坵塊
                        # 要另外用window scan的作法找出有沒有包含較多focus parcel的partial area
                        # boundary
                        pixel_ratio_of_focus_parcel_in_scanned_boundary_box, scanned_parcel_image_boundary = \
                            Preprocessing_per_frame.calculate_pixel_ratio_of_focus_parcel_in_scanned_boundary_box(
                                self.labeled_component_GT_mask,
                                focus_parcel_region,
                                minimum_parcel_image_boundary,
                                fixed_height,
                                fixed_width
                            )

                        # 涵蓋較多focus parcel的邊框會作為最終的坵塊影像邊框
                        if pixel_ratio_of_focus_parcel_in_scanned_boundary_box > \
                                pixel_ratio_of_focus_parcel_in_cropped_boundary_box:
                            # 取代先前的final_parcel_image_boundary = cropped_parcel_image_boundary
                            final_parcel_image_boundary = scanned_parcel_image_boundary

                            # ----裁切產生坵塊影像(NIRRG image和Annotation image)----
                    # logger.info("  final_parcel_image_boundary => 坵塊影像大小:", nirrg_image_array.shape)
                    nirrg_image_array = self.whole_frame_NIRRG[
                                        final_parcel_image_boundary.min_row:final_parcel_image_boundary.max_row_plus,
                                        final_parcel_image_boundary.min_col:final_parcel_image_boundary.max_col_plus
                                        ]
                    # logger.info("  final_parcel_image_boundary => 坵塊影像大小:", nirrg_image_array.shape)
                    # 從labeled_component_GT_mask擷取此焦點坵塊之parcel image大小的mask，
                    # 即焦點坵塊的annotation mask(GT mask)，像素值為True代表該像素屬於焦點坵塊內的像素
                    annotation_mask_image_array = (
                            self.labeled_component_GT_mask[
                            final_parcel_image_boundary.min_row:final_parcel_image_boundary.max_row_plus,
                            final_parcel_image_boundary.min_col:final_parcel_image_boundary.max_col_plus
                            ] == focus_parcel_region.label)  # True=parcel area
                    region_mask = self.labeled_component_GT_mask[
                            final_parcel_image_boundary.min_row:final_parcel_image_boundary.max_row_plus,
                            final_parcel_image_boundary.min_col:final_parcel_image_boundary.max_col_plus
                            ]
                    # logger.info(f"{parcel_count} : {focus_parcel_region.label}")
                    # for element in np.unique(region_mask):
                    #     logger.info(f"\t{element}:{np.sum(region_mask==element)}")
                    if saved_image_type == "png":
                        annotation_mask_image_array = annotation_mask_image_array.astype(
                            "uint8")
                        # 為了方便觀看focus parcel才將focus parcel的像素值設為255
                        annotation_mask_image_array[np.where(
                            annotation_mask_image_array == 1)] = 255

                        # ----檢查裁切產生的坵塊影像大小是否為所求----
                    if nirrg_image_array.shape[0] != fixed_height or nirrg_image_array.shape[1] != fixed_width:
                        logger.info(
                            "\n\t ***nirrg_image_array.shape[0] != fixed_height or "
                            "nirrg_image_array.shape[1] != fixed_width")
                        logger.info("\tnirrg_image_array.shape  %s ",
                                    str(nirrg_image_array.shape))
                        logger.info(f"\tself.frame_codename:{self.frame_codename}  ,  parcel_count:{parcel_count}")
                        continue
                        #sys.exit(1)

                    logger.info(f"\t{np.sum(annotation_mask_image_array > 0)}")
                    # 將背景遮住
                    for channel in range(in_shape):
                        # logger.info("changing mask")
                        nirrg_image_array[:, :, channel] = np.where(annotation_mask_image_array > 0, nirrg_image_array[:, :, channel], 0)
                        # 將NIR, R, G, Annotation image合併成四通道影像
                    # Four channels: NIR, R, G,B Annotation mask
                    if in_shape==4:
                        nirrga_image_array = np.stack(
                            (nirrg_image_array[:, :, 0],
                            nirrg_image_array[:, :, 1],
                            nirrg_image_array[:, :, 2],
                            nirrg_image_array[:, :, 3],
                            annotation_mask_image_array), axis=2)
                    if in_shape==3:
                        nirrga_image_array = np.stack(
                            (nirrg_image_array[:, :, 0],
                            nirrg_image_array[:, :, 1],
                            nirrg_image_array[:, :, 2],
                            annotation_mask_image_array), axis=2)
                    logger.info(f"  final_parcel_image_boundary => 坵塊影像大小: {str(nirrga_image_array.shape[0])} {str(nirrga_image_array.shape[1])} {str(nirrga_image_array.shape[2])}")
                    # ----設定要儲存的parcel image路徑檔名----
                    # based_on_framelet_dataset is False
                    # 檔名 e.g. f{frame_codename}_{第幾個parcel
                    # image}_NIRRGA.npy
                    # based_on_framelet_dataset is True
                    # 若從framelet dataset產生坵塊資料, 不會區分來源是哪個frame, 因此不會在檔名紀錄f{frame_codename}
                    # 檔名 e.g. {第幾個parcel image}_NIRRGA.npy

                    saving_file_name = self.saving_dirname + \
                                       f"/{parcel_count}_{image_type}" if self.based_on_framelet_dataset else self.saving_dirname + \
                                                                                                              f"/f{self.frame_codename}_{parcel_count}_{image_type}"

                    # 在檔名加入坵塊的面積大小和GT資訊
                    saving_file_name = saving_file_name + f"_area {focus_parcel_region.area}"
                    saving_file_name = saving_file_name + f"_GT {self.get_GT_label_of_focus_parcel(focus_parcel_region)}"

                    if self.rare_case_annotation_mask_folder is not None and focus_parcel_region_rare_case != 0:
                        # 在檔名加入此坵塊的rare case資訊(不影響程式執行, 可去除)
                        saving_file_name = saving_file_name + f"_RareCase{focus_parcel_region_rare_case}"

                    # ----根據設定的saved_image_type儲存parcel image----
                    saving_file_name = self.save_nirrga_image(saving_file_name, nirrga_image_array,
                                                              filetype=saved_image_type)

                    # ----將此focus parcel的資訊加入到list----
                    # 紀錄parcel大小 此資訊只用於Get rice parcel sie.ipynb
                    parcel_size_list.append(nirrg_image_array.shape)
                    parcel_image_path_list.append(
                        saving_file_name)  # 紀錄保存於硬碟中的parcel image路徑
                    # focus parcel的GT label
                    parcel_gt_label = self.get_GT_label_of_focus_parcel(
                        focus_parcel_region)
                    parcel_GT_label_list.append(
                        parcel_gt_label)  # 紀錄焦點坵塊的GT label

                    parcels_taken_from_region_props[region.bbox] = True

        # save the parcels that are actually taken into a txt file, mapped by their region.bbox
        with open(os.path.join(self.saving_dirname, "parcels_that_are_actually_taken.txt"), 'w',
                  encoding="utf-8") as _file_handler:

            _number_of_parcels_that_are_actually_taken = 0
            for k in parcels_taken_from_region_props.keys():
                _key_string = f"{k[0]}_{k[1]}_{k[2]}_{k[3]}"
                if parcels_taken_from_region_props[k] is True:
                    _number_of_parcels_that_are_actually_taken += 1
                    _file_handler.write(f"{_key_string} taken\n")
                else:
                    _file_handler.write(f"{_key_string} not_taken\n")
            logger.info("number of parcels that are actually taken: (some can be too small so we ignore them) %s",
                        str(_number_of_parcels_that_are_actually_taken))
        if self.based_on_framelet_dataset is False:
            self.__save_GT_labels_to_txt(parcel_GT_label_list)
        else:
            # 更新self.parcel_count_for_framelet_dataset
            self.parcel_count_for_framelet_dataset = parcel_count
            self.__save_GT_labels_to_txt_for_framelet_dataset(
                parcel_GT_label_list)

        return self.saving_dirname, parcel_image_path_list, parcel_GT_label_list, parcel_size_list, rare_case_regions

    def save_nirrga_image(self, saving_file_name, nirrga_image_array, filetype="png"):
        """
        儲存NIRRGA parcel image

        回傳saving_file_name
        """
        if filetype == "npy":  # Saved as .npy
            saving_file_name = saving_file_name + ".npy"
            np.save(saving_file_name, nirrga_image_array)
        if filetype == "png":  # Saved as .png
            saving_file_name = saving_file_name + ".png"
            nirrga_image_array = nirrga_image_array[:, :, :4]
            pil_image = Image.fromarray(nirrga_image_array)
            pil_image.save(saving_file_name)
        return saving_file_name


def cal_partial_area_new_boundary_of_row_or_column(area_min, area_max, target_area_length,
                                                   parcel_image_max, parcel_image_min=0):
    """
    在parcel image中挖取partial area

    area_min:
        輸入的區域的邊框最小index
    area_max:
        輸入的區域的邊框最大index
    target_area_length:
        所求的區域邊框長度
    parcel_image_max:
        坵塊影像的邊框最大index
    parcel_image_min:
        坵塊影像的邊框最小index
    """
    cur_area_length = area_max - area_min + 1  # 坵塊區域的某邊長度, 長度為像素數量(index的間隔+1)
    length_diff = target_area_length - cur_area_length  # 長度差異(像素數量)
    odd_length_diff = False  # 長度差異是否為奇數, 初始化

    if length_diff != 0:
        if length_diff > 0:  # 所求區域長度大於坵塊區域的長度, 需要擷取較大的parcel image => 邊界外擴
            if length_diff % 2 == 0:  # 偶數
                length_diff_half = int(length_diff / 2)
            else:
                length_diff_half = int(
                    math.ceil(length_diff / 2))  # 除2後有小數=>進位
                odd_length_diff = True  # 長度差異為奇數
            new_min = area_min - length_diff_half
            new_max = area_max + length_diff_half

            # 檢查邊界改變後是否超出最大最小邊界
            if new_min < parcel_image_min:  # 超出最左/上的邊界=>右/下移 超出的像素長度
                over_length_diff = parcel_image_min - new_min
                new_min = new_min + over_length_diff
                new_max = new_max + over_length_diff
            if new_max > parcel_image_max:  # 超出最右/下的邊界=>左/上移 超出的像素長度
                over_length_diff = new_max - parcel_image_max
                new_min = new_min - over_length_diff
                new_max = new_max - over_length_diff

            if odd_length_diff is True:
                new_max = new_max - 1
        if length_diff < 0:  # 所求區域長度小於坵塊區域的長度, 需要擷取較小的parcel image =>邊界內縮
            # logger.info("offset_half == offset//2 : ", offset//2)
            if abs(length_diff) % 2 == 0:  # 偶數
                length_diff_half = length_diff // 2
            else:
                length_diff_half = length_diff // 2  # 內縮的位移量會多1  e.g. -3//2 = -2
                odd_length_diff = True

            new_min = area_min - length_diff_half
            new_max = area_max + length_diff_half
            # logger.info("new_min, new_max", new_min, new_max)

            # 原有的邊界內縮不會有超出邊界的情況

            if odd_length_diff is True:
                # logger.info("odd_length_diff == True")
                new_max = new_max + 1  # 補回多出的1個內縮位移量

        return new_min, new_max
    else:  # 不需要改變邊界
        return area_min, area_max


def get_partial_area_boundary(min_col, max_col, min_row, max_row,
                              partial_area_width, partial_area_height,
                              parcel_image_min_col, parcel_image_max_col,
                              parcel_image_min_row, parcel_image_max_row):
    """
    center_col:
        焦點坵塊正中心x軸座標
    center_row:
        焦點坵塊正中心y軸座標

    若要裁切的影像來源是完整航照12260x11460, 裁切出來的坵塊影像邊界最小最大值如下
        parcel_image_min_col=0,
        parcel_image_max_col=11460-1,
        parcel_image_min_row=0,
        parcel_image_max_row=12260-1

    """

    # 寬
    area_min_col, area_max_col = cal_partial_area_new_boundary_of_row_or_column(
        area_min=min_col,
        area_max=max_col,
        target_area_length=partial_area_width,
        parcel_image_max=parcel_image_max_col,
        parcel_image_min=0)

    # 高
    area_min_row, area_max_row = cal_partial_area_new_boundary_of_row_or_column(
        area_min=min_row,
        area_max=max_row,
        target_area_length=partial_area_height,
        parcel_image_max=parcel_image_max_row,
        parcel_image_min=0)

    return area_min_col, area_max_col, area_min_row, area_max_row


def window_scan_for_searching_focus_parcel(
        image, window_height, window_width, target_value=1):
    """
    目前只使用於  固定大小<焦點坵塊  => image_max_row - window_height + 1    為正數

    但當 固定大小>焦點坵塊  => image_max_row - window_height + 1    為負數
    當 固定大小>焦點坵塊 一定包含到焦點坵塊 因此也不用做window scan

    """
    logger.info("---window_scan_for_searching_focus_parcel()")
    image_max_row = image.shape[0] - 1
    image_max_col = image.shape[1] - 1

    final_window_max_row = image_max_row - window_height + 1
    final_window_max_col = image_max_col - window_width + 1
    logger.info("image_max_row: %s, window_height : %s", str(image_max_row),  str(window_height))
    logger.info("image_max_col: %s, window_width: %s",  str(image_max_col),  str(window_width))

    max_target_value_ratio_of_image_array = -1  # 初始化
    # logger.info("-----------")
    for window_min_row in range(0, final_window_max_row + 1, 5):
        for window_min_col in range(0, final_window_max_col + 1, 5):
            window_max_row = window_min_row + window_height - 1
            window_max_col = window_min_col + window_width - 1
            window_max_row_plus = window_max_row + 1
            window_max_col_plus = window_max_col + 1
            window = image[window_min_row:window_max_row_plus, window_min_col:window_max_col_plus]

            image_array = window
            target_value_count = np.count_nonzero(image_array == target_value)
            if image_array.size == 0:
                logger.info("window_min_row: %s  window_min_col: %s",
                            str(window_min_row), str(window_min_col))
                logger.info("window_max_row: %s  window_max_col: %s",
                            str(window_max_row), str(window_max_col))
                logger.info("image_max_row: %s  image_max_col: %s ",
                            str(image_max_row), str(image_max_col))
            target_value_ratio_of_image_array = target_value_count / image_array.size

            if target_value_ratio_of_image_array > max_target_value_ratio_of_image_array:
                # 更新
                max_target_value_ratio_of_image_array = target_value_ratio_of_image_array
                # 紀錄此window的資訊
                temp_window_min_row = window_min_row
                temp_window_max_row = window_max_row
                temp_window_min_col = window_min_col
                temp_window_max_col = window_max_col

            if max_target_value_ratio_of_image_array == 1:
                break
        if max_target_value_ratio_of_image_array == 1:
            break

    if max_target_value_ratio_of_image_array == -1:
        logger.info("*** max_target_value_ratio_of_image_array == -1    影像中沒有焦點坵塊")
        sys.exit(1)

    return temp_window_min_col, temp_window_max_col, temp_window_min_row, temp_window_max_row


class BoundaryBox:
    """Creating boundary box object"""

    def __init__(self, min_row, max_row, min_col, max_col):
        self.min_row = min_row  # 高 y軸值
        self.max_row = max_row
        self.min_col = min_col  # 寬 x軸值
        self.max_col = max_col
        self.max_row_plus = max_row + 1
        self.max_col_plus = max_col + 1
        self.width = self.max_col_plus - self.min_col
        self.height = self.max_row_plus - self.min_row


class RareCaseRegion:
    """handling rare case"""

    def __init__(self, centroid, rare_case, parcel_count):
        self.centroid = centroid
        self.rare_case = rare_case  # label
        self.parcel_count = parcel_count
