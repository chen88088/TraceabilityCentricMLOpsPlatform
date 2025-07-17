import sys
import glob
import numpy as np
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import src.preprocessing.parcel_preprocessing as parcel_preprocessing
import src.utils.excel_io as Excel_IO
import src.models.parcel_based_CNN as parcel_based_CNN
import logging
from configs.config import BANDS

def get_partial_area(
        NIRRGA_image_array,
        partial_area_height,
        partial_area_width
):
    """
    輸入NIRRGA_image_array, 取得partial_area

    若region_centroid_is_center==True, 以annnotation mask中目標坵塊的質心作為partial_area中心
        case "異常的坵塊"且parcel image取partial_area : 在annnotation mask中可能會沒有目標坵塊(即取到其他坵塊的區域),
        就以annnotation mask的中心作為partial_area中心

    若region_centroid_is_center==False, 以annnotation mask的中心作為partial_area中心

    """
    # if BANDS==4 :
    #     annotation_mask = NIRRGA_image_array[:, :, 4]
    # if BANDS==3 :
    #     annotation_mask = NIRRGA_image_array[:, :, 3]
        
    # # the pixel value of the target parcel is 1(true), the other is 0(background)

    # # col index start from 0, so the max col index is the width - 1
    # parcel_image_max_col = annotation_mask.shape[1] - 1
    # parcel_image_max_row = annotation_mask.shape[0] - 1

    # labeled_component_GT_mask, labeled_component_count = label(
    #     annotation_mask, background=0, return_num=True, connectivity=2)
    # region_count = 0
    # region_list = regionprops(labeled_component_GT_mask)
    # no_region =False
    # if len(region_list) != 0:
    #     # 每個parcel的資訊    對於每個parcel image, 最多只有一個region(target parcel)
    #     for region in region_list:
    #         region_count += 1
    #         # logging.info("region_count:",region_count)

    #         focus_parcel_region = region  # 透過regionprops取得的各個連通區域物件

    #         boundary_box = focus_parcel_region.bbox
    #         min_row = boundary_box[0]
    #         max_row_plus1 = boundary_box[2]
    #         max_row = max_row_plus1 - 1
    #         min_col = boundary_box[1]
    #         max_col_plus1 = boundary_box[3]
    #         max_col = max_col_plus1 - 1
    #         minimum_parcel_image_boundary = parcel_preprocessing.BoundaryBox(
    #             min_row, max_row, min_col, max_col)

    #         centroid_col = int(region.centroid[1])  # 寬 x軸
    #         centroid_row = int(region.centroid[0])  # 高 y軸
    #         min_row = centroid_row
    #         max_row = centroid_row + 1
    #         min_col = centroid_col
    #         max_col = centroid_col + 1
    # else : # annotation mask中沒有目標坵塊之連通區域
    #     focus_parcel_region = labeled_component_GT_mask
    #     min_row = 0
    #     max_row = annotation_mask.shape[0] - 1 # 高 y軸值
    #     min_col = 0
    #     max_col = annotation_mask.shape[1] - 1 # 寬 x軸值
    #     no_region=True
    max_row, max_col, channel = NIRRGA_image_array.shape
    max_row -= 1
    max_col -= 1
    min_row, min_col = 0, 0
    parcel_image_max_col, parcel_image_max_row = max_col, max_row
    # ----計算要裁切的坵塊影像邊界----
    min_col, max_col, min_row, max_row = parcel_preprocessing.get_partial_area_boundary(
        min_col=min_col,
        max_col=max_col,
        min_row=min_row,
        max_row=max_row,
        partial_area_width=partial_area_width,
        partial_area_height=partial_area_height,
        parcel_image_min_col=0, parcel_image_max_col=parcel_image_max_col,
        parcel_image_min_row=0, parcel_image_max_row=parcel_image_max_row
    )
    max_row_plus1 = max_row + 1
    max_col_plus1 = max_col + 1

    # 以Boundary_box物件保存要裁切的邊界
    cropped_parcel_image_boundary = parcel_preprocessing.BoundaryBox(
        min_row, max_row, min_col, max_col)
    final_parcel_image_boundary = cropped_parcel_image_boundary  # 暫存 若有進行window scan 有可能更改
    # if no_region:
    #     pixel_ratio_of_focus_parcel_in_cropped_boundary_box=1
    # else:
    #     pixel_ratio_of_focus_parcel_in_cropped_boundary_box = \
    #         parcel_preprocessing.Preprocessing_per_frame.calculate_pixel_ratio_of_focus_parcel_in_boundary_box(
    #             focus_parcel_region,
    #             labeled_connected_component_GT_mask=labeled_component_GT_mask,
    #             boundary_box=cropped_parcel_image_boundary
    #         )

    # if pixel_ratio_of_focus_parcel_in_cropped_boundary_box <= 0.6 and (
    #         minimum_parcel_image_boundary.height >= partial_area_height and
    #         minimum_parcel_image_boundary.width >=
    #         partial_area_width):

    #     # 若裁切出來的partial area包含焦點坵塊的比例<60%, 可能是特殊形狀的坵塊
    #     # 要另外用window scan的作法找出有沒有包含較多focus parcel的partial area boundary
    #     pixel_ratio_of_focus_parcel_in_scanned_boundary_box, scanned_parcel_image_boundary = \
    #         parcel_preprocessing.Preprocessing_per_frame.calculate_pixel_ratio_of_focus_parcel_in_scanned_boundary_box(
    #             labeled_component_GT_mask,
    #             focus_parcel_region,
    #             minimum_parcel_image_boundary,
    #             partial_area_height,
    #             partial_area_width
    #         )

    #     # 涵蓋較多focus parcel的邊框會作為最終的部分區域邊框
    #     if pixel_ratio_of_focus_parcel_in_scanned_boundary_box > pixel_ratio_of_focus_parcel_in_cropped_boundary_box:
    #         # 取代先前的final_parcel_image_boundary = cropped_parcel_image_boundary
    #         final_parcel_image_boundary = scanned_parcel_image_boundary

    #     pixel_ratio_of_focus_parcel_in_scanned_boundary_box, scanned_parcel_image_boundary = \
    #         parcel_preprocessing.Preprocessing_per_frame.calculate_pixel_ratio_of_focus_parcel_in_scanned_boundary_box(
    #             labeled_component_GT_mask,
    #             focus_parcel_region,
    #             minimum_parcel_image_boundary,
    #             partial_area_height,
    #             partial_area_width
    #         )

    # ----裁切產生部分區域影像(NIRRG image和Annotation image)----
    partial_area = NIRRGA_image_array[
                   final_parcel_image_boundary.min_row:final_parcel_image_boundary.max_row_plus,
                   final_parcel_image_boundary.min_col:final_parcel_image_boundary.max_col_plus, 0:3]

    # ----檢查裁切產生的部分區域影像大小是否為所求----
    if partial_area.shape[0] != partial_area_height or partial_area.shape[1] != partial_area_width:
        # TODO move it to logger
        logging.info(
            "***  partial_area.shape[0] != partial_area_height or partial_area.shape[1] != partial_area_width ")
        logging.info("max_row:{}   max_col:{}".format(max_row, max_col))
        logging.info("max_row-min_row+1:{}   max_col-min_col+1:{}".format(max_row -
                                                                          min_row + 1, max_col - min_col + 1))
        sys.exit(1)

    return partial_area


def add_parcel_data(dict, parcel_NIRRGA_path, parcel_GT_label, partial_area):
    dict["parcel_NIRRGA_path_list"].append(parcel_NIRRGA_path)
    dict["parcel_GT_label_list"].append(parcel_GT_label)
    dict["partial_area"].append(partial_area)
    return dict


def do_Kmeans_clustering(image_list, clusters_amount):
    """
    將每一筆3通道2維影像攤平成一維陣列, 正規化後呼叫Kmeans演算法
    """
    flatten_image_list = np.float32(image_list).reshape(
        len(image_list), -1)  # 將每一筆3通道2維影像攤平成一維陣列
    flatten_image_list /= 255  # 正規化 縮放至0~1之間
    print("start clustering")
    kmeans_model = KMeans(n_clusters=clusters_amount, random_state=0)
    kmeans_model_fit = kmeans_model.fit(flatten_image_list)
    predictions = kmeans_model_fit.labels_  # 分群結果 屬於哪個cluster
    print("end clustering")

    return predictions


def split_rice_and_nonrice(combined_frame_dataset):
    """
    對所有parcel data, 根據焦點坵塊是否為水稻, 拆分成rice_parcel_dataset和non_rice_parcel_dataset
    """

    rice_parcel_dataset = {
        "parcel_NIRRGA_path_list": [],
        "parcel_GT_label_list": [],
        "partial_area": []
    }
    non_rice_parcel_dataset = {
        "parcel_NIRRGA_path_list": [],
        "parcel_GT_label_list": [],
        "partial_area": []
    }

    for i in range(0, len(combined_frame_dataset["parcel_NIRRGA_path_list"])):
        if combined_frame_dataset["parcel_GT_label_list"][i] == 1:  # rice
            rice_parcel_dataset = add_parcel_data(rice_parcel_dataset,
                                                  parcel_NIRRGA_path=combined_frame_dataset[
                                                      "parcel_NIRRGA_path_list"][i],
                                                  parcel_GT_label=combined_frame_dataset["parcel_GT_label_list"][i],
                                                  partial_area=combined_frame_dataset["partial_area"][i]
                                                  )
        else:  # non-rice
            non_rice_parcel_dataset = add_parcel_data(non_rice_parcel_dataset,
                                                      parcel_NIRRGA_path=combined_frame_dataset[
                                                          "parcel_NIRRGA_path_list"][i],
                                                      parcel_GT_label=combined_frame_dataset[
                                                          "parcel_GT_label_list"][i],
                                                      partial_area=combined_frame_dataset["partial_area"][i]
                                                      )
    return rice_parcel_dataset, non_rice_parcel_dataset


def build_clustering_result_storage(
        rice_clusters_amount, non_rice_clusters_amount):
    """

    建立以每個cluster為key的的dictionary "clusters_storage"

    clusters_storage:
        {
            "Rice_cluster_0":[
                "path, GT label",
                "path, GT label",
                ...
            ],
            .
            .
            .
            "Rice_cluster_n":[

            ],

            "NonRice_cluster_0":[

            ],
            .
            .
            .
            "NonRice_cluster_m":[

            ]
        }


    rice_clusters_amount:
        焦點坵塊為水稻的parcel image經過Kmeans分群的cluster數
    non_rice_clusters_amount:
        焦點坵塊為非水稻的parcel image經過Kmeans分群的cluster數
    """
    clusters_storage = {}
    for count in range(0, rice_clusters_amount):
        cluster_name = "Rice_cluster_{}".format(count)
        clusters_storage[cluster_name] = []

    for count in range(0, non_rice_clusters_amount):
        cluster_name = "NonRice_cluster_{}".format(count)
        clusters_storage[cluster_name] = []

    return clusters_storage


def combine_clustering_result(rice_parcel_dataset, non_rice_parcel_dataset,
                              rice_predictions, non_rice_predictions, clusters_storage):
    """
    根據Kmeans的分群結果, 以字串"{parcel_NIRRGA_path}, {parcel_GT_label}"的形式,
    將parcel data整理到對應的cluster list, 便於寫檔保存

    clusters_storage:
        {
            "Rice_cluster_0":[
                "path, GT label",
                "path, GT label",
                ...
            ],
            .
            .
            .
            "Rice_cluster_n":[

            ],

            "NonRice_cluster_0":[

            ],
            .
            .
            .
            "NonRice_cluster_m":[

            ]
        }

    """
    for i in range(0, len(rice_predictions)):
        elem = rice_parcel_dataset["parcel_NIRRGA_path_list"][i] + \
               ", " + str(rice_parcel_dataset["parcel_GT_label_list"][i])
        elem_clustering_result = rice_predictions[i]

        cluster_name = "Rice_cluster_{}".format(elem_clustering_result)
        clusters_storage[cluster_name].append(elem)

    for i in range(0, len(non_rice_predictions)):
        elem = non_rice_parcel_dataset["parcel_NIRRGA_path_list"][i] + \
               ", " + str(non_rice_parcel_dataset["parcel_GT_label_list"][i])
        elem_clustering_result = non_rice_predictions[i]

        cluster_name = "NonRice_cluster_{}".format(elem_clustering_result)
        clusters_storage[cluster_name].append(elem)

    return clusters_storage


def KMeans_clustering_and_save_result_txt(
        shape,
        parcel_partial_area_shape,
        rice_cluster_n,
        non_rice_cluster_n,
        excel_path,
        frame_dataset_list,
        Kmeans_result_root_dirname,
        train_on_all_frames=False,
        Data_root_folder_path='data/train_test',
        training_NRG_png_path="./data/train_test/NRG_png"
):
    """
    Kmeans_result_root_dirname:
        the path of the folder saving the Kmeans result txt for each round

    rice_clusters_amount:
        焦點坵塊為水稻的parcel image經過Kmeans分群的cluster數
    non_rice_clusters_amount:
        焦點坵塊為非水稻的parcel image經過Kmeans分群的cluster數

    """

    target_height = parcel_partial_area_shape[0]
    target_width = parcel_partial_area_shape[1]

    # 對五個回合的坵塊資料進行Kmeans clustering
    # NIRRG_folder_path = training_NRG_png_path
    # _all_png_list = glob.glob(NIRRG_folder_path + '/*.png')
    number_of_frames = len(frame_dataset_list)
    for round_number in range(1, 5 + 1):
        logging.info("\n************* round_number: {} *************".format(round_number))
        print(round_number)
        if not train_on_all_frames:
            # ----載入一個回合使用的圖框號等資訊----
            excel_info = Excel_IO.ExcelInfo(
                excel_path, round_number=round_number, number_of_frames=number_of_frames)
            # logging.info(f"excel_info.training_and_validation_ds_frame_codename_list: {' '.join(excel_info.training_and_validation_ds_frame_codename_list)}")
            # logging.info(f"excel_info.testing_ds_frame_codename_list:{' '.join(excel_info.training_and_validation_ds_frame_codename_list)}")
            combined_frame_dataset = parcel_based_CNN.combine_frame_dataset_from_each_frame(
                frame_codename_list=excel_info.training_and_validation_ds_frame_codename_list,
                frame_dataset_list=frame_dataset_list)
        else:
            # use all frames
            # frame_codename_list = range(0, number_of_frames)
            frame_codename_list = [ frame_dataset["frame_codename"] for frame_dataset in frame_dataset_list]
            combined_frame_dataset = parcel_based_CNN.combine_frame_dataset_from_each_frame(
                frame_codename_list=frame_codename_list,
                frame_dataset_list=frame_dataset_list)

        combined_frame_dataset["partial_area"] = []
        for NIRRGA_path in combined_frame_dataset["parcel_NIRRGA_path_list"]:
            NIRRGA_image_array = np.load(NIRRGA_path)
            partial_area = get_partial_area(
                NIRRGA_image_array, target_height, target_width)
            combined_frame_dataset["partial_area"].append(partial_area)
            # print(NIRRGA_path)

        logging.info("combined_frame_dataset[\"partial_area\"][0].shape: %s",
                     str(combined_frame_dataset["partial_area"][0].shape))

        rice_parcel_dataset, non_rice_parcel_dataset = split_rice_and_nonrice(
            combined_frame_dataset)
        print("Rice Clustering")
        rice_predictions = do_Kmeans_clustering(image_list=rice_parcel_dataset["partial_area"],
                                                clusters_amount=rice_cluster_n)  # 分群結果 屬於哪個cluster

        print("Non Rice Clustering")
        non_rice_predictions = do_Kmeans_clustering(image_list=non_rice_parcel_dataset["partial_area"],
                                                    clusters_amount=non_rice_cluster_n)  # 分群結果 屬於哪個cluster

        clusters_storage = build_clustering_result_storage(
            rice_cluster_n, non_rice_cluster_n)
        clusters_storage = combine_clustering_result(
            rice_parcel_dataset, non_rice_parcel_dataset, rice_predictions, non_rice_predictions, clusters_storage)

        # ----以txt紀錄Kmeans_result----
        Kmeans_result_txt_path = Kmeans_result_root_dirname + r"/Kmeans_Round{}_R{}NR{}_{}x{}.txt".format(
            round_number,
            rice_cluster_n,
            non_rice_cluster_n,
            shape[0],
            shape[1]
        )
        # 將分群結果寫入txt
        Kmeans_result_f = open(Kmeans_result_txt_path, 'w')
        for key in clusters_storage:
            Kmeans_result_f.write("-" + key + "-" + "\n")
            for parcel_data in clusters_storage[key]:
                parcel_data_with_forward_slash=parcel_data.replace("\\",'/')
                Kmeans_result_f.write(parcel_data_with_forward_slash + "\n")

        Kmeans_result_f.close()
        if train_on_all_frames:  # if we use all frames, no need to do 5 rounds
            break
