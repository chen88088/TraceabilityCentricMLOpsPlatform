# Auto

# 將genMask.py產生的annotation image轉換成只有需要的phase的水稻GT

# 檢查genMask.py產生的annotation image的pixel值狀況
# Field有給值的polygon 在轉換後的pixel值即為field的值
# Field中若有0，會轉換成無法預期的值 e.g. 15?
import shutil
import numpy as np
from PIL import Image
import sys
import glob
import os, logging

logger = logging.getLogger(__name__)


def get_area(city):
    area = "unknown"
    if city in ["宜蘭縣", "基隆市", "臺北市", "新北市", "桃園市", "新竹縣", "新竹市"]:
        area = "north"
    elif city in ["苗栗縣", "臺中市", "彰化縣", "雲林縣", "南投縣"]:
        area = "center"
    elif city in ["花蓮縣", "臺東縣"]:
        area = "east"
    elif city in ["嘉義市", "嘉義縣", "臺南市", "高雄市", "屏東縣"]:
        area = "south"
    return area


def get_match_city(image_list_sheet, row_index):
    return image_list_sheet['E' + str(row_index)].value


def get_row_in_All_AerialImage_list(all_aerial_image_list_sheet, BottomRow_withContent_of_sheet, target_frame_number):
    """
    目前只有藉此row得到位置資訊, 因此拍攝年分不同也沒差
    """
    for row_of_sheet in range(2, BottomRow_withContent_of_sheet + 1):
        if all_aerial_image_list_sheet['F' + str(row_of_sheet)].value == target_frame_number:
            return row_of_sheet

    logger.error("沒找到對應的frame number")
    sys.exit(1)


def get_Range_by_Percentage(percentage):
    if percentage <= 15:
        percetage_range = "Extremely Low"
    elif percentage < 40 and percentage > 15:
        percetage_range = "Low"
    elif percentage < 60 and percentage >= 40:
        percetage_range = "Balance"
    elif percentage > 60:
        percetage_range = "High"
    return percetage_range


def annotation_image_converting(image_array, model_version, target_phase):
    # 取得各像素值的所有pixel座標 (對於class為[0 1]的annotation image)
    a_loc = np.where(image_array == 0)  # 非水稻的其他作物 農地坵塊
    b_loc = np.where(image_array == 1)  # 水稻邱塊
    c_loc = np.where(image_array == 15) # 沒有被shp覆蓋到的區域
    f_loc = np.where(image_array == 99) # 無法判斷的農地=>忽略, 不用於訓練也不用於測試

    if model_version == "1.3":
        # 1.3 model 使用的annotation image不具有非水稻的農地坵塊
        # rice:255, non-rice:0
        # 全部先設為0，再根據target_phase取出所要的期別水稻
        image_array[a_loc] = 0
        image_array[b_loc] = 0
        image_array[f_loc] = 0
        image_array[c_loc] = 0

        if target_phase == "phase23_mixed":
            image_array[b_loc] = 255

    elif model_version == "1.5":
        # 1.4 model 使用的annotation image具有非水稻的農地坵塊
        # label_0:non-farmland 非農地坵塊的背景區域
        # label_1:background==non-rice farmland 非水稻的其他作物 農地坵塊
        # label_2:rice==rice GT farmland

        # 計算水稻比例時，若不考慮某一期別的水稻，要將其視為非水稻的農地坵塊(label_1)
        # 將農地坵塊(不論是否為水稻)全部先設為1(但unknown的農地坵塊因為要直接忽略,因此視為背景設為0)
        # 再根據target_phase取出所要的期別水稻
        image_array[a_loc] = 1
        image_array[b_loc] = 1
        image_array[f_loc] = 0
        image_array[c_loc] = 0

        if target_phase == "phase23_mixed":
            image_array[b_loc] = 2

    return image_array


def get_BottomRow_withContent_of_sheet(loaded_wb, sheet_name):
    """
    sheet.max_row的表格內容有可能為None

    bottom_row_withContent_of_sheet:
        excel中最後一筆"有內容"的表格的sheet index   
        => sheet['A' + str(bottom_row_withContent_of_sheet)].value 為最後一筆有內容的表格的value
    """
    sheet = loaded_wb.get_sheet_by_name(sheet_name)

    next_row_after_bottom = sheet.max_row + 1  # 最後一筆表格(內容可能為None)所在的row的下一個row
    for row_of_sheet in range(1, next_row_after_bottom + 1):  # 第一筆~最後一筆row的下一個row

        # 若最後一筆有內容的row的下一個row為內容為None => 回傳上一筆，即最後一筆有內容的row
        # or
        # 若遍歷到最後一筆row的下一個row才發現sheet中沒有內容為None的表格 => 回傳上一筆，即最後一筆有內容的row
        if sheet['A' + str(row_of_sheet)].value == None or row_of_sheet == next_row_after_bottom:
            bottom_row_withContent_of_sheet = row_of_sheet - 1
            return bottom_row_withContent_of_sheet


def create_rice_mask(image_folder_path, output_folder_path, model_version, target_phase):
    if os.path.isdir(output_folder_path):
        shutil.rmtree(output_folder_path)

    os.mkdir(output_folder_path)

    image_path_list = glob.glob(image_folder_path + "/*.png")
    i = 1  # 第幾列
    for image_path in image_path_list:

        # image_path = image_path_list[0]
        image = Image.open(image_path).convert("L")
        img = np.array(image)

        # ==============================================

        # 檔名為圖框號和拍攝日期 94202046_181006.png
        image_name = os.path.basename(image_path).split(".")[0]
        logger.info(image_path)
        # 去除"t"和".png"image_path
        FrameNumber_and_ShotDate = image_name.split("t")[1]
        FrameNumber_and_ShotDate = FrameNumber_and_ShotDate.replace(".png", "")
        # 下方code所用的frame_number 皆為 FrameNumber_and_ShotDate
        frame_number = FrameNumber_and_ShotDate.split("_")[0]

        # ==============================================
        logger.info("-----FrameNumber_and_ShotDate: %s", FrameNumber_and_ShotDate)

        val, count = np.unique(img, return_counts=True)  # 0:非農地 5:農地坵塊 10:水稻GT
        logger.info("     Before annotation_image_converting: %s %s", val, count)

        # 檢查 annotaion image的像素值是否有異常
        for v in val:
            if v not in [0, 1, 15]:
                logger.warning("********\n{}  灰階值異常  val:{}\n********".format(FrameNumber_and_ShotDate, val))

        # 對annotation image做phase篩選和影像強化
        img_copy = img.copy()
        img_copy = annotation_image_converting(img_copy, model_version, target_phase)

        val, count = np.unique(img_copy, return_counts=True)
        # val:
        # 1.3 model => 0:非農地坵塊背景 255:水稻坵塊 
        # 1.4 model => 0:非農地背景背景 1:非水稻的其他作物農地坵塊 2:水稻坵塊 
        logger.info("     After annotation_image_converting: %s %s", val, count)

        # 具有parcel 灰階值為100時的計算方法
        # rice_percentage_whole = 100*( count[2]/np.sum(count).astype('float64') )
        # rice_percentage_in_parcel = 100*( count[2]/(count[1]+count[2]).astype('float64') )

        # if model_version == "1.3":
        #     if len(val) == 2:
        #         rice_percentage_whole = 100 * (count[1] / np.sum(count).astype('float64'))
        #     elif len(val) == 1:  # annotation image無該phase的水稻
        #         rice_percentage_whole = 0
        # elif model_version == "1.4":
        #     if len(val) == 3:
        #         rice_percentage_whole = 100 * (count[2] / np.sum(count).astype('float64'))
        #         rice_percentage_in_parcel = 100 * (count[2] / (count[1] + count[2]).astype('float64'))
        #
        #     elif np.equal(val, np.array([0, 1])).all():
        #         rice_percentage_whole = 0
        #         rice_percentage_in_parcel = 0
        #
        #     elif np.equal(val, np.array([0, 2])).all():
        #         rice_percentage_whole = 100 * (count[1] / np.sum(count).astype('float64'))
        #         rice_percentage_in_parcel = 100
        #
        #     else:
        #         logger.warning("???")
        #         sys.exit(0)

        image_img = Image.fromarray(img_copy).convert("L")
        image_output_path = output_folder_path + "/" + FrameNumber_and_ShotDate + r".png"
        image_img.save(image_output_path)
        i = i + 1
