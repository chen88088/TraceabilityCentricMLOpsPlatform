import os
import glob
from src.preprocessing import parcel_preprocessing
import shutil

BACKUP_PATH = "/parcel_NIRRGA_backup"
def generate_parcel_dataset(
        shape, dataset_root_folder_path, image_type="npy",
        rare_case_annotation_mask_folder=None, select_specific_parcels=False, inference=False,
        NRG_png_path="./data/train_test/NRG_png",
        parcel_mask_path="./data/train_test/parcel_mask",
        selected_parcel_mask_path='./data/train_test/selected_parcel_mask',
        band =3):
    """
    shape:
        要產生的坵塊影像大小(不能比來源影像大)
    dataset_root_folder_path:
        存放"產生的坵塊資料"的資料夾路徑
    rare_case_annotation_mask_folder:
        存放"標記著每個rare case坵塊的annotation image"的資料夾路徑

    """
    if inference:
        select_specific_parcels = False

    output_dataset_root_folder_path = "{}/For_training_testing/{}x{}".format(
        dataset_root_folder_path, shape[0], shape[1])

    # 使用的GTmask是將polygon image和polyline image疊加而產生的parcel mask
    # the path of folder saving the parcel mask(==annotation image) ignoring
    # the parcel with unknown class
    GTmask_folder_path = parcel_mask_path
    selected_polygon_mask_folder_path = selected_parcel_mask_path

    if image_type == "npy":

        NIRRG_folder_path = NRG_png_path
    elif image_type == "png":
        # 若要輸出RBG影像 使用TIF擷取RGB通道
        NIRRG_folder_path = NRG_png_path

    # 存放dataset的資料夾
    if os.path.isdir(output_dataset_root_folder_path):
        shutil.rmtree(output_dataset_root_folder_path)  # 刪除舊資料
    os.makedirs(output_dataset_root_folder_path)

    _all_png_list = glob.glob(NIRRG_folder_path + '/*.png')
    number_of_frames = len(_all_png_list)
    # generate the parcel image based on the frame aerial image with frame_codename is in 0~number_of_frames
    # (the frame_codename is based on the the name of used CIR images saved in folder)
    target_frame_codename = (0, number_of_frames)

    if shape is not None:
        fixed_shape = (shape[0], shape[1])  # (寬, 高)
    else:
        fixed_shape = None
    output_Mask_dataset_root_folder_path = output_dataset_root_folder_path

    Data_preprocessing_GT = parcel_preprocessing.Data_preprocessing(
        dataset_root_folder_path=output_Mask_dataset_root_folder_path,
        NIRRG_folder_path=NIRRG_folder_path,
        GTmask_folder_path=GTmask_folder_path,
        fixed_shape=fixed_shape,
        target_frame_codename=target_frame_codename,
        saved_image_type=image_type,
        rare_case_annotation_mask_folder=rare_case_annotation_mask_folder,
        select_specific_parcels=False,
        inference=inference
    )
    # this operation extracts npy files from the rice mask
    frame_dataset_list, rare_case_regions = Data_preprocessing_GT.start_preprocessing(in_shape= band)

    if select_specific_parcels:
        # copy original parcel data
        if os.path.isdir(output_dataset_root_folder_path +
                         BACKUP_PATH):
            shutil.rmtree(output_dataset_root_folder_path +
                          BACKUP_PATH)  # 刪除舊資料
            os.makedirs(output_dataset_root_folder_path +
                        BACKUP_PATH)
        shutil.copytree(output_dataset_root_folder_path + "/parcel_NIRRGA",
                        output_dataset_root_folder_path + BACKUP_PATH)

        output_polygon_dataset_root_folder_path = output_dataset_root_folder_path

        Data_preprocessing_GT_selected_parcel = parcel_preprocessing.Data_preprocessing(
            dataset_root_folder_path=output_polygon_dataset_root_folder_path,
            NIRRG_folder_path=NIRRG_folder_path,
            GTmask_folder_path=selected_polygon_mask_folder_path,
            fixed_shape=fixed_shape,
            target_frame_codename=target_frame_codename,
            saved_image_type=image_type,
            rare_case_annotation_mask_folder=rare_case_annotation_mask_folder,
            select_specific_parcels=True
        )
        # this operation extracts npy files from the selected parcel mask, and instead of saving them
        # to data\train_test\For_training_testing\320x320\parcel_NIRRGA, this time it will save the .npys
        # to parcel_NIRRGA_selected_parcels
        frame_dataset_list1, rare_case_regions1 = Data_preprocessing_GT_selected_parcel.start_preprocessing(in_shape= band)

        for i in range(number_of_frames):
            # run_select_specific_parcels will look at both folders "parcel_NIRRGA" and
            # "parcel_NIRRGA_selected_parcels",
            # and delete parcels in parcel_NIRRGA that do not appear in parcel_NIRRGA_selected_parcels.
            Data_preprocessing_GT_selected_parcel.run_select_specific_parcels(
                frame_code=i)
        Data_preprocessing_GT_selected_parcel.cleanup()
    if rare_case_annotation_mask_folder is None:
        return frame_dataset_list
    else:
        return frame_dataset_list, rare_case_regions
