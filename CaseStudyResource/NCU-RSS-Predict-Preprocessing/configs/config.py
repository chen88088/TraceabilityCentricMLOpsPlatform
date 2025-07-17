

IMG_Path = r"D:\TIF"
SHP_Path = r"D:\SHP"
workspace = r"C:\Users\chen88088\Documents\ArcGIS\Projects\PNGOutput\PNGOutput.gdb"
Tool_box = r"C:\program files\arcgis\pro\Resources\ArcToolbox\toolboxes\Conversion Tools.tbx"
directory = r"C:\Users\chen88088\Desktop"
sql_query = "Label_Num = 10112 or Label_Num = 10113"#10112跟10113分別代表水稻的成長期跟水稻的黃熟期。101代表水稻




Data_root_folder_path="/data"
NIRRG_path = directory + Data_root_folder_path +'/predict_NIRRG'
out_crop_mask = "/data/Predict_IMG_CROP"

save_polyline = directory+ Data_root_folder_path+'/polyline'
Train_Mask = directory + Data_root_folder_path + "/RSS13_Training_rice_mask"
parcel_path = directory + Data_root_folder_path + '/RSS15_Training_rice_mask'
Mask_Path = directory+Data_root_folder_path + "/Mask_rice"
target_phase = "phase23_mixed"#預計將哪些phase設為positive
field = "Label_Num"#shp中用來記錄作物類別的column
