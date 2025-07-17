# 更改下方路徑即可 Need to change the path
# workspace:  ArcGIS project .gdb path
# Tool_box:   ArcGIS toolboxes .tbx path
# directory:  Github Download Project path
# sql_query:  set what to classify crop

import os


workspace = r"C:\Users\AOIpc\Documents\ArcGIS\Projects\MidTerm Report\MidTerm Report.gdb"
Tool_box = r"C:\Users\AOIpc\AppData\Local\Programs\ArcGIS\Pro\Resources\ArcToolBox\toolboxes\Conversion Tools.tbx"
directory = r"D:\RSS-1.5_code\NCU-RSS-Predict-Postprocessing"
Predict_True_Lable = "\'水稻\'"
Predict_False_Lable = "\'非水稻\'"
SHP_Path = r"SHP"
IMG_Path = r'IMG'
TIF_Path = r'TIF'
Pred_Path = r'PRED'
OPENMAX_PATH = r'OPENMAX\test_openset_scores.txt' # the OOD detection prediction
MODEL_NAME = r"NCU-RSS-1.6-TW-0616" # please rename as the model used to make the prediction
#--------------------------DON'T CHANGE anything below----------------------------------#
debug=False
sql_query = "MEAN > 123"
raster_Mask_Path =   directory +"/raster_Mask"
raster_PNG_Path = r'raster_PNG'
raster_PGW_Path = r'raster_PGW'
Prediction_SHP_Result_Path = directory + '/Prediction_SHP_Result'
Prediction_SHP_OOD_Path = directory + '/Prediction_SHP_Result_ood'
class_field = 'Label_Num'
pk_field='VPC'
label_num_2_class={
            "10112":"水稻",
            "10113":"水稻",
            "10114":"非水稻",
            "10200":"非水稻",
            "90100":"非水稻",
            "90200":"非水稻",
            "99900":"非水稻", 
            "0":"非水稻"    
        }


prediction_pickle_save_path= os.path.join (directory , 'prediction_pickle')
false_entries_path=os.path.join (directory, 'fps_fns')
known_ood_path=os.path.join (directory, 'known_oods')
metrics_path=os.path.join (directory, 'metrics/metrics.csv')
OOD_detection_method='threshold'#['z_score','threshold','cut_point']
OOD_threshold_params=('acc',0.85)#[('f1_score',0.8),
                                # ('acc',0.9),
                                # ('kappa',0.8)]
OOD_z_score_params = ('acc',1.8)  #[('f1_score',2),
                                # ('acc',2),
                                # ('kappa',2)]   
OOD_cut_point_params = ('acc',0.04) 



