import arcpy
import os
import glob
import shutil
import re
from configs.config import SHP_Path, Pred_Path, Prediction_SHP_Result_Path, MODEL_NAME
from datetime import datetime

shp_files = glob.glob(os.path.join(SHP_Path, '*.shp'))
# shp_files = [os.path.basename(shp) for shp in shp_files]
pred_files = os.listdir(Pred_Path)

if os.path.isdir(Prediction_SHP_Result_Path):
    shutil.rmtree(Prediction_SHP_Result_Path)
os.mkdir(Prediction_SHP_Result_Path)

for shp_file in shp_files:
    print(shp_file)
    # 創建一個空的 dict 來儲存 FID 對應 class 欄位的值
    pred_label_and_confidence_values = dict()
    
    frame_name = os.path.basename(shp_file).split('.')[0].split('PD_')[1]
    frame_date = re.sub('\D','',os.path.basename(shp_file).split("_")[2])
    taken_date = frame_date[2:4]+"/"+frame_date[4:]+"/"+'20'+frame_date[:2]
    print(taken_date)
    with open(os.path.join(Pred_Path, 'Pred_'+frame_name+'.txt'), 'r') as file:
        for line in file:
            fid, pred_label, pred_confidence = line.strip().split()
            pred_label_and_confidence_values[fid] = [pred_label, pred_confidence]
            # print(fid, pred_label, pred_confidence)

    shp_out_name = "Prediction_" + os.path.basename(shp_file).split('_')[1] + "_" + \
            os.path.basename(shp_file).split('_')[2]

    arcpy.MakeFeatureLayer_management(shp_file, shp_out_name)

    arcpy.management.AddField(in_table=shp_out_name, field_name="prediction",
                                  field_type="TEXT", field_is_nullable="NULLABLE")
    arcpy.management.AddField(in_table=shp_out_name , field_name="confidence", field_type="Double" )

    arcpy.management.AddField(in_table=shp_out_name, field_name="method",
                                  field_type="TEXT", field_is_nullable="NULLABLE")
    
    fields = [field.name for field in arcpy.ListFields(shp_file)]
    if 'Date' not in fields:
        arcpy.management.AddField(in_table=shp_out_name, field_name="Date",
                                  field_type="DATE", field_is_nullable="NULLABLE")

    with arcpy.da.UpdateCursor(shp_out_name, ['FID_Label', 'prediction', 'confidence', 'method','Label_Num','Date', 'Image_date', 'Image_type', 'Image_file']) as cursor:
        for row in cursor:
            fid = str(row[0])

            key = str(row[4])+f'{fid:0>4}'
            # if fid in pred_label_and_confidence_values:
            if fid in pred_label_and_confidence_values:

                if pred_label_and_confidence_values[fid][0] == '1':
                # if pred_label_and_confidence_values[key][0] == '1':
                    row[1] = '水稻'
                else :
                    row[1] = '非水稻'
                row[2] = float(pred_label_and_confidence_values[fid][1])
                # row[2] = float(pred_label_and_confidence_values[key][1])

                row[3] = MODEL_NAME # should be revised using the name of the model
                row[5] = datetime.now()
                row[6] = str(taken_date)
                row[7] = 'DMC3'
                row[8] = os.path.basename(shp_file).replace("PD_",'').replace(".shp",'.tif')


                cursor.updateRow(row)
            else:
                #if enter into this condition, means the polygon is empty
                pass


    arcpy.FeatureClassToShapefile_conversion(
            shp_out_name, Prediction_SHP_Result_Path)