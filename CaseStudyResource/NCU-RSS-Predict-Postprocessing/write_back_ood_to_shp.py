import arcpy
import os
import glob
import shutil
import numpy as np
from configs.config import workspace, Prediction_SHP_Result_Path, OPENMAX_PATH, Prediction_SHP_OOD_Path, Tool_box, directory, raster_PGW_Path, TIF_Path


shp_files = glob.glob(os.path.join(Prediction_SHP_OOD_Path, '*.shp'))
# shp_files = [os.path.basename(shp) for shp in shp_files]

if os.path.isdir(Prediction_SHP_OOD_Path):
    shutil.rmtree(Prediction_SHP_OOD_Path)
os.mkdir(Prediction_SHP_OOD_Path)

pred_label_and_confidence_values = dict()
frames = []
with open(OPENMAX_PATH, 'r') as file:
    for line in file:
        gt, filepath, openmax_score, pred = line.strip().split()
        frame = "_".join(filepath.split("_")[:5])
        fid = filepath.split("_")[-1].split(".")[0]
        frames.append(frame)
        pred_label_and_confidence_values[filepath.split(".")[0]] = [fid, pred, openmax_score]

for shp in np.unique(np.array(frames)) :
    keys = pred_label_and_confidence_values.keys()
    keys = [k for k in keys if shp in k]
    frame_name = "_".join(os.path.basename(shp).split('_')[:2])

    shp_file = os.path.join(Prediction_SHP_Result_Path,"Prediction_" + frame_name +".shp")
    print(shp_file)

    shp_out_name = os.path.join(Prediction_SHP_OOD_Path,"Prediction_" + frame_name + ".shp")

    arcpy.MakeFeatureLayer_management(shp_file, shp_out_name)

    arcpy.management.AddField(in_table=shp_out_name, field_name="rare",field_alias = "罕見坵塊",
                                  field_type="LONG", field_is_nullable="NULLABLE")

    with arcpy.da.UpdateCursor(shp_out_name, ['FID', 'rare']) as cursor:
    # with arcpy.da.UpdateCursor(shp_out_name, ['FID_Label', 'prediction', 'confidence', 'method']) as cursor:

        for row in cursor:
            fid = str(row[0])

            if pred_label_and_confidence_values[shp+"_"+fid][1]=='unknown':
                row[1] = 1
            else :
                row[1] = 0
            cursor.updateRow(row)
           
    arcpy.FeatureClassToShapefile_conversion(
            shp_out_name, Prediction_SHP_OOD_Path)