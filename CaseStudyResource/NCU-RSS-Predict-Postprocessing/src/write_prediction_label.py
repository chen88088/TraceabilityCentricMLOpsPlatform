import glob
import os
import shutil
import arcpy

from configs.config import raster_Mask_Path, Prediction_SHP_Result_Path, sql_query, \
Predict_False_Lable, Predict_True_Lable, debug


def write_prediction_label():
    All_SHP_Path = raster_Mask_Path + "/*.shp"
    shp_files = glob.glob(All_SHP_Path)
    if os.path.isdir(Prediction_SHP_Result_Path):
        shutil.rmtree(Prediction_SHP_Result_Path)
    os.mkdir(Prediction_SHP_Result_Path)
    for shp_file in shp_files:
        shp_file_name = format(os.path.basename(shp_file).split('.')[0])
        shp_out_name = "Prediction_" + os.path.basename(shp_file_name).split('_')[1] + "_" + \
            os.path.basename(shp_file_name).split('_')[2]

        arcpy.MakeFeatureLayer_management(shp_file, shp_out_name)

        # Create prediction field according to MEAN
        arcpy.management.AddField(in_table=shp_out_name, field_name="prediction",
                                  field_type="TEXT", field_is_nullable="NULLABLE")
        arcpy.CalculateField_management(
            shp_out_name, "prediction", Predict_False_Lable, "PYTHON3")
        arcpy.SelectLayerByAttribute_management(
            shp_out_name, "NEW_SELECTION", sql_query)
        arcpy.CalculateField_management(
            shp_out_name, "prediction", Predict_True_Lable, "PYTHON3")

        # create field c0, where c0=mean if prediction== Predict_True_Lable else 255-mean
        arcpy.management.AddField(
            in_table=shp_out_name, field_name="c0", field_type="Double")
        Predict_True_sql_query = "prediction = %s" % Predict_True_Lable
        arcpy.SelectLayerByAttribute_management(
            shp_out_name, "NEW_SELECTION", Predict_True_sql_query)
        expression = "!MEAN!/255"
        arcpy.CalculateField_management(
            shp_out_name, "c0", expression, "PYTHON3")

        Predict_False_sql_query = "prediction = %s" % Predict_False_Lable
        arcpy.SelectLayerByAttribute_management(
            shp_out_name, "NEW_SELECTION", Predict_False_sql_query)
        expression = "(255-!MEAN!)/255"
        arcpy.CalculateField_management(
            shp_out_name, "c0", expression, "PYTHON3")
        
        # confidence=c0
        arcpy.management.AddField(in_table=shp_out_name , field_name="confidence", field_type="Double" )
        expression = "!c0!"
        _sql_query = "FID > -1" 
        arcpy.SelectLayerByAttribute_management(
            shp_out_name, "NEW_SELECTION", _sql_query)
        arcpy.CalculateField_management(shp_out_name ,"confidence", expression, "PYTHON3")
        # save
        arcpy.SelectLayerByAttribute_management(
            shp_out_name, "NEW_SELECTION", "FID IS NOT NULL")
        if not debug:
            arcpy.DeleteField_management(shp_out_name, [ "MEAN","c0"])
        arcpy.FeatureClassToShapefile_conversion(
            shp_out_name, Prediction_SHP_Result_Path)
