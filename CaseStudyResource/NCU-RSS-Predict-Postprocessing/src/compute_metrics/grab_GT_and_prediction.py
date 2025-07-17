import glob
import os
import shutil

import arcpy
from configs.config import workspace, Prediction_SHP_Result_Path, Tool_box,class_field,pk_field,label_num_2_class,prediction_pickle_save_path
import logging
logger = logging.getLogger(__name__)
import pickle
def grab_GT_and_prediction(workspace: str, shp_file_path: str):
    # To allow overwriting outputs change overwriteOutput option to True.
    arcpy.env.overwriteOutput = True
    arcpy.ImportToolbox(Tool_box)
    # Model Environment settings
    with arcpy.EnvManager(scratchWorkspace=workspace,
                          workspace=workspace):
        my_layer = "layer_for_selection"
        arcpy.MakeFeatureLayer_management(shp_file_path, my_layer)
        logging.info("grab_GT_and_prediction for %s" % shp_file_path)

        arcpy.management.CalculateGeometryAttributes(shp_file_path, [["Shape_Area", "AREA_GEODESIC"]], "METERS", "SQUARE_METERS")

        try:
            gt_and_predictions= [
            row[:4] for row in
            arcpy.da.SearchCursor(my_layer, [pk_field,class_field,"prediction","Shape_Area"])
        ]
        except:
            raise  Exception("arcpy.da.SearchCursor failed, probably becauses %s or %s is not in the shp file."%(class_field,"prediction"))

        logging.info("len of gt_list %d" % len(gt_and_predictions))
       

       
        return gt_and_predictions

def main():
    shps=glob.glob(Prediction_SHP_Result_Path+'/*.shp')
    
    if os.path.exists(prediction_pickle_save_path):
        shutil .rmtree(prediction_pickle_save_path)
    os.mkdir(prediction_pickle_save_path)
    for shp in shps:
        gt_and_predictions=grab_GT_and_prediction(workspace=workspace,shp_file_path=shp)
        logging.info(shp)
        logging.info("first 10 (gt,prediction) looks like this:")
        logging.info(gt_and_predictions[:10])

        
        pks=[x[0] for x in gt_and_predictions ]
        gts=[x[1] for x in gt_and_predictions ]
        gts=[label_num_2_class[str(x)] for x in gts]
        
        predicts=[x[2] for x in gt_and_predictions ]

        true_label=label_num_2_class["10112"]
        Shape_Areas=[x[3] for x in gt_and_predictions ]
       
        
        gts=[1 if x==true_label else 0 for x in gts]
        predicts=[1 if x==true_label else 0 for x in predicts]
        # save to pickle
        save_name=os.path.join(prediction_pickle_save_path, os.path.basename(shp).split(".")[0])
        save_dict={"frame_name":save_name,"pks":pks,"gts":gts,"predicts":predicts,"Shape_Areas":Shape_Areas}
        with open(save_name+'.pickle', 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        


