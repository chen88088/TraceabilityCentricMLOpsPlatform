import glob
import os
import shutil

import arcpy
from configs.config import workspace, Prediction_SHP_Result_Path, Tool_box,class_field,pk_field,label_num_2_class,prediction_pickle_save_path
import logging
logger = logging.getLogger(__name__)
import pickle
def grab_ood_cases(workspace: str, shp_file_path: str):
    # To allow overwriting outputs change overwriteOutput option to True.
    arcpy.env.overwriteOutput = True
    arcpy.ImportToolbox(Tool_box)
    # Model Environment settings
    with arcpy.EnvManager(scratchWorkspace=workspace,
                          workspace=workspace):
        my_layer = "layer_for_selection"
        arcpy.MakeFeatureLayer_management(shp_file_path, my_layer)
        try:
            ood_cases= [
            row[:5] for row in
            arcpy.da.SearchCursor(my_layer, [pk_field,class_field,"prediction","OOD_case","Shape_Area"])
        ]
        except:
            raise  Exception("arcpy.da.SearchCursor failed, probably becauses %s or %s is not in the shp file."%(class_field,"prediction"))

        logging.info("len of gt_list %d" % len(ood_cases))
       

       
        return ood_cases

def main_grab_ood_case():
    shps=glob.glob(Prediction_SHP_Result_Path+'/*.shp')
    
    if os.path.exists(prediction_pickle_save_path):
        shutil .rmtree(prediction_pickle_save_path)
    os.mkdir(prediction_pickle_save_path)
    for shp in shps:
        gt_and_predictions=grab_ood_cases(workspace=workspace,shp_file_path=shp)
        logging.info(shp)
        logging.info("first 10 (gt,prediction) looks like this:")
        logging.info(gt_and_predictions[:10])

        
        pks=[x[0] for x in gt_and_predictions ]
        gts=[x[1] for x in gt_and_predictions ]
        gts=[label_num_2_class[str(x)] for x in gts]

        
        predicts=[x[2] for x in gt_and_predictions ]

        true_label=label_num_2_class["10112"]
        ood_cases=[x[3] for x in gt_and_predictions ]
        shape_areas=[x[4] for x in gt_and_predictions]
        gts=[1 if x==true_label else 0 for x in gts]
        predicts=[1 if x==true_label else 0 for x in predicts]
        # save to pickle
        save_name=os.path.join(prediction_pickle_save_path, os.path.basename(shp).split(".")[0])
        save_dict={"frame_name":save_name,"pks":pks,"gts":gts,"predicts":predicts,"ood_cases":ood_cases,"Shape_Areas":shape_areas}
        with open(save_name+'.pickle', 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        


