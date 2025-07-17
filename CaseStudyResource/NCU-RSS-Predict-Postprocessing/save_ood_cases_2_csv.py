# logger init
import glob
import pickle
import shutil
import sys
import os
import numpy as np
from logging.handlers import RotatingFileHandler
import logging
import time
from compute_metrics import parcel_based_kappa_with_area_weights
import pandas as pd
if not os.path.isdir('./logs'):
    try:
        os.mkdir('./logs')
    except:
        raise Exception("error in os.mkdir")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        # logging.FileHandler('logs/app-basic.log'),
        logging.handlers.TimedRotatingFileHandler('logs/app-basic.log', when='midnight', interval=1, backupCount=30),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
# end logger init

from configs.config import prediction_pickle_save_path,known_ood_path

def show_hypothetic_kappa_when_oods_are_all_wrong(gts,ood_cases,shape_areas,predicts):
        # computing parcel based kappa when all ood are misclassified, and all id are correctly classified
    hypothetical_pred=list(gts)
    for i in range(len(gts)):
        if ood_cases[i]>=1:
            if gts[i]==0:
                hypothetical_pred[i]=1
            elif gts[i]==1:
                hypothetical_pred[i]=0
    p_kappa_w_area_weights=parcel_based_kappa_with_area_weights(gts,hypothetical_pred,shape_areas)
    print("p_kappa_w_area_weights %f"%(p_kappa_w_area_weights))

    # computing the intersection of ood and misclassification
    misclassifications=[]
    for i in range(len(predicts)):
        if predicts[i]!=gts[i]:misclassifications.append(i)
    
    
    oods=[]
    for i in range(len(ood_cases)):
        if ood_cases[i]>=1:oods.append(i)

    intersection=set(misclassifications).intersection(oods)
    print("intersection of ood and misclassification / misclassification=%f"%(len(intersection)/len(misclassifications)))

if __name__ == '__main__':
    frame_dicts_paths=glob.glob(prediction_pickle_save_path+"/*.pickle")
    if os.path.exists(known_ood_path):
        shutil .rmtree(known_ood_path)
    os.mkdir(known_ood_path)
    
    # lists that saves results to later write to csv
    frame_name_list=[]
  

    for frame_dicts_path in frame_dicts_paths:
        with open(frame_dicts_path, 'rb') as handle:
             frame_dict = pickle.load(handle)

        frame_name=frame_dict["frame_name"]
        frame_name=os.path.basename(frame_name)
        logging.info("frame_name : %s"%frame_name)
        frame_name_list.append(frame_name)

        pks=frame_dict['pks']
        gts=frame_dict["gts"]
        predicts=frame_dict["predicts"]
        ood_cases=frame_dict['ood_cases']
        shape_areas=frame_dict['Shape_Areas']
        assert(len(gts)==len(predicts))
        
        # saving known ood
        
        with open ( os.path.join(known_ood_path,'%s_known_oods.txt'%frame_name),'w')as fh:
            fh.write('pk\tlabel\tpredict\tOOD_case\n')

            for i in range(len(gts)):
                if ood_cases[i]!=0:
                    
                    fh.write("%s\t%s\t%s\t%s\n"%(pks[i],gts[i],predicts[i],ood_cases[i]))

        #show_hypothetic_kappa_when_oods_are_all_wrong(gts,ood_cases,shape_areas,predicts)
    

    