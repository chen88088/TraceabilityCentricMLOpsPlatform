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

from src.compute_metrics.compute_metrics import compute_kappa,compute_acc, compute_f1,confusion 
from configs.config import metrics_path,prediction_pickle_save_path,false_entries_path
import numpy as np

def cohen_kappa(confusion_matrix):
    observed_agreement = np.trace(confusion_matrix)
    
    total_samples = np.sum(confusion_matrix)
    
    row_sum = np.sum(confusion_matrix, axis=0)
    column_sum = np.sum(confusion_matrix, axis=1)
    
    expected_agreement = np.sum(row_sum * column_sum) / (total_samples ** 2)
    
    kappa = (observed_agreement - expected_agreement) / (total_samples - expected_agreement)
    
    return kappa
def parcel_based_kappa_with_area_weights(gts,preds,weights):
    TP_amount = 0  # 11
    TN_amount = 0  # 10
    FP_amount = 0  # 01
    FN_amount = 0  # 00

    for i in range(len(gts)):
        to_add=weights[i]/10000
        if   gts[i]==0 and preds[i]==0:TN_amount+=to_add
        elif gts[i]==0 and preds[i]==1:FP_amount+=to_add
        elif gts[i]==1 and preds[i]==0:FN_amount+=to_add
        elif gts[i]==1 and preds[i]==1:TP_amount+=to_add
    c_m_with_area_weights = np.array([
        [TP_amount, FP_amount],
        [FN_amount, TN_amount]
    ])
    return cohen_kappa(c_m_with_area_weights)
            

if __name__ == '__main__':
    frame_dicts_paths=glob.glob(prediction_pickle_save_path+"/*.pickle")
    if os.path.exists(false_entries_path):
        shutil .rmtree(false_entries_path)
    os.mkdir(false_entries_path)
    
    # lists that saves results to later write to csv
    frame_name_list=[]
    kappa_list=[]
    kappa_w_area_list=[]
    f1_score_list=[]
    acc_list=[]

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
        Shape_Areas=frame_dict['Shape_Areas']
        assert(len(gts)==len(predicts))
        
        kappa=compute_kappa(gts,predicts)
        logging.info("kappa : %f"%kappa)
        kappa_list.append(kappa)

        _parcel_based_kappa_with_area_weights=parcel_based_kappa_with_area_weights(gts,predicts,Shape_Areas)
        logging.info("parcel_based_kappa_with_area_weights : %f"%_parcel_based_kappa_with_area_weights)
        kappa_w_area_list.append(_parcel_based_kappa_with_area_weights)

        acc=compute_acc(gts,predicts)
        logging.info("accuracy_score : %f"%acc)
        acc_list.append(acc)

        f1_score=compute_f1(gts,predicts)
        logging.info("f1_score : %f"%f1_score)
        f1_score_list.append(f1_score)
         

        matrix = confusion(gts,predicts)
        str_matrix = "["
        for x in matrix.ravel():
            #print(x)
            str_matrix=str_matrix+" "+str(x)
        str_matrix+="]"
        logging.info(str_matrix)

        # save predictions
        df = pd.DataFrame({'parcel_nums': pks,
                    'labels': gts,
                    'predictions':predicts})
    
        df.to_csv(os.path.join(os.path.join(false_entries_path,'%s_predictions.csv'%frame_name)))

        # saving fps, fns
        
        with open ( os.path.join(false_entries_path,'%s_fps_fns.txt'%frame_name),'w')as fh:
            fh.write('pk\tlabel\tpredict\terror_type\n')

            for i in range(len(gts)):
                if gts[i]!=predicts[i]:
                    error_type='fn'
                    if  gts[i]==0:
                        error_type='fp'
                    fh.write("%s\t%s\t%s\t%s\n"%(pks[i],gts[i],predicts[i],error_type))
    
    
    metrics_path_base=metrics_path.split('/')[0]
    if not os.path.exists(metrics_path_base):os.mkdir(metrics_path_base)
    df = pd.DataFrame({'frame_name': frame_name_list,
                    'acc': acc_list,
                    'kappa':kappa_list,
                    'kappa_w_area_weights':kappa_w_area_list,
                    'f1_score':f1_score_list})

    print(os.path.join(metrics_path))
    df.to_csv(os.path.join(metrics_path))

    

    