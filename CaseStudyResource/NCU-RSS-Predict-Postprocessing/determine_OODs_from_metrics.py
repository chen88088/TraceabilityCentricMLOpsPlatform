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
from configs.config import metrics_path,OOD_cut_point_params,OOD_detection_method,OOD_threshold_params,OOD_z_score_params
def find_sudden_slope_change(numbers, threshold):
    """
    Find the index where the change in slope is sudden in a list of sorted numbers.

    Parameters:
    - numbers: A list of sorted numbers in the range [0, 1].
    - threshold: A threshold value to determine what constitutes a sudden change.

    Returns:
    - The index of the sudden slope change, or None if no sudden change is found.
    """
    for i in range(1, len(numbers)):
        slope_change = abs(numbers[i]- numbers[i - 1])
        if slope_change >= threshold:
            return i
    return None
def detect_outliers_z_score(data, threshold=3):
    # data: (frame_name,metric)
    values=[x[1] for x in data]
    
    mean = np.mean(values)
    std_dev = np.std(values)
    
    z_scores = [(x - mean) / std_dev for x in values]
    
    outliers = [(data[i][0],z_scores[i]) for i, z in enumerate(z_scores) if abs(z) > threshold]
    
    return outliers


def detect_outliers_threshold(data, threshold=0.8):
    # data: (frame_name,metric)
   
    outliers = [data[i] for i in range(len(data)) if data[i][1] < threshold]
    
    return outliers
def main():
    # read csv
    df = pd.read_csv(metrics_path)
    frame_name=df['frame_name']
    acc =df["acc"]
    kappa =df["kappa"]
    f1_score=df['f1_score']
    

    # group framenames and target metrics
    frame_name_acc=[]
    for i in range(len(frame_name)):
        frame_name_acc.append((frame_name[i],acc[i]))

    frame_name_kappa=[]
    for i in range(len(frame_name)):
        frame_name_kappa.append((frame_name[i],kappa[i]))
    
    frame_name_f1_score=[]
    for i in range(len(frame_name)):
        frame_name_f1_score.append((frame_name[i],f1_score[i]))

    
    # detect ood
    if OOD_detection_method=='z_score':
        if OOD_z_score_params[0]=='acc':
            all_frames=detect_outliers_z_score(frame_name_acc,0)
            oods=detect_outliers_z_score(frame_name_acc,OOD_z_score_params[1])
        if OOD_z_score_params[0]=='kappa':
            all_frames=detect_outliers_z_score(frame_name_kappa,0)
            oods=detect_outliers_z_score(frame_name_kappa,OOD_z_score_params[1])
        if OOD_z_score_params[0]=='f1_score':
            all_frames=detect_outliers_z_score(frame_name_f1_score,0)
            oods=detect_outliers_z_score(frame_name_f1_score,OOD_z_score_params[1])
        # print metrics
        logging.info('frame : z_score')
        for i in range(len(all_frames)):
            logging.info('%s : %f'%(all_frames[i][0],all_frames[i][1]))
    
    elif OOD_detection_method=='threshold':
        # print metrics
        logging.info('frame : acc kappa f1_score')
        for i in range(len(frame_name_acc)):
            logging.info('%s : %f %f %f'%(frame_name_acc[i][0],frame_name_acc[i][1],frame_name_kappa[i][1],frame_name_f1_score[i][1]))
    
        if OOD_threshold_params[0]=='acc':
            oods=detect_outliers_threshold(frame_name_acc,OOD_threshold_params[1])
        if OOD_threshold_params[0]=='kappa':
            oods=detect_outliers_threshold(frame_name_kappa,OOD_threshold_params[1])
        if OOD_threshold_params[0]=='f1_score':
            oods=detect_outliers_threshold(frame_name_f1_score,OOD_threshold_params[1])
    
    elif OOD_detection_method=='cut_point':
        if OOD_cut_point_params[0]=='acc':
            numbers=frame_name_acc
        if OOD_cut_point_params[0]=='kappa':
            numbers=frame_name_kappa
        if OOD_cut_point_params[0]=='f1_score':
            numbers=frame_name_f1_score
        sorted_frame_metric= sorted(numbers,key=lambda a : a[1],reverse=True )
        # print metrics
        logging.info('frame : metric')
        for i in range(len(sorted_frame_metric)):
            logging.info('%s : %f'%(sorted_frame_metric[i][0],sorted_frame_metric[i][1]))
    
        threshold_value = OOD_cut_point_params[1]
        sorted_numbers=[x[1]for x in sorted_frame_metric]
        index_of_sudden_change = find_sudden_slope_change(sorted_numbers, threshold_value)

        if index_of_sudden_change is not None:
            
            logging.info(f"Sudden change detected at index {index_of_sudden_change}")
            oods=sorted_frame_metric[index_of_sudden_change:]
        else:
            print("No sudden change detected.")
            oods=[]
    # print ood
    logging.info('___\nlisting oods')
    for f in frame_name_acc:
        is_ood=False
        for ood in oods:
            if f[0]in ood:is_ood=True
        logging.info('%s : %d'%(f[0],is_ood))
   
if __name__ == '__main__':
    main()