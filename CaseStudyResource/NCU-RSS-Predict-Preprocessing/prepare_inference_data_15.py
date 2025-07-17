import os,sys
import shutil
import time
import logging
from logging.handlers import RotatingFileHandler
from configs.config import IMG_Path, Mask_Path, NIRRG_path, Train_Mask,Data_root_folder_path,parcel_path,target_phase, directory
from src.rss15processing import rss15processing
from src.TIF2PNG import TIF2PNG
from src.generate_intermediate_mask import generate_intermediate_mask
from src.annotation_rice_processing import create_rice_mask
from src.resize_img import resize_image_tool
from PIL import Image

logs_path = directory + '/logs'
logs_file = logs_path + '/app-basic.log'
if not os.path.isdir(logs_path):
    os.mkdir(logs_path)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        # logging.FileHandler('logs/app-basic.log'),
        logging.handlers.TimedRotatingFileHandler(logs_file, when='midnight', interval=1, backupCount=30),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
def prepare_inference_data_15():
    logging.info("Start to generate mask...")
    generate_intermediate_mask()

    logging.info("Start to run excel_setting...")
    create_rice_mask(Mask_Path, Train_Mask, "1.5", target_phase)

    logging.info("Start to run rss14processing...")
    rss15processing(Train_Mask, parcel_path)
    resize_image_tool(imgs=parcel_path,resize_method=Image.NEAREST)
    

    logging.info("Start to convert TIFF to PNG...")
    TIF2PNG(IMG_Path, NIRRG_path, 'NRGI', enh=1)
    resize_image_tool(imgs=NIRRG_path,resize_method=Image.LANCZOS)
if __name__ == '__main__':
    
    if os.path.isdir(Data_root_folder_path):
        shutil.rmtree(Data_root_folder_path)
    os.makedirs(Data_root_folder_path)
    

    prepare_inference_data_15()
    print("Finished!!!")
