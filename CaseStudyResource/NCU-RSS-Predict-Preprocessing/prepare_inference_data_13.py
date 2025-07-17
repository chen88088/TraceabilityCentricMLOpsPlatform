import os,sys
import shutil
import time
import logging
from logging.handlers import RotatingFileHandler
from configs.config import IMG_Path, NIRRG_path, out_crop_mask,Data_root_folder_path
from src.TIF2PNG import TIF2PNG
from src.crop_image import crop_image_tool
from PIL import Image
if not os.path.isdir('logs'):
    os.mkdir('logs')
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

def prepare_inference_data_13():
    logging.info("tif2png...")
    start = time.time()
    TIF2PNG(IMG_Path, NIRRG_path, 'NRG', enh=1)
    if os.path.isdir(out_crop_mask):
        shutil.rmtree(out_crop_mask)
    os.mkdir(out_crop_mask)
    logging.info("cropping_tif_for_inference...")
    crop_image_tool(imgs=NIRRG_path, out_path= out_crop_mask, resize_method=Image.LANCZOS)
    end = time.time()
    print(format(end-start))

if __name__ == '__main__':
    
    if os.path.isdir(Data_root_folder_path):
        shutil.rmtree(Data_root_folder_path)
    os.mkdir(Data_root_folder_path)
    

    prepare_inference_data_13()
