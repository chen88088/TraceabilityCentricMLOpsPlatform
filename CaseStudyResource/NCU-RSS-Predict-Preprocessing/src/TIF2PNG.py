import glob
import os, logging
import arcpy as arcpy
from PIL import Image

import numpy as np

logger = logging.getLogger(__name__)


def TIF2PNG(input, output_path, itype, enh=0):
    logger.info('TIF to PNG')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if type(input) != list:
        files = glob.glob(input + '/*.tif')
    else:
        files = input

    for file in files:
        # logger.info('From: %s ', file)
        # img = np.array(Image.open(file))
        # width, height = Image.open(file).size
        img = arcpy.RasterToNumPyArray(file)
        logger.info('From: %s ', img.shape)

        logger.info('From: %s ', np.transpose(img, (1, 2, 0)).shape) 
        img = np.transpose(img, (1, 2, 0)) #change the order w,h,b
        height, width, band = img.shape

        # 3 BANDS
        if itype == 'NRG':
            i = 0
            result = np.empty([height, width, 3], dtype='uint8')
            for band in [3, 0, 1]:
                array = img[:, :, band]
                if enh == 1:
                    min, max = np.percentile(array, (2, 98))  # Image enhancement
                    ratio = (array - min) / (max - min)
                    array = np.maximum(np.minimum(ratio * 255, 255), 0).astype("uint8")
                result[:, :, i] = array
                i = i + 1
        else:
            # 4 BANDS
            result = np.empty([height, width, 4], dtype='uint8')
            for band in [0, 1, 2, 3]:
                array = img[:, :, band]
                if enh == 1:
                    min, max = np.percentile(array, (2, 98))  # Image enhancement
                    ratio = (array - min) / (max - min)
                    array = np.maximum(np.minimum(ratio * 255, 255), 0).astype("uint8")
                result[:, :, band] = array
        img = Image.fromarray(result)
        filename = os.path.basename(file).split('.')[0]
        filename1 = filename.split("_")
        img.save(output_path + '/' + filename1[0] + "_" + filename1[1] + '.png')
        logger.info('To: %s', output_path + '/' + os.path.basename(file).split('.')[0] + '.png')
