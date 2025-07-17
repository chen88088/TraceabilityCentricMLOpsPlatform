import glob
import os
import shutil

import arcpy
import numpy as np
from arcpy import env
from PIL import Image

from configs.config import workspace, TIF_Path, directory, raster_PNG_Path, raster_PGW_Path


def create_folder():
    if os.path.isdir(raster_PNG_Path):
        shutil.rmtree(raster_PNG_Path)
    os.mkdir(raster_PNG_Path)

    if os.path.isdir(raster_PGW_Path):
        shutil.rmtree(raster_PGW_Path)
    os.mkdir(raster_PGW_Path)


def create_pgw():
    All_IMG_Path = directory + "/" + TIF_Path + "/*.tif"
    files = glob.glob(All_IMG_Path)
    
    for file in files:
        print(file)
        output_dir = directory + "/" + raster_PNG_Path
        arcpy.RasterToOtherFormat_conversion(file, output_dir, "PNG")
        
    __raster_PGW_Path_regex = directory + "/" + raster_PNG_Path + "/*.pgw"
    __raster_AUX_Path_regex = directory + "/" + raster_PNG_Path + "/*"
    pgws = glob.glob(__raster_PGW_Path_regex)
    for pgw in pgws:
        shutil.move(pgw, raster_PGW_Path)

        filename = format(os.path.basename(pgw).split('.')[0])
        pgwname = filename.split("_")

        old_img = directory + "/" + raster_PGW_Path + "/" + filename + ".pgw"
        new_img = directory + "/" + raster_PGW_Path + "/" + pgwname[0] + "_" + pgwname[1] + ".pgw"
        os.rename(old_img, new_img)
    auxs = glob.glob(__raster_AUX_Path_regex)
    auxs =[x for x in auxs if '.png.aux' in x]
    for aux in auxs:
        shutil.move(aux, raster_PGW_Path)

        filename = format(os.path.basename(aux).split('.')[0])
        auxname = filename.split("_")

        old_img = directory + "/" + raster_PGW_Path + "/" + filename + ".png.aux.xml"
        new_img = directory + "/" + raster_PGW_Path + "/" + auxname[0] + "_" + auxname[1] + ".png.aux.xml"
        os.rename(old_img, new_img)

def generate_pgw():
    env.workspace = workspace
    create_folder()
    create_pgw()
   
