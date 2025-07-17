import glob
import os
import shutil

import numpy as np
from PIL import Image

def crop_image_tool(imgs=None, out_path=None, resize_method=None):
    imgs = glob.glob(imgs + '/*.png')
    print('\nCrop Image')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    width = 11520
    height = 12288
    crop_width = 256
    crop_height = 256
    file_count = 1
    for file in imgs:
        # read image and resize
        ic = 1
        im = Image.open(file)
        im = im.resize((width, height), resize_method)

        # get file name and make a folder for it
        filename = os.path.basename(file).split('.')[0]

        print("crop ", file_count, ": ", file, "...")
        for i in range(0, width, crop_width):
            for r in range(0, height, crop_height):
                a = im.crop((i, r, i + crop_width, r + crop_height))
                a.save(out_path + r'/{}_{}.png'.format(ic, filename))
                ic = ic + 1
        file_count = file_count + 1
