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
    crop_width = 512
    crop_height = 512
    file_count = 1
    for file in imgs:
        # read image and resize
        ic = 1
        im = Image.open(file)
        im = im.resize((width, height), resize_method)

        # get file name and make a folder for it
        

        print("crop ", file_count, ": ", file, "...")
        for i in range(0, width, crop_width):
            for r in range(0, height, crop_height):
                a = im.crop((i, r, i + crop_width, r + crop_height))
                filename =str(ic)*8+'_'+str(ic)*6+'z'
                a.save(out_path + r'/{}.png'.format(filename))
                ic = ic + 1
                if ic==3:break
            if ic==3:break
        file_count = file_count + 1
        break
if __name__=="__main__":
    print('Cropping smaller images for unit test')
    testing_data_dir='test/data'
    if not os.path.exists(testing_data_dir):os.mkdir(testing_data_dir)
    testing_data_dir='test/data/train_test'
    if not os.path.exists(testing_data_dir):os.mkdir(testing_data_dir)
    NRG_png_dir='test/data/train_test/NRG_png'
    if not os.path.exists(NRG_png_dir):os.mkdir(NRG_png_dir)
    parcel_mask_dir='test/data/train_test/parcel_mask'
    if not os.path.exists(parcel_mask_dir):os.mkdir(parcel_mask_dir)
    selected_parcel_mask_dir='test/data/train_test/selected_parcel_mask'
    if not os.path.exists(selected_parcel_mask_dir):os.mkdir(selected_parcel_mask_dir)
    # 94191004_181006z.png works
    crop_image_tool(imgs='data/train_test/NRG_png',out_path=NRG_png_dir,resize_method=Image.Resampling.LANCZOS)
    crop_image_tool(imgs='data/train_test/parcel_mask',out_path=parcel_mask_dir,resize_method=Image.NEAREST)
    crop_image_tool(imgs='data/train_test/selected_parcel_mask',out_path=selected_parcel_mask_dir,resize_method=Image.NEAREST)