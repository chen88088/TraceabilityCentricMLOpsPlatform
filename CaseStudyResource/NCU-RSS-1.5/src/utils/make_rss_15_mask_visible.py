import glob
import os
from PIL import Image
import numpy as np

path='D:/1111_work/git_repos/data_rss15/inference/parcel_mask_5'
masks=glob.glob(path+'/*.png')

for mask in masks:
    img_name=os.path.basename(mask).split('.')[0]

    im = np.array( Image.open(mask))
    im = np.where(im == 1, 122, im)
    im = np.where(im == 2, 255, im)
    im = Image.fromarray(im)
    if not os.path.exists(path+'/viz'):
        os.makedirs(path+'/viz')
    save_dest=path+'/viz' + '/'+img_name + '.png'

    im.save(save_dest)
