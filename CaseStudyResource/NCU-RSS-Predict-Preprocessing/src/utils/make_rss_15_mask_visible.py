import glob,os
from PIL import Image
import numpy as np
masks=glob.glob('data/RSS15_Training_rice_mask/*.png')

for mask in masks:
    img_name=os.path.basename(mask).split('.')[0]
    
    im = np.array( Image.open(mask))
    im = np.where(im==1, 122, im)
    im = np.where(im==2, 255, im)
    im = Image.fromarray(im)
    if not os.path.exists("data/RSS15_Training_rice_mask_visible"):
        os.makedirs("data/RSS15_Training_rice_mask_visible")
    save_dest="data/RSS15_Training_rice_mask_visible"+'/'+img_name+'.png'
    
    im.save(save_dest)