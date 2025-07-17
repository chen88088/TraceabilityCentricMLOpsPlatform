import glob
import os


from PIL import Image

def resize_image_tool(imgs=None, resize_method=None):
    imgs = glob.glob(imgs + '/*.png')
    print('\nCrop Image')


    width = 11460
    height = 12260
    
    num =0
    for file in imgs:
        
        num = num+1
        # read image and resize
        
        im = Image.open(file)
        im = im.resize((width, height), resize_method)

        im.save(file)
        print (f"finish image{num} resizing !" )
