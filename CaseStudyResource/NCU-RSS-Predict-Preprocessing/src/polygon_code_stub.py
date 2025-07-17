import glob,os
from configs.config import IMG_Path
def polygon_code_stub():
    #NCU_94191004_181006z_14~4907_hr4_0000
    l=list()
    all_tifs=glob.glob(IMG_Path+'/*.tif')
    for t in all_tifs:
        _base_name=os.path.basename(t)
        l.append("NCU_"+_base_name+"_0000")
    s=set(l)
    return s