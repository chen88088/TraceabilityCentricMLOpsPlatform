import time

from src.write_prediction_label import write_prediction_label
from src.generate_average_mask import generate_average_mask
from src.generate_pgw import generate_pgw

if __name__ == '__main__':
    print("Start!!")

    start = time.time()

    # create pgws in raster_pgw 
    generate_pgw() 

    # copy prediction mask to raster_PGW_Path, and resize each prediction mask to fit corresponding tif
    # compte avg pixel value for each parcel, and store result at raster_Mask
    generate_average_mask()

    # convert avg pixel value to prediction label
    write_prediction_label()

    end = time.time()
    print("time:" + format(end - start))

    print("Finished!!")
