# logger init
import glob
import sys
import os
from logging.handlers import RotatingFileHandler
import logging
import time
if not os.path.isdir('./logs'):
    try:
        os.mkdir('./logs')
    except:
        raise Exception("error in os.mkdir")
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
# end logger init

from src.compute_metrics.grab_GT_and_prediction import main

if __name__ == '__main__':
    print("Start!!")

    start = time.time()

    main()
    end = time.time()
    print("time:" + format(end - start))

    print("Finished!!")
