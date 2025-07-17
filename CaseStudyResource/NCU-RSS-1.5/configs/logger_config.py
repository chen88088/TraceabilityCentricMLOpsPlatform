# Created by AOIpc at 3/2/2023
import logging
from logging import config
import os
log_folder = os.path.join(os.getcwd(), 'data', 'logs')
if not os.path.exists(log_folder):
    os.mkdir(os.path.join(os.getcwd(), 'data', 'logs'))

LOGGING_CONFIG = {
    "version": 1,
    "loggers": {
        "root": {
            "level": "DEBUG",
            "propagate": False,
            "handlers": ["stream_handler"],
        },
        "train_info_log": {
            "level": "DEBUG",
            "propagate": False,
            "handlers": ["training_info"],
        },
        "train_debug_log": {
            "level": "DEBUG",
            "propagate": False,
            "handlers": ["training_debug"],
        },
        "parcel_info_log": {
            "level": "DEBUG",
            "propagate": False,
            "handlers": ["parcel_info", "stream_handler"],
        }
    },
    "handlers": {
        "stream_handler": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "level": "DEBUG",
            "formatter": "default_formatter",
        },
        "training_debug": {
            "class": "logging.FileHandler",
            "filename": "./data/logs/train_debug.log",
            "mode": "w",
            "level": "DEBUG",
            "formatter": "default_formatter",
        },
        "training_info": {
            "class": "logging.FileHandler",
            "filename": "./data/logs/train_info.log",
            "mode": "w",
            "level": "DEBUG",
            "formatter": "default_formatter",
        },
        "parcel_info": {
            "class": "logging.FileHandler",
            "filename": "./data/logs/parcel_generating_info.log",
            "mode": "w",
            "level": "DEBUG",
            "formatter": "default_formatter",
        }
    },
    "formatters": {
        "default_formatter": {
            "format": "%(asctime)s-%(levelname)s-%(name)s::%(module)s|%(lineno)s:: %(message)s",
        },
    },
}
config.dictConfig(LOGGING_CONFIG)
