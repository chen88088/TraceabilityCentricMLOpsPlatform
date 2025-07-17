"""Module to generating config for the training and testing"""
import json


class Config:
    """Config class which contains data, train and model hyperparameters"""

    def __init__(self, data_generation, kmeans, random_sampling,
                 train_and_val, testing, inference):
        self.data_generation = data_generation
        self.kmeans = kmeans
        self.random_sampling = random_sampling
        self.train_and_val = train_and_val
        self.testing = testing
        self.inference = inference

    @classmethod
    def from_json(cls, cfg):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data_generation, params.kmeans, params.random_sampling,
                   params.train_and_val, params.testing, params.inference)

    @classmethod
    def from_commandline(cls, cfg, args):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data_generation, params.kmeans, params.random_sampling,
                   params.train_and_val, params.testing, params.inference)


class HelperObject(object):
    """Helper class to convert json into Python object"""

    def __init__(self, dict_):
        self.__dict__.update(dict_)
