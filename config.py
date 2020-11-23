from os.path import *
from os import makedirs
import pdb

import sys
sys.path.append('/mnt/experiments/privacy-GAN/')

from data import *
from utils.generic import load_model_config
from utils.data import load_data_config, load_dataset


class Config(object):
    """docstring for Config"""
    def __init__(self, dataset_name, model_version):
        super(Config, self).__init__()
        self.dataset_name = dataset_name
        self.model_version = model_version
        
        self.data_config = load_data_config(dataset_name)
        self.model_config = load_model_config(model_version)

        self.__dict__.update(self.data_config)
        self.__dict__.update(self.model_config)

        self.set_paths()


    def set_paths(self):
        self.HOME = HOME
        self.DATASETS_PATH = DATASETS_PATH
        self.SAVE_ROOT = SAVE_ROOT

        self.CKPT_DIR = join(self.SAVE_ROOT, self.model_version, 'checkpoints')
        self.LOG_DIR = join(self.SAVE_ROOT, self.model_version, 'logs')

        for x in [self.CKPT_DIR, self.LOG_DIR]:
            makedirs(x, exist_ok=True)



if __name__ == '__main__':
    sample_config = Config('adult', 'default')
    pdb.set_trace()
