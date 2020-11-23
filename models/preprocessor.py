import warnings
warnings.simplefilter('ignore')

import sys
import pdb
import argparse
import json
import dill
import pickle
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import time

sys.path.append('/mnt/experiments/privacy-GAN/')
from data import *
from utils.generic import *
from utils.data import load_data_config, load_dataset
from models.encoders_decoders import LabelEncoderDecoder

class Preprocessor(object):
    """docstring for Preprocessor"""
    def __init__(self, config):
        super(Preprocessor, self).__init__()
        self.config = config
        self.std_scaler_path = join(self.config.CKPT_DIR, "standard_scaler.save")
        self.minmax_scaler_path = join(self.config.CKPT_DIR, "minmax_scaler.save")

    def scale(self, data):
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        joblib.dump(scaler, self.std_scaler_path)
        
        scaler2 = MinMaxScaler()
        data_scaled = scaler2.fit_transform(data)
        joblib.dump(scaler2, self.minmax_scaler_path)
        
        return data_scaled # returns numpy array

    def inverse_scale(self, data_scaled):
        scaler2 = joblib.load(self.minmax_scaler_path)
        data_unscaled = scaler2.inverse_transform(data_scaled)
        
        scaler = joblib.load(self.std_scaler_path)
        data = scaler.inverse_transform(data_unscaled)
        
        return data # returns numpy array

if __name__ == '__main__':
    from config import Config

    sample_config = Config('lacity', 'default')
    clean_data = load_dataset('lacity', 'clean')
    
    enc_dec_object = LabelEncoderDecoder(sample_config)
    encoded_data = enc_dec_object.encode(clean_data)

    preprocess_obj = Preprocessor(sample_config)
    pdb.set_trace()
    scaled = preprocess_obj.scale(encoded_data)
    inv_scaled = preprocess_obj.inverse_scale(scaled)


        