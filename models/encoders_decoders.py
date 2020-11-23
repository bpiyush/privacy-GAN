import warnings
warnings.simplefilter('ignore')

import sys
import pdb
import argparse
import json
import dill
import pickle
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import time

sys.path.append('/mnt/experiments/privacy-GAN/')
from data import *
from utils.generic import *
from utils.data import load_data_config, load_dataset


def split_dataset(data_stack, config):

    real_data = data_stack.iloc[:, :config.num_real]
    cat_data = data_stack.iloc[:, config.num_real:]

    return real_data, cat_data


def join_dataset(real_data, cat_data):

    data_stack = pd.concat([real_data, cat_data], axis=1)

    return data_stack


class LabelEncoderDecoder(object):
    """docstring for LabelEncoderDecoder"""
    def __init__(self, config):
        super(LabelEncoderDecoder, self).__init__()
        self.config = config
        # self.dataset_name = config.dataset_name
        # self.clean_data = load_dataset(self.dataset_name, 'clean')
        self.num_real = self.config.num_real

    def encode(self, clean_data):
        data_stack = clean_data.copy()

        # splitting real and categorical columns
        real_data, cat_data = split_dataset(data_stack, self.config)

        # encoding
        d = defaultdict(LabelEncoder)
        cat_data = cat_data.apply(lambda x: d[x.name].fit_transform(x.astype('str')))
        
        # writing encoder
        encoder_filepath = join(self.config.CKPT_DIR, "label_encoder.pickle")
        with open(encoder_filepath, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

        data_enc = join_dataset(real_data, cat_data)

        return data_enc


    def decode(self, data_enc):

        # splitting real and categorical columns
        real_data, cat_data = split_dataset(data_enc, self.config)

        # reading encoder
        encoder_filepath = join(self.config.CKPT_DIR, "label_encoder.pickle")
        with open(encoder_filepath, 'rb') as handle:
            d = pickle.load(handle)
            
        # inverse encoding
        for cat_col in cat_data.columns:

            column_values = d[cat_col].transform(d[cat_col].classes_)
            
            max_cond = (cat_data[cat_col] >= max(column_values))
            cat_data.loc[max_cond, cat_col] = max(column_values)

            min_cond = (cat_data[cat_col] <= min(column_values))
            cat_data.loc[min_cond, cat_col] = min(column_values)
            
            cat_data.loc[:, cat_col] = cat_data[cat_col] + 0.5
            cat_data.loc[:, cat_col] = cat_data[cat_col].astype(int)
            
            cat_data.loc[:, cat_col] = d[cat_col].inverse_transform(cat_data[cat_col])
            
        data_stack = join_dataset(real_data, cat_data)

        return data_stack


class OneHotEncoderDecoder(object):
    """docstring for OneHotEncoderDecoder"""
    def __init__(self, config):
        super(OneHotEncoderDecoder, self).__init__()
        self.config = config
        self.num_real = self.config.num_real
        self.encoder_filepath = join(self.config.CKPT_DIR, "onehot_encoder.pickle")
    
    def encode(self, data):
        
        ## Easier and working code for OHE
        data = data.values

        data_real = data[:,:self.num_real]
        data_cat = data[:, self.num_real:]

        for i in tqdm(range(data_cat.shape[1]), total=(data_cat.shape[1])):
            enc = LabelEncoder()
            data_cat[:,i] = enc.fit_transform(data_cat[:, i])

        ohe = OneHotEncoder()
        data_cat_ohe = ohe.fit_transform(data_cat).toarray()

        with open(self.encoder_filepath, 'wb') as handle:
            pickle.dump(ohe, handle)

        data_real = pd.DataFrame(data_real)
        data_cat_ohe = pd.DataFrame(data_cat_ohe)

        data_oe = pd.concat([data_real, data_cat_ohe], axis=1)
        
        return data_oe

    def decode(self, data, data_enc):
        real_data = data_enc.iloc[:, :self.num_real]
        cat_data = data_enc.iloc[:, self.num_real:]
    
        X = pd.get_dummies(data.iloc[:, self.num_real:])
        cat_data.columns = list(X.columns.astype(str))
        
        cat_data = self.reverse_dummy(cat_data)

        # combining the columns back
        data_stack = pd.concat([real_data, cat_data], axis=1, ignore_index=True)
        
        return data_stack   
    
    def reverse_dummy(self, df_dummies):
        pos = defaultdict(list)
        vals = defaultdict(list)

        for i, c in tqdm(enumerate(df_dummies.columns), total=len(df_dummies.columns)):
            if "_" in c:
                k, v = c.split("_", 1)
                pos[k].append(i)
                vals[k].append(v)
            else:
                pos["_"].append(i)

        df = pd.DataFrame({k: pd.Categorical.from_codes(
                                  np.argmax(df_dummies.iloc[:, pos[k]].values, axis=1),
                                  vals[k])
                          for k in vals})

        df[df_dummies.columns[pos["_"]]] = df_dummies.iloc[:, pos["_"]]
        
        return df

        


if __name__ == '__main__':
    from config import Config
    
    parser = setup_argparse()
    parser.add_argument('-d', '--dataset_name', default='adult')
    parser.add_argument('-e', '--enc_dec', default='le')
    parser.add_argument('-s', '--subset', type=bool, default=False)
    args = get_parser_args(parser)

    EncoderDecoder = LabelEncoderDecoder
    if args.enc_dec == 'ohe':
        EncoderDecoder = OneHotEncoderDecoder
    
    sample_config = Config(args.dataset_name, 'default')
    data_version = 'clean'
    if args.subset:
        assert args.dataset_name != 'adult'
        data_version = 'clean_subset'
    clean_data = load_dataset(args.dataset_name, data_version)

    enc_dec_object = EncoderDecoder(sample_config)
    encoded_data = enc_dec_object.encode(clean_data)
    pdb.set_trace()
    
    if args.enc_dec == 'ohe':
        decoded_data = enc_dec_object.decode(clean_data, encoded_data)
    elif args.enc_dec == 'le':
        decoded_data = enc_dec_object.decode(encoded_data)
        
    pdb.set_trace()

    
        