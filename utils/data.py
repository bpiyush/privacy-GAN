import sys
import pdb

sys.path.append('/mnt/experiments/privacy-GAN/')
from data import *
from utils.generic import *


def load_data_config(dataset_name):
    filepath = join(HOME, 'data/config', '{}.yml'.format(dataset_name))
    return read_yml(filepath)


def load_dataset(dataset_name, state='raw'):
    
    filepath = join(DATASETS_PATH, dataset_name, '{}_data.csv'.format(state))
    data = pd.read_csv(filepath, index_col=False, low_memory=False)
    
    return data


def peek_into_dataset(dataset_name, state='raw'):
    data = load_dataset(dataset_name, state)
    print(data.head(3))
    return


def sample_subset_of_dataset(data, num_examples_reqd, seed=0):
    
    np.random.seed(seed)
    idx = np.random.randint(0, data.shape[0], num_examples_reqd)
    
    return data.iloc[idx, :]


def save_subset_of_dataset(dataset_name, num_examples_reqd):
    
    data = load_dataset(dataset_name, 'clean')
    
    print("=> Sampling a subset of {} examples from dataset ...".format(num_examples_reqd))
    data_subset = sample_subset_of_dataset(data, num_examples_reqd) 
    
    print("=> Saving the new subset dataset ...")
    save_filepath = join(DATASETS_PATH, dataset_name, "clean_subset_data.csv")
    data.to_csv(save_filepath, index=False)
    
    return
