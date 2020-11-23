import warnings
warnings.simplefilter('ignore')

import sys
import pdb
import argparse

sys.path.append('/mnt/experiments/privacy-GAN/')
from data import *
from utils.generic import *
from utils.data import *


def seperate_real_and_categorical_columns(data):
    
    cols = data.columns
    num_cols = list(data._get_numeric_data().columns)
    non_num_cols = list(set(cols) - set(num_cols))

    num_data = data[num_cols]
    non_num_data = data[non_num_cols]
    data = pd.concat([num_data, non_num_data], axis=1)
    
    return data


def convert_target_column_to_categorical(target_data, basis='median'):
    
    if basis == 'median':
        mid_value = target_data.median()
    
    target_data_copy = target_data.copy()
    target_data.loc[target_data_copy <= mid_value]  = 'low'
    target_data.loc[target_data_copy > mid_value]  = 'high'
    
    return target_data
    

def seperate_out_target_variable(data_config, data, target_column):
    
    cols = list(data.columns)
    cols.pop(cols.index(target_column))
    non_target_data = data[cols]
    target_data = data[target_column]
    if data_config['target_variable_type'] == 'real':
        target_data = convert_target_column_to_categorical(target_data)
    new_data = pd.concat([non_target_data, target_data], axis=1)
    
    return new_data


def replace_nan_by_mean(dataset, num_real):
    # For numeric columns
    for i in range(num_real):
        col = dataset.iloc[:, i]
        col = col.fillna(col.mean())
        dataset.iloc[:, i] = col
    
    # For non-numeric columns
    for i in range(num_real, dataset.shape[1]):
        col = dataset.iloc[:, i]
        temp = (col.values).astype('str')
        val = np.unique(temp)
        if 'nan' in val:
            true_classes = list(set(val) - set(['nan']))
            num_classes = len(true_classes)
            indices = [i for i, value in enumerate(list(temp)) if value == 'nan']            
            to_replace_by = np.random.randint(low=0, high=num_classes, size=len(indices))
            dataset.iloc[indices, i] = to_replace_by
            
    return dataset


def clean_dataset(dataset_name):
    print("<---------- Cleaning {} dataset ---------->".format(dataset_name))
    data_config = load_data_config(dataset_name)
    raw_data = load_dataset(dataset_name, state='raw')
    
    print("=> Seperating numeric and non_numeric columns ...")
    data = seperate_real_and_categorical_columns(raw_data)

    print("=> Dropping direct identifiers ...")
    direct_identifiers = data_config['identifiers']
    data = data.drop(direct_identifiers, axis=1)
    
    print("=> Replacing NaN values by column mean ...")
    num_real = data_config['num_real']
    data = replace_nan_by_mean(data, num_real=num_real)

    print("=> Shifting the target column to the last index ...")
    target_column = data_config['target_column']
    data = seperate_out_target_variable(data_config, data, target_column)
        
    print("=> Saving cleaned dataset ...")
    save_filepath = join(DATASETS_PATH, dataset_name, "clean_data.csv")
    data.to_csv(save_filepath, index=False)
    
    return


if __name__ == '__main__':
    
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Taking user inputs.')
    parser.add_argument('-d', '--dataset_name', type=str, choices=['adult', 'lacity'], help='Dataset name', required=True)
    parser.add_argument('-s', '--save_subset', type=bool, help='Save only a subset of the cleaned dataset?', default=False)
    parser.add_argument('-n', '--num_in_subset', type=int, help='Number of examples in to-be-saved subset', default=18000)
    args = parser.parse_known_args(argv)[0]
    
    clean_dataset(args.dataset_name)
    if args.save_subset:
        save_subset_of_dataset(args.dataset_name, args.num_in_subset)
    

    
