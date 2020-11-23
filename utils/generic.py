import sys
import pdb
import json
import yaml
import argparse

sys.path.append('/mnt/experiments/privacy-GAN/')
from data import *


def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def read_yml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data

def load_model_config(model_version):
    filepath = join(HOME, 'models/config', '{}.yml'.format(model_version))
    return read_yml(filepath)

def save_common_config(model_version):
    folder = join(SAVE_ROOT, model_version)
    pass
    
def setup_argparse(description='Taking user inputs'):
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description=description)
    
    return parser

def get_parser_args(parser):
    argv = sys.argv[1:]
    return parser.parse_known_args(argv)[0]
