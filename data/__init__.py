import os
import sys
import pdb
import json
import yaml
import numpy as np
import pandas as pd
from glob import glob
from os.path import *

HOME = '/mnt/experiments/privacy-GAN'
DATASETS_PATH = '/mnt/datasets/'
DATASET_NAMES = ['adult', 'lacity'] 
SAVE_ROOT = '/mnt/outputs/'

