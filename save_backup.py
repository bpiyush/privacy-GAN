import os
import sys
sys.path.append('../')

dirpath = '/mnt/experiments/privacy-GAN'

os.system('tar cvfz /mnt/experiments/privacy-GAN.tar.gz {}'.format(dirpath))