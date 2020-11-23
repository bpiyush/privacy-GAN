import warnings
warnings.simplefilter('ignore')

import sys
import math
import pdb
import tensorflow as tf

sys.path.append('/mnt/experiments/privacy-GAN/')
from utils.model import lrelu, relu
from utils.generic import *


class Discriminator(object):
    """docstring for Discriminator"""
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config

        self.kernel_size = config.kernel_size
        self.filters = config.filters
        self.strides = config.strides
        self.activation = eval(config.activation)
        self.momentum = config.momentum
        self.noise_dim = config.noise_dim

    def batch_norm(self, _input, is_training):
        return tf.contrib.layers.batch_norm(_input, is_training=is_training, decay=self.momentum)

    def dropout(self, _input, keep_prob):
        return tf.layers.dropout(_input, keep_prob)

    def discriminate(self, img_in, img_dim, keep_prob, reuse=None):

        with tf.variable_scope("discriminator", reuse=reuse):
            d1 = img_dim
            d2 = 1 # num of channels
            
            x = tf.reshape(img_in, shape=[-1, img_dim, img_dim, 1])
            x = tf.layers.conv2d(x, kernel_size=self.kernel_size, filters=self.filters, 
                strides=self.strides, padding='same', activation=self.activation)
            x = tf.layers.dropout(x, keep_prob)

            x = tf.layers.conv2d(x, kernel_size=self.kernel_size, filters=self.filters, 
                strides=self.strides, padding='same', activation=self.activation)
            x = tf.layers.dropout(x, keep_prob)

            x = tf.layers.conv2d(x, kernel_size=self.kernel_size, filters=self.filters, 
                strides=self.strides, padding='same', activation=self.activation)
            x = tf.layers.dropout(x, keep_prob)

            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=d1 * d1 * d1, activation=self.activation)
            x = tf.layers.dense(x, units=d2, activation=tf.nn.sigmoid)

            return x


if __name__ == '__main__':

    from config import Config
    parser = setup_argparse()
    parser.add_argument('-d', '--dataset_name', default='adult')
    args = get_parser_args(parser)
    
    config = Config(args.dataset_name, 'default')
    disc = Discriminator(config)

    num_cols = config.num_real + config.num_cat
    img_dim = int(math.ceil(math.sqrt(num_cols)))
    zeroes_to_pad = img_dim * img_dim - num_cols

    tf.reset_default_graph()

    # Placeholders - filled when we run tensorflow session
    img_in = tf.placeholder(dtype=tf.float32, shape=[None, img_dim, img_dim], name='X')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    disc_out = disc.discriminate(img_in, img_dim, keep_prob)
    pdb.set_trace()


        