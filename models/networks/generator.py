import warnings
warnings.simplefilter('ignore')

import sys
import math
import pdb
import tensorflow as tf

sys.path.append('/mnt/experiments/privacy-GAN/')
from utils.model import lrelu, relu
from utils.generic import *


class Generator(object):
    """docstring for Generator"""
    def __init__(self, config):
        super(Generator, self).__init__()
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

    def generate(self, inputs, img_dim, keep_prob, is_training):

        with tf.variable_scope("generator", reuse=None):
            
            x = inputs
            d1 = img_dim
            d2 = 1 # num of channels
            
            x = tf.layers.dense(x, units=d1 * d1 * d2, activation=self.activation)
            x = self.dropout(x, keep_prob)
            x = self.batch_norm(x, is_training)
            x = tf.reshape(x, shape=[-1, d1, d1, d2])
            x = tf.layers.conv2d_transpose(x, kernel_size=self.kernel_size, filters=self.filters, 
                                           strides=self.strides, padding='same', activation=self.activation)
            x = self.dropout(x, keep_prob)
            x = self.batch_norm(x, is_training)
            x = tf.layers.conv2d_transpose(x, kernel_size=self.kernel_size, filters=self.filters, 
                                           strides=self.strides, padding='same', activation=self.activation)
            x = self.dropout(x, keep_prob)
            x = self.batch_norm(x, is_training)
            x = tf.layers.conv2d_transpose(x, kernel_size=self.kernel_size, filters=self.filters, 
                                           strides=self.strides, padding='same', activation=self.activation)
            x = self.dropout(x, keep_prob)
            x = self.batch_norm(x, is_training)
            x = tf.layers.conv2d_transpose(x, kernel_size=self.kernel_size, filters=d2, 
                                           strides=self.strides, padding='same', activation=tf.nn.tanh)

            return x

if __name__ == '__main__':

    from config import Config
    parser = setup_argparse()
    parser.add_argument('-d', '--dataset_name', default='adult')
    args = get_parser_args(parser)
    
    config = Config(args.dataset_name, 'default')
    gen = Generator(config)

    num_cols = config.num_real + config.num_cat
    img_dim = int(math.ceil(math.sqrt(num_cols)))
    zeroes_to_pad = img_dim * img_dim - num_cols

    tf.reset_default_graph()

    # Placeholders - filled when we run tensorflow session
    noise = tf.placeholder(dtype=tf.float32, shape=[None, config.noise_dim], name='noise')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    g_noise = gen.generate(noise, img_dim, keep_prob, is_training)
    pdb.set_trace()


        