import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import os
import pdb
import sys
import math
import numpy as np
from os.path import *
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import tensorflow as tf

sys.path.append('/mnt/experiments/privacy-GAN/')
from config import Config
from utils.generic import *
from utils.model import lrelu, relu, label, remove_label
from utils.data import load_data_config, load_dataset

from models.losses import *
from models.encoders_decoders import LabelEncoderDecoder, OneHotEncoderDecoder
from models.preprocessor import Preprocessor
from models.networks.generator import Generator
from models.networks.discriminator import Discriminator
from models.networks.classifier import Classifier

class PrivacyGANTrainer(object):
    """docstring for PrivacyGANTrainer"""
    def __init__(self, config):
        super(PrivacyGANTrainer, self).__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.delta_mean = config.tableGAN_params["delta_mean"]
        self.delta_std = config.tableGAN_params["delta_std"]
        self.noise_dim = config.noise_dim
        self.model_type = config.model_type

        self.G = Generator(self.config)
        self.D = Discriminator(self.config)
        self.C = Classifier(self.config)



    def prepare_data(self, data, zeroes_to_pad):

        data = tf.pad(data, [[0,0],[0,zeroes_to_pad]], mode='CONSTANT', constant_values=0)
        dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size=100).batch(self.batch_size)
        itera = dataset.make_initializable_iterator()
        next_ele = itera.get_next()

        return itera, next_ele


    def define_placeholders(self, img_dim):

        self.X_in = tf.placeholder(dtype=tf.float32, shape=[None, img_dim, img_dim], name='X')
        self.noise = tf.placeholder(dtype=tf.float32, shape=[None, self.noise_dim], name='noise')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        return


    def define_losses(self, img_dim):

        g = self.G.generate(self.noise, img_dim, self.keep_prob, self.is_training)
        self.g = g
        d_real = self.D.discriminate(self.X_in, img_dim, self.keep_prob)
        d_fake = self.D.discriminate(g, img_dim, self.keep_prob, reuse=True)

        orig_lables = label(self.X_in, img_dim)
        orig_predictions = self.C.classify(remove_label(self.X_in, img_dim), img_dim, self.keep_prob)

        gene_lables = label(g, img_dim)
        gene_predictions = self.C.classify(remove_label(g, img_dim), img_dim, self.keep_prob, reuse=True)

        loss_c = classification_loss(orig_lables, orig_predictions, self.batch_size)

        loss_d_real = binary_cross_entropy(tf.ones_like(d_real), d_real)
        loss_d_fake = binary_cross_entropy(tf.zeros_like(d_fake), d_fake)
        loss_d = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake))

        loss_g_disc = tf.reduce_mean(binary_cross_entropy(tf.ones_like(d_fake), d_fake))
        loss_g_info = information_loss(self.X_in, g, self.delta_mean, self.delta_std)
        loss_g_class = classification_loss(gene_lables, gene_predictions, self.batch_size)
        loss_g = loss_g_disc + (self.model_type == 'tableGAN-without-clf-loss') * loss_g_info + (self.model_type == 'tableGAN') * loss_g_class

        return loss_c, loss_d, loss_g


    def train(self, itera, next_ele, img_dim, loss_definitions, optimizers):
        loss_c = loss_definitions["classifier"]
        loss_d, loss_g = loss_definitions["discriminator"], loss_definitions["generator"]

        zeroes = np.zeros((self.batch_size, img_dim, img_dim))
        noise_init = np.zeros((self.batch_size, self.noise_dim))

        sess = tf.Session() # may be due to interactive session
        self.sess = sess
        sess.run(tf.global_variables_initializer(), feed_dict={self.X_in: zeroes, self.noise:noise_init, self.keep_prob: 0.6, self.is_training:True})

        d_loss_arr = []
        g_loss_arr = []
        
        iterator = tqdm(range(self.config.num_epochs), total=self.config.num_epochs)

        for i in iterator:
            sess.run(itera.initializer)
            itr_take = 0

            while True:
                
                train_d, train_g = True, True
                train_c = self.config.model_type == "tableGAN"
                keep_prob_train = self.config.keep_prob_train
                n = np.random.uniform(0.0, 1.0, [self.batch_size, self.noise_dim]).astype(np.float32)
                try:
                    batch = sess.run(next_ele)
                except:
                    break # epoch is done, need to reset the iterator
                batch = [np.reshape(b, [img_dim, img_dim]) for b in batch]
                if len(batch) != self.batch_size:
                    continue

                if train_d:
                    sess.run(optimizers['discriminator'], feed_dict={self.noise: n, self.X_in: batch, self.keep_prob: keep_prob_train, self.is_training:True})

                if train_g:
                    sess.run(optimizers['generator'], feed_dict={self.noise: n, self.X_in: batch, self.keep_prob: keep_prob_train, self.is_training:True})

                if train_c:
                    sess.run(optimizers['classifier'], feed_dict={self.noise: n, self.X_in: batch, self.keep_prob: keep_prob_train, self.is_training:True})

                if itr_take % 100 == 0:
                    d_ls, g_ls = sess.run([loss_d, loss_g], feed_dict={self.X_in: batch, self.noise: n, self.keep_prob: keep_prob_train, self.is_training:True})
                    d_loss_arr.append(d_ls)
                    g_loss_arr.append(g_ls)

                    iterator.set_description(" V: {} | G: {:.2f} | D: {:.2f}".format(self.config.model_version, g_ls, d_ls))
                    iterator.refresh()

                itr_take += 1

        return d_loss_arr, g_loss_arr


    def train_wrapper(self, data):

        data = np.array(data)
        self.num_input_samples, self.num_columns = data.shape

        print("==> Input data shape: {}".format(data.shape))
        num_cols = data.shape[1]
        img_dim = int(math.ceil(math.sqrt(num_cols)))
        self.img_dim = img_dim
        zeroes_to_pad = img_dim * img_dim - num_cols
        self.zeroes_to_pad = zeroes_to_pad

        print("==> Resetting TF graph ...")
        tf.reset_default_graph()

        print("==> Preparing the dataset object ...")
        itera, next_ele = self.prepare_data(data, zeroes_to_pad)

        print("==> Defining placeholders ...")
        self.define_placeholders(img_dim)

        networks = ["classifier", "discriminator", "generator"]
        trainable_variables = defaultdict(list)
        regularizers = defaultdict(list)
        loss_definitions = defaultdict(list)
        optimizers = defaultdict(list)
        learning_rates = {"classifier": self.config.c_lr, "discriminator": self.config.d_lr, "generator": self.config.g_lr}

        print("==> Defining losses ...")
        loss_c, loss_d, loss_g = self.define_losses(img_dim)

        print("==> Defining optimizers ...")
        for net in networks:
            loss_definitions[net] = eval('loss_{}'.format(net[0]))
            variables = [var for var in tf.trainable_variables() if var.name.startswith(net)]
            trainable_variables[net] = variables
            regularizers[net] = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), variables)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            for net in networks:
                variables, reg, loss, lr = trainable_variables[net], regularizers[net], loss_definitions[net], learning_rates[net]
                optimizers[net] = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss + reg, var_list=variables)

        print("==> Ready to train ...")
        d_loss_arr, g_loss_arr = self.train(itera, next_ele, img_dim, loss_definitions, optimizers)


    def generate_samples_in_one_go(self, N):

        n = np.random.uniform(0.0, 1.0, [N, self.noise_dim]).astype(np.float32)
        gen_points = self.sess.run(self.g, feed_dict = {self.noise: n, self.keep_prob: 1.0, self.is_training:False})
        gen_points = gen_points.reshape(N, self.img_dim * self.img_dim)
        gen_points = np.delete(gen_points, np.s_[-self.zeroes_to_pad:], 1)

        return gen_points


    def generate_samples(self, num_reqd=0, num_samples_in_one_go=3000):

        if num_reqd <= num_samples_in_one_go:
            return generate_samples_in_one_go(num_reqd)

        if num_reqd == 0:
            num_reqd = self.num_input_samples

        gen_samples = pd.DataFrame(0, index=list(range(num_reqd)), columns=list(range(self.num_columns)))

        total_points_generated = 0
        
        pbar = tqdm(total = num_reqd)
        pbar.set_description('Generating samples')

        while total_points_generated <= num_reqd:
            if total_points_generated == num_reqd - num_reqd % num_samples_in_one_go:
                break

            gen_points = self.generate_samples_in_one_go(num_samples_in_one_go)
            gen_samples.iloc[total_points_generated: total_points_generated + num_samples_in_one_go, :] = gen_points
            total_points_generated += num_samples_in_one_go
            pbar.update(num_samples_in_one_go)
        
        remaining_points = num_reqd - total_points_generated
        gen_points = self.generate_samples_in_one_go(remaining_points)
        gen_samples.iloc[total_points_generated: , :] = gen_points
        pbar.update(remaining_points)
        
        print("==> Generated data shape: ", gen_samples.shape)

        return gen_samples

        

    # Util 1: Loss function plot



if __name__ == '__main__':
    parser = setup_argparse()
    parser.add_argument('-d', '--dataset_name', default='adult')
    parser.add_argument('-s', '--subset', type=bool, default=False)
    args = get_parser_args(parser)
    
    print("=> Loading config ...")
    config = Config(args.dataset_name, 'default')

    print("=> Loading the clean dataset ...")
    data_version = 'clean'
    if args.subset:
        assert args.dataset_name != 'adult'
        data_version = 'clean_subset'
    clean_data = load_dataset(args.dataset_name, data_version)

    print("=> Encoding categorical part of the data ...")
    EncoderDecoder = LabelEncoderDecoder
    if config.encode_method == 'ohe':
        EncoderDecoder = OneHotEncoderDecoder

    enc_dec_object = EncoderDecoder(config)
    encoded_data = enc_dec_object.encode(clean_data)

    print("=> Preprocessing the data ...")
    preprocess_object = Preprocessor(config)
    scaled_data = preprocess_object.scale(encoded_data)

    trainer_object = PrivacyGANTrainer(config)
    trainer_object.train_wrapper(scaled_data)

    generated_data = trainer_object.generate_samples(num_reqd=5000)

    print("=> Inverse preprocessing generated data ...")
    generated_data = preprocess_object.inverse_scale(generated_data)
    generated_data = pd.DataFrame(data=generated_data, index=None, columns=config.cleaned_columns)
    pdb.set_trace()
    

    print("=> Decoding generated data ...")
    if config.encode_method == 'ohe':
        generated_data = enc_dec_object.decode(clean_data, generated_data)
    elif config.encode_method == 'le':
        generated_data = enc_dec_object.decode(generated_data)

    pdb.set_trace()
    
