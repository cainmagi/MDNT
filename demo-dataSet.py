#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################################
# Demo for working with MDNT data set.
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6
#   tensorflow r1.13+
#   numpy, matplotlib
# This test should be run after the saveH5 test, which means you
# need to perform:
# ```
# python demo-saveH5.py
# ```
# at first.
# Test the performance of datasets. The data of this test is from 
# a dataset handle.
# Use
# ```
# python demo-dataSet.py -m tr -s dataset
# ```
# to train the network. Then use
# ```
# python demo-dataSet.py -m ts -s dataset -rd model-...
# ```
# to perform the test.
# Version: 1.15 # 2019/3/28
# Comments:
#   1. Add a depth to test the padding policy.
#   2. Introduce the correlation into the metrics.
#   3. Add tensorboard logger.
# Version: 1.10 # 2019/3/27
# Comments:
#   Use revised data parser to get batches.
# Version: 1.10 # 2019/3/26
# Comments:
#   Change the method of pre-processing data.
# Version: 1.00 # 2019/3/26
# Comments:
#   Create this project.
####################################################################
'''

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import random
import matplotlib.pyplot as plt

import mdnt
import os, sys
os.chdir(sys.path[0])
#mdnt.layers.conv.NEW_CONV_TRANSPOSE=False

def plot_sample(x_test, x_input=None, decoded_imgs=None, n=10):
    '''
    Plot the first n digits from the input data set
    '''
    plt.figure(figsize=(20, 4))
    rows = 1
    if decoded_imgs is not None:
        rows += 1
    if x_input is not None:
        rows += 1
        
    def plot_row(x, row, n, i):
        ax = plt.subplot(rows, n, i + 1 + row*n)
        plt.imshow(x[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    for i in range(n):
        # display original
        row = 0
        plot_row(x_test, row, n, i)
        if x_input is not None:
            # display reconstruction
            row += 1
            plot_row(x_input, row, n, i)
        if decoded_imgs is not None:
            # display reconstruction
            row += 1
            plot_row(decoded_imgs, row, n, i)
    plt.show()
    
def mean_loss_func(lossfunc, name=None, *args, **kwargs):
    def wrap_func(*args, **kwargs):
        return tf.keras.backend.mean(lossfunc(*args, **kwargs))
    if name is not None:
        wrap_func.__name__ = name
    return wrap_func
    
def correlation(y_true, y_pred):
    m_y_true = tf.keras.backend.mean(y_true, axis=0)
    m_y_pred = tf.keras.backend.mean(y_pred, axis=0)
    s_y_true = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true), axis=0) - tf.keras.backend.square(m_y_true))
    s_y_pred = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred), axis=0) - tf.keras.backend.square(m_y_pred))
    s_denom = s_y_true * s_y_pred
    s_numer = tf.keras.backend.mean(y_true * y_pred, axis=0) - m_y_true * m_y_pred
    s_index = tf.keras.backend.greater(s_denom, 0)
    return tf.keras.backend.mean(tf.boolean_mask(s_numer,s_index)/tf.boolean_mask(s_denom,s_index))
    
def build_model():
    # Build the model
    channel_1 = 64  # 32 channels
    channel_2 = 128  # 64 channels
    channel_3 = 256  # 128 channels
    # this is our input placeholder
    input_img = tf.keras.layers.Input(shape=(28, 28, 1))
    # Create encode layers
    conv_1 = mdnt.layers.AConv2D(channel_1, (3, 3), strides=(2, 2), normalization='inst', activation='prelu', padding='same')(input_img)
    conv_2 = mdnt.layers.AConv2D(channel_2, (3, 3), strides=(2, 2), normalization='inst', activation='prelu', padding='same')(conv_1)
    conv_3 = mdnt.layers.AConv2D(channel_3, (3, 3), strides=(2, 2), normalization='inst', activation='prelu', padding='same')(conv_2)
    deconv_3 = mdnt.layers.AConv2DTranspose(channel_2, (3, 3), strides=(2, 2), output_padding=((0,1), (0,1)), normalization='inst', activation='prelu', padding='valid')(conv_3)
    deconv_2 = mdnt.layers.AConv2DTranspose(channel_1, (3, 3), strides=(2, 2), normalization='inst', activation='prelu', padding='same')(deconv_3)
    deconv_1 = mdnt.layers.AConv2DTranspose(1, (3, 3), strides=(2, 2), normalization='bias', activation=tf.nn.sigmoid, padding='same')(deconv_2)
        
    # this model maps an input to its reconstruction
    denoiser = tf.keras.models.Model(input_img, deconv_1)
    denoiser.summary(line_length=90, positions=[.55, .85, .95, 1.])
    return denoiser

if __name__ == '__main__':
    import argparse
    
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
    parser = argparse.ArgumentParser(
        description='Perform regression on an analytic non-linear model in frequency domain.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Parse arguments.
    
    parser.add_argument(
        '-m', '--mode', default='', metavar='str',
        help='''\
        The mode of this demo.
            tr (train): training mode, would train and save a network.
            ts (test) : testing mode, would reload a network and give predictions.
        '''
    )
    
    parser.add_argument(
        '-r', '--rootPath', default='checkpoints', metavar='str',
        help='''\
        The root path for saving the network.
        '''
    )
    
    parser.add_argument(
        '-s', '--savedPath', default='model', metavar='str',
        help='''\
        The folder of the submodel in a particular train/test.
        '''
    )
    
    parser.add_argument(
        '-rd', '--readModel', default='model', metavar='str',
        help='''\
        The name of the model. (only for testing)
        '''
    )
    
    parser.add_argument(
        '-lr', '--learningRate', default=0.01, type=float, metavar='float',
        help='''\
        The learning rate for training the model. (only for training)
        '''
    )
    
    parser.add_argument(
        '-e', '--epoch', default=20, type=int, metavar='int',
        help='''\
        The number of epochs for training. (only for training)
        '''
    )
    
    parser.add_argument(
        '-tbn', '--trainBatchNum', default=256, type=int, metavar='int',
        help='''\
        The number of samples per batch for training. (only for training)
        '''
    )
    
    parser.add_argument(
        '-tsn', '--testBatchNum', default=10, type=int, metavar='int',
        help='''\
        The number of samples for testing. (only for testing)
        '''
    )
    
    parser.add_argument(
        '-sd', '--seed', default=None, type=int, metavar='int',
        help='''\
        Seed of the random generaotr. If none, do not set random seed.
        '''
    )

    parser.add_argument(
        '-gn', '--gpuNumber', default=-1, type=int, metavar='int',
        help='''\
        The number of used GPU. If set -1, all GPUs would be visible.
        '''
    )
    
    args = parser.parse_args()
    def setSeed(seed):
        np.random.seed(seed)
        random.seed(seed+12345)
        tf.set_random_seed(seed+1234)
    if args.seed is not None: # Set seed for reproductable results
        setSeed(args.seed)
    if args.gpuNumber != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuNumber)
    
    def preproc(x):
        '''
        Now we have changed the preprocessing function so that its input should be batches.
        '''
        x = x / 255.
        x = x.reshape(len(x), 28, 28, 1)
        # Add noise
        noise_factor = 0.5
        x_noisy = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
        x_noisy = np.clip(x_noisy, 0., 1.)
        return (x_noisy, x)
    
    if args.mode.casefold() == 'tr' or args.mode.casefold() == 'train':
        denoiser = build_model()
        denoiser.compile(optimizer=mdnt.optimizers.optimizer('amsgrad', l_rate=args.learningRate), 
                            loss=mean_loss_func(tf.keras.losses.binary_crossentropy, name='mean_binary_crossentropy'),
                            metrics=[correlation])
        
        folder = os.path.abspath(os.path.join(args.rootPath, args.savedPath))
        if os.path.abspath(folder) == '.' or folder == '':
            args.rootPath = 'checkpoints'
            args.savedPath = 'model'
            folder = os.path.abspath(os.path.join(args.rootPath, args.savedPath))
        if tf.gfile.Exists(folder):
            tf.gfile.DeleteRecursively(folder)
        checkpointer = mdnt.utilities.callbacks.ModelCheckpoint(filepath=os.path.join(folder, 'model'),
                                                                record_format='{epoch:02d}e-val_loss_{val_loss:.2f}',
                                                                keep_max=5, save_best_only=True, verbose=1,  period=5)
        tf.gfile.MakeDirs(folder)
        logger = tf.keras.callbacks.TensorBoard(log_dir=os.path.join('.', 'logs', args.savedPath), 
            histogram_freq=5, write_graph=True, write_grads=False, write_images=False, update_freq=5)
        parser_train = mdnt.data.H5GParser('mnist-train', ['X'],  batchSize=args.trainBatchNum, preprocfunc=preproc)
        parser_test = mdnt.data.H5GParser('mnist-test', ['X'],  batchSize=args.trainBatchNum, preprocfunc=preproc)
        denoiser.fit(parser_train, steps_per_epoch=len(parser_train),
                    epochs=args.epoch,
                    validation_data=parser_test, validation_steps=len(parser_test),
                    callbacks=[checkpointer, logger])
    
    elif args.mode.casefold() == 'ts' or args.mode.casefold() == 'test':
        parser_test = mdnt.data.H5GParser('mnist-test', ['X'],  batchSize=args.testBatchNum, shuffle=False, preprocfunc=preproc)
        data_test = iter(parser_test)
        noisy, clean = next(data_test)
        denoiser = mdnt.load_model(os.path.join(args.rootPath, args.savedPath, args.readModel)+'.h5', 
                                   headpath=os.path.join(args.rootPath, args.savedPath, 'model')+'.json', 
                                   custom_objects={'mean_binary_crossentropy':mean_loss_func(tf.keras.losses.binary_crossentropy)})
        denoiser.summary(line_length=90, positions=[.55, .85, .95, 1.])
        decoded_imgs = denoiser.predict(noisy)
        plot_sample(clean, noisy, decoded_imgs, n=args.testBatchNum)
    else:
        print('Need to specify the mode manually. (use -m)')
        parser.print_help()