#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################################
# Demo for weight decay callback
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6
#   tensorflow r1.13+
#   numpy, matplotlib
# Test the performance of weight decay callback on momentum SGD.
# The L1 and L2 penalty could not work well on adaptive learning
# rate algorithms. For example, L1 penalty could not maintain the
# sparsity when using Adam.
# However, the implementation ModelWeightsReducer solves this prob-
# lem properly. For example, considering the L1 regularization, 
# instead of applying the gradient from L1 penalty, our method
# applies soft thresholding to the weights directly, so the spar-
# sity will be maintained directly and do not get influenced by
# a specifc optimizing method.
# This test also check the Ghost layer. To learn more about how to
# use Ghost layer, check mdnt.layers.Ghost.
# Check the performances by:
# (1) Train and save the training parameters:
# ```
# python demo-weightDecay.py -m tr
# ```
# (2) Read a saved model:
# ```
# python demo-weightDecay.py -m ts -rd model-...
# ```
# Version: 1.00 # 2019/6/24
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

VEC_LEN = 30
    
def build_model():
    # Build the model. It only has one layer but all elements of the output is trainable.

    input_vec = tf.keras.layers.Input(shape=(VEC_LEN,))
    var_input = mdnt.layers.Ghost(use_kernel=True)(input_vec)
    
    # this model maps an input to its reconstruction
    determine = tf.keras.models.Model(input_vec, var_input)
    determine.summary()
    
    return determine

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
        '-s', '--savedPath', default='decay', metavar='str',
        help='''\
        The folder of the submodel in a particular train/test.
        '''
    )
    
    parser.add_argument(
        '-rd', '--readModel', default='model-03e-val_loss_0.00', metavar='str',
        help='''\
        The name of the model. (only for testing)
        '''
    )
    
    parser.add_argument(
        '-lr', '--learningRate', default=0.001, type=float, metavar='float',
        help='''\
        The learning rate for training the model. (only for training)
        '''
    )
    
    parser.add_argument(
        '-e', '--epoch', default=3, type=int, metavar='int',
        help='''\
        The number of epochs for training. (only for training)
        '''
    )
    
    parser.add_argument(
        '-sd', '--seed', default=1, type=int, metavar='int',
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
    
    def load_data():
        y = 2 * np.random.rand(1, VEC_LEN) - 1
        y[np.abs(y)<0.7] = 0.0
        y[y>0.6] = 1.0
        y[y<-0.6] = -1.0
        y = y + 0.6 * np.random.rand(1, VEC_LEN) - 0.3
        y[y>0.6] = 1.0
        y[y<-0.6] = -1.0
        x = np.ones([1, VEC_LEN])
        y = np.repeat(y, 5000, axis=0)
        x = np.repeat(x, 5000, axis=0)
        return x, y
    
    if args.mode.casefold() == 'tr' or args.mode.casefold() == 'train':
        x, y = load_data()
        determine = build_model()
        determine.compile(optimizer=mdnt.optimizers.optimizer('adam', l_rate=args.learningRate), 
                            loss=tf.losses.mean_squared_error)
        
        folder = os.path.abspath(os.path.join(args.rootPath, args.savedPath))
        if os.path.abspath(folder) == '.' or folder == '':
            args.rootPath = 'checkpoints'
            args.savedPath = 'model'
            folder = os.path.abspath(os.path.join(args.rootPath, args.savedPath))
        if tf.gfile.Exists(folder):
            tf.gfile.DeleteRecursively(folder)
        checkpointer = mdnt.utilities.callbacks.ModelCheckpoint(filepath=os.path.join(folder, 'model'),
                                                                record_format='{epoch:02d}e-val_loss_{val_loss:.2f}',
                                                                keep_max=1, save_best_only=True, verbose=1,  period=3)
        regularizer = mdnt.utilities.callbacks.ModelWeightsReducer(lam=0.2, mu=1e-3)
        tf.gfile.MakeDirs(folder)
        determine.fit(x, y,
                    epochs=args.epoch,
                    batch_size=1,
                    shuffle=True,
                    validation_data=(x[:1,...], y[:1,...]),
                    callbacks=[checkpointer, regularizer])
        y_pred = determine.predict(x[:1,...])
        plt.plot(y[0, ...], label='Truth'), plt.plot(y_pred[0, ...], label='Prediction'), plt.plot(np.zeros_like(y_pred[0, ...]), 'r--')
        plt.legend(), plt.gcf().set_size_inches(14, 3), plt.tight_layout()
        plt.show()
    
    elif args.mode.casefold() == 'ts' or args.mode.casefold() == 'test':
        x, y = load_data()
        determine = mdnt.load_model(os.path.join(args.rootPath, args.savedPath, args.readModel)+'.h5',
                                    headpath=os.path.join(args.rootPath, args.savedPath, 'model')+'.json')
        determine.summary()
        y_pred = determine.predict(x[:1,...])
        plt.plot(y[0, ...], label='Truth'), plt.plot(y_pred[0, ...], label='Prediction'), plt.plot(np.zeros_like(y_pred[0, ...]), 'r--')
        plt.legend(), plt.gcf().set_size_inches(14, 3), plt.tight_layout()
        plt.show()
    else:
        print('Need to specify the mode manually. (use -m)')
        parser.print_help()