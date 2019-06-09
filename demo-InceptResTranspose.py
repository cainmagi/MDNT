#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################################
# Demo for transposed inception residual layers.
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6
#   tensorflow r1.13+
#   numpy, matplotlib
# Test the performance of modern transposed inception residual
# layers.
# Use
# ```
# python demo-InceptResTranspose.py -m tr -s tirinst -mm inst
# ```
# to train the network. Then use
# ```
# python demo-InceptResTranspose.py -m ts -s tirinst -mm inst -rd model-...
# ```
# to perform the test. Here we use `inst` to set instance
# normalization.
# Version: 1.00 # 2019/6/4
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

def build_model(mode='bias'):
    # Make configuration
    mode = mode.casefold()
    if not mode in ['batch', 'inst', 'group']:
        if mode == 'bias':
            mode = True
        else:
            mode = False
    # Build the model
    channel_1 = 32  # 32 channels
    channel_2 = 64  # 64 channels
    channel_3 = 128  # 128 channels
    # this is our input placeholder
    input_img = tf.keras.layers.Input(shape=(28, 28, 1))
    # Create encode layers
    conv_1 = mdnt.layers.AConv2D(channel_1, (3, 3), strides=(2, 2), normalization=mode, activation='prelu', padding='same')(input_img)
    for i in range(3):
        conv_1 = mdnt.layers.Inceptres2D(channel_1, (3, 3), normalization=mode, activation='prelu')(conv_1)
    conv_2 = mdnt.layers.Inceptres2D(channel_2, (3, 3), strides=(2, 2), normalization=mode, activation='prelu')(conv_1)
    for i in range(3):
        conv_2 = mdnt.layers.Inceptres2D(channel_2, (3, 3), normalization=mode, activation='prelu')(conv_2)
    conv_3 = mdnt.layers.Inceptres2D(channel_3, (3, 3), strides=(2, 2), normalization=mode, activation='prelu')(conv_2)
    for i in range(3):
        conv_3 = mdnt.layers.Inceptres2D(channel_3, (3, 3), normalization=mode, activation='prelu')(conv_3)
    # Create decode layers
    deconv_1 = mdnt.layers.Inceptres2DTranspose(channel_2, (3, 3), strides=(2, 2), output_mshape=conv_2.get_shape(), normalization=mode, activation='prelu')(conv_3)
    for i in range(3):
        deconv_1 = mdnt.layers.Inceptres2D(channel_2, (3, 3), normalization=mode, activation='prelu')(deconv_1)
    deconv_2 = mdnt.layers.Inceptres2DTranspose(channel_1, (3, 3), strides=(2, 2), output_mshape=conv_1.get_shape(), normalization=mode, activation='prelu')(deconv_1)
    for i in range(3):
        deconv_2 = mdnt.layers.Inceptres2D(channel_1, (3, 3), normalization=mode, activation='prelu')(deconv_2)
    deconv_3 = mdnt.layers.AConv2DTranspose(1, (3, 3), strides=(2, 2), output_mshape=input_img.get_shape(), normalization='bias', padding='same', activation=tf.nn.sigmoid)(deconv_2)
    # this model maps an input to its reconstruction
    denoiser = tf.keras.models.Model(input_img, deconv_3)
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
        '-mm', '--modelMode', default='bias', metavar='str',
        help='''\
        The mode of this demo.
            bias:  use biases instead of using normalization.
            batch: use batch normalization.
            inst : use instance normalization.
            group: use group normalization.
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
        '-rlr', '--reduceLR', type=str2bool, nargs='?', const=True, default=False, metavar='bool',
        help='''\
        Use the automatic learning rate reducing scheduler (would reduce LR to 0.1 of the initial configuration if need). (only for train)
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
        (x_train, _), (x_test, _) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape(len(x_train), 28, 28, 1)
        x_test = x_test.reshape(len(x_test), 28, 28, 1)
        #plot_sample(x_train, n=10)
        
        # Add noise
        noise_factor = 0.5
        x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
        x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        x_test_noisy = np.clip(x_test_noisy, 0., 1.)
        return x_train, x_test, x_train_noisy, x_test_noisy
    
    if args.mode.casefold() == 'tr' or args.mode.casefold() == 'train':
        x_train, x_test, x_train_noisy, x_test_noisy = load_data()
        denoiser = build_model(args.modelMode)
        if args.modelMode.casefold() == 'bias':
            denoiser.compile(optimizer=mdnt.optimizers.optimizer('nmoment', l_rate=args.learningRate), 
                                loss=mean_loss_func(tf.keras.losses.binary_crossentropy, name='mean_binary_crossentropy'))
        else:
            denoiser.compile(optimizer=mdnt.optimizers.optimizer('amsgrad', l_rate=args.learningRate), 
                                loss=mean_loss_func(tf.keras.losses.binary_crossentropy, name='mean_binary_crossentropy'))
        
        folder = os.path.abspath(os.path.join(args.rootPath, args.savedPath))
        if os.path.abspath(folder) == '.' or folder == '':
            args.rootPath = 'checkpoints'
            args.savedPath = 'model'
            folder = os.path.abspath(os.path.join(args.rootPath, args.savedPath))
        if tf.gfile.Exists(folder):
            tf.gfile.DeleteRecursively(folder)
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='-'.join((os.path.join(folder, 'model'), '{epoch:02d}e-val_acc_{val_loss:.2f}.h5')), save_best_only=True, verbose=1,  period=5)
        tf.gfile.MakeDirs(folder)
        logger = tf.keras.callbacks.TensorBoard(log_dir=os.path.join('./logs/', args.savedPath), 
            histogram_freq=5, write_graph=True, write_grads=False, write_images=False, update_freq=5)
        if args.reduceLR:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=args.learningRate*0.1, verbose=1)
            get_callbacks = [checkpointer, logger, reduce_lr]
        else:
            get_callbacks = [checkpointer, logger]
        denoiser.fit(x_train_noisy, x_train,
                    epochs=args.epoch,
                    batch_size=args.trainBatchNum,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=get_callbacks)
    
    elif args.mode.casefold() == 'ts' or args.mode.casefold() == 'test':
        denoiser = mdnt.load_model(os.path.join(args.rootPath, args.savedPath, args.readModel)+'.h5', custom_objects={'mean_binary_crossentropy':mean_loss_func(tf.keras.losses.binary_crossentropy)})
        denoiser.summary(line_length=90, positions=[.55, .85, .95, 1.])
        #for l in denoiser.layers:
        #    print(l.name, l.trainable_weights)
        _, x_test, _, x_test_noisy = load_data()
        decoded_imgs = denoiser.predict(x_test_noisy[:args.testBatchNum, :])
        plot_sample(x_test[:args.testBatchNum, :], x_test_noisy[:args.testBatchNum, :], decoded_imgs, n=args.testBatchNum)
    else:
        print('Need to specify the mode manually. (use -m)')
        parser.print_help()