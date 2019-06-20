#!python
# -*- coding: UTF8-*- #
'''
####################################################################
# Demo for denoising.
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6
#   tensorflow r1.13+
#   numpy, matplotlib
# Test the performance of modern layers (advanced blocks) on
# denoising task. By using different options, users could switch to
# different network structures.
# (1) Assign the network
#     using `-nw` to assign the network type, for example, training
#     a inception network requires:
#     ```
#     python demo-denoising.py -m tr -nw ince -s tiinst -mm inst
#     ```
# (2) Use different normalization.
#     `-mm` is used to assign the normalization type, for example,
#     training a network with batch normalization:
#     ```
#     python demo-denoising.py -m tr -s trbatch -mm batch
#     ```
# (3) Use different dropout.
#     `-md` is used to set dropout type. If setting a dropout, the
#     dropout rate would be 0.3. For example, use additive noise:
#     ```
#     python demo-denoising.py -m tr -s trinst_d_add --mm inst --md add
#     ```
# (4) Reduce learning rate during training.
#     Tests prove that reducing learning rate would help training
#     loss converge better. However, the validation loss would be
#     worse in this case. Use `-rlr` to enable automatic learning
#     rate scheduler:
#     ```
#     python demo-denoising.py -m tr -s trinst_lr -rlr --mm inst
#     ```
# (5) Use a different network block depth.
#     Use `-dp` to set depth. For different kinds of blocks, the
#     parameter `depth` has different meanings. For example, if
#     want to use a more narrow incept-plus, use:
#     ```
#     python demo-denoising.py -m tr -nw incp -dp 1 -s tipinstd1 -mm inst
#     ```
# (6) Load a network and perform test.
#     Because the network configuration has also been saved, users
#     do not need to set options when loading a model, for example:
#     ```
#     python demo-denoising.py -m ts -s trinst -rd model-...
#     ```
# Version: 1.26 # 2019/6/20
# Comments:
#   Enable the option for using a different optimizer.
# Version: 1.25 # 2019/6/14
# Comments:
#   Fix a fatal bug for validation.
# Version: 1.20 # 2019/6/13
# Comments:
#   Merge different network tests on denoising together.
# Version: 1.10 # 2019/6/12
# Comments:
#   Enable the tests for dropout methods.
# Version: 1.00 # 2019/6/9
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

def get_network_handle(nwName):
    if nwName == 'res':
        return mdnt.layers.Residual2D, mdnt.layers.Residual2DTranspose
    elif nwName == 'resn':
        return mdnt.layers.Resnext2D, mdnt.layers.Resnext2DTranspose
    elif nwName == 'ince':
        return mdnt.layers.Inception2D, mdnt.layers.Inception2DTranspose
    elif nwName == 'incr':
        return mdnt.layers.Inceptres2D, mdnt.layers.Inceptres2DTranspose
    elif nwName == 'incp':
        return mdnt.layers.Inceptplus2D, mdnt.layers.Inceptplus2DTranspose

def build_model(mode='bias', dropout=None, nwName='res', depth=None):
    # Make configuration
    mode = mode.casefold()
    if not mode in ['batch', 'inst', 'group']:
        if mode == 'bias':
            mode = True
        else:
            mode = False
    # Get network handles:
    Block, BlockTranspose = get_network_handle(nwName)
    kwargs = dict()
    if depth is not None:
        kwargs['depth'] = depth
    # Build the model
    channel_1 = 32  # 32 channels
    channel_2 = 64  # 64 channels
    channel_3 = 128  # 128 channels
    # this is our input placeholder
    input_img = tf.keras.layers.Input(shape=(28, 28, 1))
    # Create encode layers
    conv_1 = mdnt.layers.AConv2D(channel_1, (3, 3), strides=(2, 2), normalization=mode, activation='prelu', padding='same')(input_img)
    for i in range(3):
        conv_1 = Block(channel_1, (3, 3), normalization=mode, activation='prelu', **kwargs)(conv_1)
    conv_2 = Block(channel_2, (3, 3), strides=(2, 2), normalization=mode, activation='prelu', dropout=dropout, **kwargs)(conv_1)
    for i in range(3):
        conv_2 = Block(channel_2, (3, 3), normalization=mode, activation='prelu', **kwargs)(conv_2)
    conv_3 = Block(channel_3, (3, 3), strides=(2, 2), normalization=mode, activation='prelu', dropout=dropout, **kwargs)(conv_2)
    for i in range(3):
        conv_3 = Block(channel_3, (3, 3), normalization=mode, activation='prelu', **kwargs)(conv_3)
    # Create decode layers
    deconv_1 = BlockTranspose(channel_2, (3, 3), strides=(2, 2), output_mshape=conv_2.get_shape(), normalization=mode, activation='prelu', dropout=dropout, **kwargs)(conv_3)
    for i in range(3):
        deconv_1 = Block(channel_2, (3, 3), normalization=mode, activation='prelu', **kwargs)(deconv_1)
    deconv_2 = BlockTranspose(channel_1, (3, 3), strides=(2, 2), output_mshape=conv_1.get_shape(), normalization=mode, activation='prelu', dropout=dropout, **kwargs)(deconv_1)
    for i in range(3):
        deconv_2 = Block(channel_1, (3, 3), normalization=mode, activation='prelu', **kwargs)(deconv_2)
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
        The mode of this demo. (only for training)
            bias:  use biases instead of using normalization.
            batch: use batch normalization.
            inst : use instance normalization.
            group: use group normalization.
        '''
    )

    parser.add_argument(
        '-md', '--dropoutMode', default=None, metavar='str',
        help='''\
        The mode of dropout type in this demo. (only for training)
            None:    do not use dropout.
            plain:   use tf.keras.layers.Dropout.
            add:     use scale-invariant addictive noise.
            mul:     use multiplicative noise.
            alpha:   use alpha dropout.
            spatial: use spatial dropout.
        '''
    )

    parser.add_argument(
        '-nw', '--network', default='res', metavar='str',
        help='''\
        The basic network block. (only for training)
             res: residual block.
            resn: ResNeXt block.
            ince: inception block.
            incr: inception-residual block.
            incp: inception-plus block.
        '''
    )

    parser.add_argument(
        '-o', '--optimizer', default='amsgrad', metavar='str',
        help='''\
        The optimizer for training the network. When setting normalization as 'bias', the optimizer would be overrided by SGD+Momentum.
        '''
    )

    parser.add_argument(
        '-dp', '--blockDepth', default=None, type=int, metavar='int',
        help='''\
        The depth of each network block. (only for training)
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
        Use the automatic learning rate reducing scheduler (would reduce LR to 0.1 of the initial configuration if need). (only for training)
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
        denoiser = build_model(mode=args.modelMode, dropout=args.dropoutMode, nwName=args.network, depth=args.blockDepth)
        if args.modelMode.casefold() == 'bias':
            denoiser.compile(optimizer=mdnt.optimizers.optimizer('nmoment', l_rate=args.learningRate), 
                                loss=mean_loss_func(tf.keras.losses.binary_crossentropy, name='mean_binary_crossentropy'), metrics=[mdnt.functions.metrics.correlation])
        else:
            denoiser.compile(optimizer=mdnt.optimizers.optimizer(args.optimizer, l_rate=args.learningRate), 
                                loss=mean_loss_func(tf.keras.losses.binary_crossentropy, name='mean_binary_crossentropy'), metrics=[mdnt.functions.metrics.correlation])
        
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
        logger = tf.keras.callbacks.TensorBoard(log_dir=os.path.join('./logs/', args.savedPath), 
            histogram_freq=(args.epoch//5), write_graph=True, write_grads=False, write_images=False, update_freq=10)
        if args.reduceLR:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=args.learningRate*0.1, verbose=1)
            get_callbacks = [checkpointer, logger, reduce_lr]
        else:
            get_callbacks = [checkpointer, logger]
        denoiser.fit(x_train_noisy, x_train,
                    epochs=args.epoch,
                    batch_size=args.trainBatchNum,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test),
                    callbacks=get_callbacks)
    
    elif args.mode.casefold() == 'ts' or args.mode.casefold() == 'test':
        denoiser = mdnt.load_model(os.path.join(args.rootPath, args.savedPath, args.readModel)+'.h5', 
                                   headpath=os.path.join(args.rootPath, args.savedPath, 'model')+'.json', 
                                   custom_objects={'mean_binary_crossentropy':mean_loss_func(tf.keras.losses.binary_crossentropy)})
        denoiser.summary(line_length=90, positions=[.55, .85, .95, 1.])
        #for l in denoiser.layers:
        #    print(l.name, l.trainable_weights)
        _, x_test, _, x_test_noisy = load_data()
        decoded_imgs = denoiser.predict(x_test_noisy[:args.testBatchNum, :])
        plot_sample(x_test[:args.testBatchNum, :], x_test_noisy[:args.testBatchNum, :], decoded_imgs, n=args.testBatchNum)
    else:
        print('Need to specify the mode manually. (use -m)')
        parser.print_help()