#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################################
# Demo 1st for external layer
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6
#   tensorflow r1.13+
#   numpy, matplotlib
# In this demo, we define a transformation from Cartesian coordinate
# system to polar coordinate system in numpy, and let it involved in
# tf-keras framework. Although it is not necessary to do that, we
# just use such a method to check the applicability of External API.
# Version: 1.00 # 2019/5/23
# Comments:
#   Create this project.
####################################################################
'''

import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

import mdnt
import os, sys
os.chdir(sys.path[0])

def cart2polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return rho, theta

def cart2polar_back(x, y, rho, theta, grad_rho, grad_theta):
    xgrad = grad_rho * x / rho + grad_theta * ( ( x*np.cos(theta) )/( rho*rho*np.sin(theta) ) - 1/(rho*np.sin(theta)) )
    ygrad = grad_rho * y / rho + grad_theta * ( 1/(rho*np.cos(theta)) - ( y*np.sin(theta) )/( rho*rho*np.cos(theta) ) ) 
    return xgrad, ygrad

def cart2polar_oshape(ishape):
    return ishape

customObj = {
    'cart2polar': cart2polar,
    'cart2polar_back': cart2polar_back,
    'cart2polar_oshape': cart2polar_oshape
}
    
def build_model():
    # Build the model
    # this is our input placeholder
    input_1 = tf.keras.layers.Input(shape=(1,))
    input_2 = tf.keras.layers.Input(shape=(1,))
    # Create 2-in 2-out external layer
    output_1, output_2 = mdnt.layers.PyExternal(forward=cart2polar, 
                                       backward=cart2polar_back, 
                                       output_shape=cart2polar_oshape, 
                                       Tin=[tf.float32, tf.float32], 
                                       Tout=[tf.float32, tf.float32],
                                       yEnable=True
                                      )([input_1, input_2])
    lfunc = tf.keras.layers.Lambda(lambda x: tf.gradients(x[0], [x[1], x[2]]))
    gradients_rho = lfunc([output_1, input_1, input_2])
    gradients_theta = lfunc([output_2, input_1, input_2])

    # this model maps an Cartesian to polar
    cart2polar_model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=[output_1, output_2, *gradients_rho, *gradients_theta])
    cart2polar_model.summary()
    
    return cart2polar_model

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
            s (show)  : display mode, would create a model and plot the results.
            wt (write): write mode, would write the model into --savedPath.
            rd (read) : read mode, would reload the model and plot the results.
        '''
    )
    
    parser.add_argument(
        '-mn', '--modelName', default='default', metavar='str',
        help='''\
        The name scope for specified model. This is necessary for using multiple
        models when using crossing layer instances.
        '''
    )
    
    parser.add_argument(
        '-s', '--savedPath', default='TestExternal-1', metavar='str',
        help='''\
        The file name (with path) of the saved model. This path is also used for reading model.
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

    def draw_quiver(X, Y, U, V, label=r''):
        U.resize(X.shape)
        V.resize(Y.shape)
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
        Q = ax.quiver(X, Y, U, V, units='x', pivot='tip', width=0.022,
                    scale=1 / 0.2)
        qk = ax.quiverkey(Q, 0.9, 0.9, 1, label, labelpos='E',
                        coordinates='figure')
        ax.scatter(X, Y, color='0.5', s=1)
        plt.show()
    
    if args.mode.casefold() == 's' or args.mode.casefold() == 'show':
        with tf.name_scope(args.modelName):
            cart2polar_model = build_model()
    elif args.mode.casefold() == 'wt' or args.mode.casefold() == 'write':
        with tf.name_scope(args.modelName):
            cart2polar_model = build_model()
        mdnt.save_model(cart2polar_model,
                        filepath=args.savePath+'.h5',
                        headpath=args.savePath+'_model.json',
                        optmpath=args.savePath+'_optm.json')
        cart2polar_model.save(args.savedPath+'.h5')
        print('Model saved to {0}'.format(args.savedPath+'.h5'))
        exit(0)
    elif args.mode.casefold() == 'rd' or args.mode.casefold() == 'read':
        with tf.name_scope(args.modelName):
            cart2polar_model = mdnt.load_model(filepath=args.savedPath+'.h5',
                                               headpath=args.savePath+'_model.json',
                                               optmpath=args.savePath+'_optm.json'
                                               custom_objects=customObj)
        cart2polar_model.summary()
    else:
        print('Need to specify the mode manually. (use -m)')
        parser.print_help()
        exit(0)
    X, Y = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
    _, _, grad_rx, grad_ry, grad_tx, grad_ty = cart2polar_model.predict([np.reshape(X, [X.size, 1]), np.reshape(Y, [Y.size, 1])])
    draw_quiver(X, Y, grad_rx, grad_ry, r'$\dfrac{\partial \rho}{\partial \mathbf{x}}$')
    draw_quiver(X, Y, grad_tx, grad_ty, r'$\dfrac{\partial \theta}{\partial \mathbf{x}}$')