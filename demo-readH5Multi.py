#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################################
# Demo for reading H5
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6
#   tensorflow r1.13+
#   numpy, matplotlib
# This is just a simple demo for reading multiple unrelated sets.
# Version: 1.00 # 2019/3/31
# Comments:
#   Create this project.
####################################################################
'''
import numpy as np
import tensorflow as tf

import mdnt
import os, sys
os.chdir(sys.path[0])

if __name__ == '__main__':
    def preproc(x, y):
        '''
        Now we have changed the preprocessing function so that its input should be batches.
        '''
        x = x / 255.
        # Add noise
        noise_factor = 0.5
        x_noisy = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
        return (x_noisy,)
        
    def preprocComb(p1, p2):
        '''
        The processing function for the combiner.
        '''
        X = (p1[0], p2[0])
        Y = p2[1]
        return X, Y
        
    P1 = mdnt.data.H5GParser('mnist-test', ['X', 'Y'],  batchSize=32, preprocfunc=preproc)
    P2 = mdnt.data.H5GParser('mnist-train', ['X', 'Y'],  batchSize=32, preprocfunc=None)
    P = mdnt.data.H5GCombiner(P1, P2, preprocfunc=preprocComb)
    dp = iter(P)
    
    for i in range(3000): # 32*3000 = 96000 > 60000 >> 10000
        try:
            value = next(dp)
            print('Iter {0}:'.format(i), [value[0][0].shape, value[0][1].shape, value[1].shape])
        except StopIteration:
            dp = iter(P)