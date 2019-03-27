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
# This is just a simple demo for reading H5 file.
# Warning:
#   Although the test is passed in this script, the standard tf
#   dataset is proved to be incompatible with tf-K architecture.
#   We need to wait until tf fix the bug.
# Version: 1.00 # 2019/3/26
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
        
    P = mdnt.data.H5GParser('mnist-test', ['X', 'Y'],  batchSize=32, preprocfunc=preproc)
    dp = P.getDataset()
    
    for i in range(10):
        value = next(dp)
        print([value[0].shape])