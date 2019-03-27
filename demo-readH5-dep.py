#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################################
# Demo for reading H5 (deprecated test)
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

from mdnt.data.deprecated import h5py
import os, sys
os.chdir(sys.path[0])
h5py.REMOVE_DEPRECATION = True

if __name__ == '__main__':
    def preproc(x, y):
        x = x / 255.
        # Add noise
        noise_factor = 0.5
        x_noisy = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
        return x_noisy, x
        
    P = h5py.H5GParser('mnist-test', ['X', 'Y'],  batchSize=32, preprocfunc=lambda x, y: tf.py_function(preproc, [x, y], [tf.float32]*2))
    dp = P.getDataset()
    
    iterator = dp.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(10):
            value = sess.run(next_element)
            print([value[0].shape, value[1].shape])