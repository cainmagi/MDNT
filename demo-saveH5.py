#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################################
# Demo for saving H5
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6
#   tensorflow r1.13+
#   numpy, matplotlib
# This is just a simple demo for saving H5 file.
# The script would produce a mnist dataset.
# Version: 1.10 # 2019/3/26
# Comments:
#   Enable the data stored by compressed mode without impact on the
#   efficiency of reading.
# Version: 1.00 # 2019/3/26
# Comments:
#   Create this project.
####################################################################
'''

from tensorflow.keras.datasets import mnist

import mdnt
import os, sys
os.chdir(sys.path[0])

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    S_train = mdnt.data.H5SupSaver('mnist-train')
    S_test  = mdnt.data.H5SupSaver('mnist-test')
    S_train.config(logver=1, compression='gzip', shuffle=True)
    S_test.config(logver=1, compression='gzip', shuffle=True)
    S_train.dump('X', x_train, chunks=(1, 28,28))
    S_train.dump('Y', y_train)
    S_test.dump('X', x_test, chunks=(1, 28,28))
    S_test.dump('Y', y_test)
    S_test.open('mnist-test-single')
    S_test.dump('X', x_test)