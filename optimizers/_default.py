'''
################################################################
# Optimizers - Default tools.
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Basic tools for this module.
# The default tools would be imported directly into the current
# sub-module.
# Version: 0.10 # 2019/3/23
# Comments:
#   Create this submodule.
################################################################
'''

from tensorflow.python.keras import optimizers

def optimizer(name='adam', l_rate=0.01, decay=0.0):
    '''
    Define the optimizer by default parameters except learning rate.
    Note that most of optimizers do not suggest users to modify their
    speically designed parameters.
    Options:
        name: the name of optimizer (default='adam') (available: 'adam', 
              'amsgrad', 'adamax', 'nadam', 'adadelta', 'rms', 'adagrad',
              'nmoment', 'sgd')
        l_rate: learning rate (default=0.01)
        decay: decay ratio ('adadeltaDA' do not support this option)
    Return:
        the particular optimizer object.
    '''
    name = name.casefold()
    if name == 'adam':
        return optimizers.Adam(l_rate, decay=decay)
    elif name == 'amsgrad':
        return optimizers.Adam(l_rate, decay=decay, amsgrad=True)
    elif name == 'adamax':
        return optimizers.Adamax(l_rate, decay=decay)
    elif name == 'nadam':
        return optimizers.Nadam(l_rate, schedule_decay=decay)
    elif name == 'adadelta':
        return optimizers.Adadelta(l_rate, decay=decay)
    elif name == 'rms':
        return optimizers.RMSprop(l_rate, decay=decay)
    elif name == 'adagrad':
        return optimizers.Adagrad(l_rate, decay=decay)
    elif name == 'nmoment':
        return optimizers.SGD(lr=l_rate, momentum=0.6, decay=decay, nesterov=True)
    elif name == 'sgd':
        return optimizers.SGD(l_rate, decay=decay)