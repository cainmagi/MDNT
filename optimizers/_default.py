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
# Version: 0.16 # 2019/6/24
# Comments:
#   Change the quick interface to support more MDNT optimizers.
# Version: 0.16 # 2019/6/24
# Comments:
#   Change the warning interface to tensorflow version.
# Version: 0.15 # 2019/6/23
# Comments:
#   Add support for plain momentum SGD.
# Version: 0.12 # 2019/6/21
# Comments:
#   1. Support two more tensorflow based optimizers in fast
#      interface.
#   2. Adjust the default momentum rate of Nesterov SGD to 0.9.
# Version: 0.10 # 2019/3/23
# Comments:
#   Create this submodule.
################################################################
'''

from tensorflow.python.keras import optimizers
from tensorflow.python.training import adagrad_da, proximal_gradient_descent
from tensorflow.contrib.opt.python.training import weight_decay_optimizers
from tensorflow.python.platform import tf_logging as logging
from .adaptive import MNadam, Adabound, Nadabound

def _raise_TF_warn():
    logging.warning('You are using TFOptimizer in this case. '
                  'It does not support saveing/loading optimizer'
                  ' via save_model() and load_model(). In some '
                  'cases, the option decay may not apply to this'
                  ' interface.')

def optimizer(name='adam', l_rate=0.01, decay=0.0, **kwargs):
    '''
    Define the optimizer by default parameters except learning rate.
    Note that most of optimizers do not suggest users to modify their
    speically designed parameters.
    We suggest users to specify gamma according to the practice when
    using Adabound optimizers.
    Options:
        name: the name of optimizer (default='adam') (available: 'adam', 
              'amsgrad', 'adamax', 'adabound', 'amsbound', 'nadam', 
              'namsgrad', 'nadabound', 'namsbound', 'adadelta', 'rms', 
              'adagrad', 'adamw', 'nmoment', 'sgd', 'proximal')
        l_rate: learning rate (default=0.01)
        decay: decay ratio ('adadeltaDA' do not support this option)
        other parameters: see the usage of the specific optimizer.
    Return:
        the particular optimizer object.
    '''
    name = name.casefold()
    if name == 'adam':
        return optimizers.Adam(l_rate, decay=decay, **kwargs)
    elif name == 'amsgrad':
        return optimizers.Adam(l_rate, decay=decay, amsgrad=True, **kwargs)
    elif name == 'adamax':
        return optimizers.Adamax(l_rate, decay=decay, **kwargs)
    elif name == 'adabound':
        return Adabound(l_rate, decay=decay, **kwargs)
    elif name == 'amsbound':
        return Adabound(l_rate, decay=decay, amsgrad=True, **kwargs) 
    elif name == 'nadam':
        return MNadam(l_rate, decay=decay, **kwargs)
    elif name == 'namsgrad':
        return MNadam(l_rate, decay=decay, amsgrad=True, **kwargs)
    elif name == 'nadabound':
        return Nadabound(l_rate, decay=decay, **kwargs)
    elif name == 'namsbound':
        return Nadabound(l_rate, decay=decay, amsgrad=True, **kwargs)
    elif name == 'adadelta':
        return optimizers.Adadelta(l_rate, decay=decay, **kwargs)
    elif name == 'rms':
        return optimizers.RMSprop(l_rate, decay=decay, **kwargs)
    elif name == 'adagrad':
        return optimizers.Adagrad(l_rate, decay=decay, **kwargs)
    elif name == 'adamw':
        _raise_TF_warn()
        if decay != 0.0:
            logging.warning('This optimizer uses \'decay\' as \'weight_decay\'.')
        else:
            raise ValueError('Should use \'decay\' > 0 for AdamW.')
        return weight_decay_optimizers.AdamWOptimizer(weight_decay=decay, learning_rate=l_rate, **kwargs)
    elif name == 'nmoment':
        return optimizers.SGD(lr=l_rate, momentum=0.9, decay=decay, nesterov=True, **kwargs)
    elif name == 'moment':
        return optimizers.SGD(lr=l_rate, momentum=0.9, decay=decay, nesterov=False, **kwargs)
    elif name == 'sgd':
        return optimizers.SGD(lr=l_rate, decay=decay, **kwargs)
    elif name == 'proximal':
        _raise_TF_warn()
        if decay != 0.0:
            logging.warning('This optimizer does not support \'decay\'.')
        return proximal_gradient_descent.ProximalGradientDescentOptimizer(l_rate, **kwargs)