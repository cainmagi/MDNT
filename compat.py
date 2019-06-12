'''
################################################################
# Compatibility check
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Use this module to check whether we need to open the
# compatible mode.
# Version: 0.10 # 2019/6/12
# Comments:
# 1. Modify the required version for compatible mode.
# 2. Provide a stronger property collecting method for
#    compatibility.
# Version: 0.10 # 2019/3/27
# Comments:
#   Create this compatible module.
################################################################
'''

# Check compatibility
import tensorflow

if [int(i) for i in tensorflow.__version__.split('-')[0].split('.')] < [1, 13, 0]:
    COMPATIBLE_MODE = True
else:
    COMPATIBLE_MODE = False

def collect_properties(layer, sublayer):
    '''
    Collect the following parameters from sublayer to layer:
        _trainable_weights
        _non_trainable_weights
        _updates
        _losses
    '''
    if COMPATIBLE_MODE: # for compatibility
        layer._trainable_weights.extend(sublayer._trainable_weights)
        layer._non_trainable_weights.extend(sublayer._non_trainable_weights)
        layer._updates.extend(sublayer._updates)
        layer._losses.extend(sublayer._losses)
        if hasattr(layer, '_callable_losses') and hasattr(sublayer, '_callable_losses'): # for compatibility on 1.12.0
            layer._callable_losses.extend(sublayer._callable_losses)