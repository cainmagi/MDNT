'''
################################################################
# Functions - Others
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# =============================================================
# Warning:
# THIS MODULE IS A PRIVATE ONE, USERS SHOULD NOT GET ACCESS TO
# THIS PART.
# =============================================================
# Some basic functions.
# Version: 0.10 # 2019/6/13
# Comments:
#   Create this submodule.
################################################################
'''

from tensorflow.python.keras import backend as K

def get_channels(y, data_format=None):
    '''get channels
    Get all dimensions other than the channel dimension and the batch dimension.
    Arguments:
        data_format: 'channels_first' or 'channels_last', 
    Input:
        y: tensor, where we need to find the dimension list.
    Output:
        tuple, the channel (dimension) list.
    '''
    get_dims = len(y.get_shape())
    if get_dims < 3:
        raise ValueError('The input tensor should has channel dimension, i.e. it should have at least 3 axes.')
    if data_format is None:
        data_format = K.image_data_format()
    if data_format == 'channels_last':
        get_reduced_axes = tuple(range(1, get_dims-1))
    else:
        get_reduced_axes = tuple(range(2, get_dims))
    return get_reduced_axes