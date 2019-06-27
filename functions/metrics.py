'''
################################################################
# Functions - Metrics
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Extend metrics. These functions should not be used as train-
# ing losses.
# Version: 0.10 # 2019/6/13
# Comments:
#   Create this submodule, and finish signal_to_noise, 
#   correlation and jaccard_index.
################################################################
'''

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import control_flow_ops
from .others import get_channels

def signal_to_noise(y_true, y_pred, mode='snr', data_format=None, epsilon=1e-8):
    '''Signal-to-noise ratio. (metric)
    Calculate the signal-to-noise ratio. It support different modes.
    Arguments:
        mode:        (1)  snr: mean [ y_true^2 / (y_pred - y_true)^2 ]
                     (2) psnr: mean [ max( y_true^2 ) / (y_pred - y_true)^2 ]
        data_format: 'channels_first' or 'channels_last'. The default setting is generally
                     'channels_last' like other tf.keras APIs.
        epsilon:      used for avoid zero division.
    Input:
        y_true: label, tensor in any shape.
        y_pred: prediction, tensor in any shape.
    Output:
        scalar, the mean SNR.
    '''
    get_reduced_axes = get_channels(y_true, data_format)
    if mode.casefold() == 'psnr':
        signal = math_ops.reduce_max(gen_math_ops.square(y_true), axis=get_reduced_axes)
    else:
        signal = math_ops.reduce_sum(gen_math_ops.square(y_true), axis=get_reduced_axes)
    noise = math_ops.reduce_sum(gen_math_ops.square(y_true - y_pred), axis=get_reduced_axes) + epsilon
    coeff = (10.0/2.3025851) # 10/log_e(10)
    return coeff*math_ops.reduce_mean(gen_math_ops.log(math_ops.divide(signal, noise)))

def correlation(y_true, y_pred):
    '''Pearson correlation coefficient. (metric)
    The linear corrlation between y_true and y_pred is between -1.0 and 1.0, indicating
    positive correlation and negative correlation respectively. In particular, if the 
    correlation is 0.0, it means y_true and y_pred are irrelevant linearly.
    This function is implemented by:
        corr = [mean(y_true * y_pred) - mean(y_true) * mean(y_pred)] 
               / [ std(y_true) * std(m_y_pred) ]
    This function has been revised to prevent the division fail (0/0). When either y_true
    or y_pred is 0, the correlation would be set as 0.0.
    Input:
        y_true: label, tensor in any shape.
        y_pred: prediction, tensor in any shape.
    Output:
        scalar, the mean linear correlation between y_true and y_pred.
    '''
    m_y_true = math_ops.reduce_mean(y_true, axis=0)
    m_y_pred = math_ops.reduce_mean(y_pred, axis=0)
    s_y_true = gen_math_ops.sqrt(math_ops.reduce_mean(gen_math_ops.square(y_true), axis=0) - gen_math_ops.square(m_y_true))
    s_y_pred = gen_math_ops.sqrt(math_ops.reduce_mean(gen_math_ops.square(y_pred), axis=0) - gen_math_ops.square(m_y_pred))
    s_denom = s_y_true * s_y_pred
    s_numer = math_ops.reduce_mean(y_true * y_pred, axis=0) - m_y_true * m_y_pred
    s_index = gen_math_ops.greater(s_denom, 0)
    f1 = lambda: constant_op.constant(0.0)
    f2 = lambda: math_ops.reduce_mean(array_ops.boolean_mask(s_numer,s_index)/array_ops.boolean_mask(s_denom,s_index))
    return control_flow_ops.case([(math_ops.reduce_any(s_index), f2)], default=f1)

def jaccard_index(y_true, y_pred, data_format=None):
    '''Jaccard index, or Intersection over Union (IoU). (metric)
    The IoU is thought to be a better measurement to estimate the accuracy for segmentation.
    If both y_true and y_pred are binary, the intersection I(y_true, y_pred) shows the part
    where the prediction is correct, while the union U(y_true, y_pred) contains both correct
    prediction and wrong prediction. I/U shows the proportion of correct prediction.
    Compared to other error functions (like MSE), it is more concentrated on the part where
    y_true=1 or y_pred=1.
    This function is implemented by:
        jacc = logical_and(y_true, y_pred) / logical_or(y_true, y_pred)
    Arguments:
        data_format: 'channels_first' or 'channels_last'. The default setting is generally
                     'channels_last' like other tf.keras APIs.
    Input:
        y_true: label, tensor in any shape, should have at least 3 axes.
        y_pred: prediction, tensor in any shape, should have at least 3 axes.
    Output:
        scalar, the mean Jaccard index between y_true and y_pred over all channels.
    '''
    get_reduced_axes = get_channels(y_true, data_format)
    bin_y_true = gen_math_ops.greater(y_true, 0.5)
    bin_y_pred = gen_math_ops.greater(y_pred, 0.5)
    valNumer = gen_math_ops.logical_and(bin_y_pred, bin_y_true)
    valDomin = gen_math_ops.logical_or(bin_y_pred, bin_y_true)
    valNumer = math_ops.reduce_sum(math_ops.cast(valNumer, dtype=y_pred.dtype), axis=get_reduced_axes)
    valDomin = math_ops.reduce_sum(math_ops.cast(valDomin, dtype=y_pred.dtype), axis=get_reduced_axes)
    return math_ops.reduce_mean(math_ops.div_no_nan(valNumer, valDomin))