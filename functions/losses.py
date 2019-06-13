'''
################################################################
# Functions - Losses
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Extend loss functions. These functions could serve as both
# losses and metrics.
# Version: 0.10 # 2019/6/13
# Comments:
#   Create this submodule, and finish linear_jaccard_loss
#   and lovasz_jaccard_loss.
################################################################
'''

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_array_ops

from .others import get_channels

from functools import reduce
def _get_prod(x):
    try:
        return reduce(lambda a,b:a*b, x)
    except TypeError:
        return x

def linear_jaccard_loss(y_true, y_pred, data_format=None):
    '''Simple linear approximation for Jaccard index, 
           or Intersection over Union (IoU). (loss)
    This function is a simple and linear approximation for IoU. The main idea is:
        1. logical_and(y_true * y_pred) could be approximated by y_true * y_pred;
        2. logical_or(y_true * y_pred) could be approximated by 
           y_true + y_pred - y_true * y_pred.
    Such an approximation could ensure that when both y_true and y_pred are
    binary, this approximation would returns the exact same value compared to
    the original metric, IoU.
    It has been proved that when both x, y in [0, 1], there is
        x * y < x + y - x * y.
    To learn more about IoU, please check mdnt.metrics.jaccard_index.
    This function is implemented by:
        appx_jacc = 1 - [ sum(y_true * y_pred) ] / [ sum(y_true + y_pred - y_true * y_pred) ]
    We use unsafe division in the above equation. When x / y = 0, the unsafe division would
    returns 0.
    NOTE THAT THIS IMPLEMENTATION IS THE COMPLEMENTARY OF JACCARD INDEX.
    Arguments:
        data_format: 'channels_first' or 'channels_last'. The default setting is generally
                     'channels_last' like other tf.keras APIs.
    Input:
        y_true: label, tensor in any shape, should have at least 3 axes.
        y_pred: prediction, tensor in any shape, should have at least 3 axes.
    Output:
        scalar, the approximated and complementary mean Jaccard index between y_true and
        y_pred over all channels.
    '''
    get_reduced_axes = get_channels(y_true, data_format)
    get_mul = y_true * y_pred
    valNumer = math_ops.reduce_sum(get_mul, axis=get_reduced_axes)
    valDomin = math_ops.reduce_sum(y_true + y_pred - get_mul, axis=get_reduced_axes)
    return 1-math_ops.reduce_mean(math_ops.div_no_nan(valNumer, valDomin))

def _lovasz_jaccard_flat(errors, y_true):
    '''PRIVATE: calculate lovasz extension for jaccard index along a vector.
    Input:
        errors: error vector (should be in 0~1).
        y_true: labels.
    Output:
        scalar: the jaccard index calculated on the input vector. 
    '''
    p = errors.get_shape().as_list()
    if len(p) != 1:
        raise ValueError('Input should be vectors (1D).')
    p = p[0]
    bin_y_true = math_ops.cast(gen_math_ops.greater(y_true, 0.5), dtype=errors.dtype)
    error_ind = sort_ops.argsort(errors, direction='DESCENDING')
    sorted_errors = array_ops.gather(errors, error_ind)
    sorted_labels = array_ops.gather(bin_y_true, error_ind)
    get_sum = math_ops.reduce_sum(sorted_labels)
    intersection = get_sum - math_ops.cumsum(sorted_labels)
    union = get_sum + math_ops.cumsum(1.0 - sorted_labels)
    g = 1.0 - math_ops.div_no_nan(intersection, union)
    if p > 1:
        g = array_ops.concat((g[0:1], g[1:] - g[:-1]), axis=0)
    return math_ops.reduce_sum(sorted_errors*gen_array_ops.stop_gradient(g))

def lovasz_jaccard_loss(y_true, y_pred, error_func=None, data_format=None):
    '''Lovasz extension for Jaccard index, or Intersection over Union (IoU). (loss)
    This function applies the theory of Lovasz extension. Although Lovasz extension could
    be used on any submodular set function, the implementation is aimed at constructing
    the trainable complementary of IoU.
    To learn more about this topic, please refer:
        The Lovasz-Softmax loss: A tractable surrogate for the optimization of the 
        intersection-over-union measure in neural networks
        https://arxiv.org/abs/1705.08790
    This implementation is not adapted from the author's github codes. It computes the
    Lovasz loss on each channel of each sample independently, and then calculate the
    average value.
    NOTE THAT THIS IMPLEMENTATION IS THE COMPLEMENTARY OF JACCARD INDEX.
    Arguments:
        error_func:  the function that is used to calculate errors. If set None, would use
                     L1 norm (linear interpolation).
        data_format: 'channels_first' or 'channels_last'. The default setting is generally
                     'channels_last' like other tf.keras APIs.
    Input:
        y_true: label, tensor in any shape, should have at least 3 axes.
        y_pred: prediction, tensor in any shape, should have at least 3 axes.
    Output:
        scalar, the approximated and complementary mean Jaccard index between y_true and
        y_pred over all channels.
    '''
    get_shapes = y_true.get_shape().as_list()
    get_dims = len(get_shapes)
    if get_dims < 3:
        raise ValueError('The input tensor should has channel dimension, i.e. it should have at least 3 axes.')
    if data_format is None:
        data_format = K.image_data_format()
    if data_format == 'channels_last':
        get_permute_axes = (0, get_dims-1, *range(1, get_dims-1))
        get_length = _get_prod(get_shapes[1:-1])
        y_true = array_ops.transpose(y_true, perm=get_permute_axes) # switch to channels_first
        y_pred = array_ops.transpose(y_pred, perm=get_permute_axes)
    else:
        get_length = _get_prod(get_shapes[2:])
    y_true = gen_array_ops.reshape([-1, get_length])
    y_pred = gen_array_ops.reshape([-1, get_length])
    if error_func is None:
        error_func = losses.mean_absolute_error
    def split_process(inputs):
        get_y_true, get_y_pred = inputs
        get_errors = error_func(get_y_true, get_y_pred)
        return _lovasz_jaccard_flat(get_errors, get_y_true)
    get_losses = functional_ops.map_fn(split_process, (y_true, y_pred), dtype=y_pred.dtype)
    return math_ops.reduce_mean(get_losses)