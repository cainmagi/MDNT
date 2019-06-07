'''
################################################################
# Layers - Modern convolutional layers (deprecated)
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# We store the failed versions of APIs for .conv here.
# Version: 0.10 # 2019/6/7
# Comments:
#   A failed try for quick group convolution (QGroupConv), move
#   it to deprecated.
################################################################
'''

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import math_ops

from tensorflow.keras.layers import BatchNormalization, LeakyReLU, PReLU
from tensorflow.python.keras.layers.convolutional import Conv, Conv2DTranspose, Conv3DTranspose, UpSampling1D, UpSampling2D, UpSampling3D, ZeroPadding1D, ZeroPadding2D, ZeroPadding3D, Cropping1D, Cropping2D, Cropping3D
from .normalize import InstanceNormalization, GroupNormalization

from .. import compat
if compat.COMPATIBLE_MODE:
    from tensorflow.python.keras.engine.base_layer import InputSpec
else:
    from tensorflow.python.keras.engine.input_spec import InputSpec

NEW_CONV_TRANSPOSE = True
USE_QUICK_GCONV = False

def _get_macro_conv(key='NEW_CONV_TRANSPOSE'):
    if key == 'USE_QUICK_GCONV':
        return USE_QUICK_GCONV
    else:
        return NEW_CONV_TRANSPOSE

class _QGroupConv(_GroupConv):
    """Quick computing version for abstract nD group convolution layer.
    This is the quick computing version of the convolution.
    The work flow of `GroupConv` could be viewed as
        output = concat (i=1~G) ( convND(input[group_i]) )
    which means if we have G groups, we need to compute the `convND` op for G times.
    The original implementation calls operator `convND` for many times, which is
    inefficient. To solve this problem, we use such a work flow:
        output = sum (i=1~G) ( depth_convND(input)[group_i] )
    The difference is, we only need to call `depth_convND` (tf.nn.depthwise_conv2d) 
    once. Furthermore, if we apply tf.reshape and tf.sum, we could also calculate 
    the sum operator once. This is why we could use the above method to improve the
    efficiency.
    However, since there is only tf.nn.depthwise_conv2d in tensorflow, we could not
    use it to calculate GroupConv3D. But we could still calculate GroupConv1D by
    reducing the 2D convolution to 1D case.
    To learn more about group convolution, see the docstring for `GroupConv`.
    Arguments:
        rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
              (rank > 2 is not allowed.)
        lgroups: Integer, the group number of the latent convolution branch. The
            number of filters in the whole latent space is lgroups * lfilters.
        lfilters: Integer, the dimensionality in each the lattent group (i.e. the
            number of filters in each latent convolution branch).
        kernel_size: An integer or tuple/list of n integers, specifying the
            length of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, ..., channels)` while `channels_first` corresponds to
            inputs with shape `(batch, channels, ...)`.
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function. Set it to None to maintain a
            linear activation.
        use_bias: Boolean, whether the layer uses a bias.
        kernel_initializer: An initializer for the convolution kernel.
        bias_initializer: An initializer for the bias vector. If None, the default
            initializer will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The function
            must take as input the unprojected variable and must return the
            projected variable (which must have the same shape). Constraints are
            not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
        trainable: Boolean, if `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        name: A string, the name of the layer.
    """

    def __init__(self, rank,
                 lgroups,
                 lfilters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(_GroupConv, self).__init__(
                trainable=trainable,
                name=name,
                activity_regularizer=regularizers.get(activity_regularizer),
                **kwargs)
        self.rank = rank
        if rank > 2:
            raise ValueError('The quick group convolution does not support 3D or any higher dimension.')
        initRank = rank
        self.lgroups = lgroups
        self.lfilters = lfilters
        self.kernel_size = conv_utils.normalize_tuple(
                kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        if (self.padding == 'causal' and not isinstance(self, (Conv1D, SeparableConv1D))):
            raise ValueError('Causal padding is only supported for `Conv1D` and ``SeparableConv1D`.')
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
             dilation_rate, rank, 'dilation_rate')
        if rank == 1: # when rank=1, expand the tuples to 2D case.
            self.kernel_size = (1, *self.kernel_size)
            self.strides = (1, *self.strides)
            self.dilation_rate = (1, *self.dilation_rate)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

        self.group_input_dim = None
        self.exp_dim_pos = None

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
            self._data_format = 'NCHW'
            if self.rank == 1:
                self.exp_dim_pos = 2
        else:
            channel_axis = -1
            if self.rank == 1:
                self.exp_dim_pos = 1
            self._data_format = 'NHWC'
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        if input_dim % self.lgroups != 0:
            raise ValueError('To grouplize the input channels, the input channel number should be a multiple of group number (N*{0}), but given {1}'.format(self.lgroups, input_dim))
        self.group_input_dim = input_dim // self.lgroups
        self._strides = (1, *self.strides, 1)
        kernel_shape = self.kernel_size + (input_dim, self.lfilters)

        self.kernel = self.add_weight(
                name='kernel',
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
                dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.lfilters * self.lgroups,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
        if self.padding == 'causal':
            self.op_padding = 'valid'
        else:
            self.op_padding = self.padding
        self.built = True

    def call(self, inputs):
        if self.rank == 1:
            inputs = array_ops.expand_dims(inputs, axis=self.exp_dim_pos)
        outputs= nn_impl.depthwise_conv2d(input=inputs,
                                          filter=self.kernel,
                                          strides=self._strides,
                                          padding=self.op_padding.upper(),
                                          rate=self.dilation_rate,
                                          data_format=self._data_format)
        # Grouplize the output channels.
        r2_outputs_shape = outputs.get_shape().as_list()
        if self.data_format == 'channels_first':
            #get_oshape = r2_outputs_shape[:1].concatenate([self.lgroups*self.lfilters, self.group_input_dim]).concatenate(r2_outputs_shape[2:])
            get_oshape = [-1, self.lgroups*self.lfilters, self.group_input_dim, *r2_outputs_shape[2:]]
            outputs = array_ops.reshape(outputs,  get_oshape)
            outputs = math_ops.reduce_sum(outputs, axis=1, keepdims=False)
        else:
            #get_oshape = r2_outputs_shape[:-1].concatenate([self.lgroups*self.lfilters, self.group_input_dim])
            get_oshape = [-1, *r2_outputs_shape[1:-1], self.lgroups*self.lfilters, self.group_input_dim]
            outputs = array_ops.reshape(outputs, get_oshape)
            outputs = math_ops.reduce_sum(outputs, axis=-1, keepdims=False)
        if self.rank == 1:
            outputs = array_ops.squeeze(outputs, axis=self.exp_dim_pos)
        outputs_list = []

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.lfilters * self.lgroups, 1))
                    outputs += bias
                if self.rank == 2:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs