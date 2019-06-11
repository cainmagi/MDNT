'''
################################################################
# Layers - Modern network block unit
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
# In this module, we define the basic units that is used to
# construct network blocks in higher level. For example, the
# basic unit of a residual block is:
#   1. normalization;
#   2. activation;
#   3. convolution;
#   4. addition after several repeats.
# The norm-actv-conv structure is proved to be effective by 
# this paper:
#   https://arxiv.org/abs/1603.05027
# Version: 0.15 # 2019/6/6
# Comments:
#   Enable the units to work with group convolution.
# Version: 0.10 # 2019/5/29
# Comments:
#   Create this submodule.
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

from tensorflow.keras.layers import BatchNormalization, LeakyReLU, PReLU
from tensorflow.python.keras.layers.convolutional import Conv, Conv2DTranspose, Conv3DTranspose, UpSampling1D, UpSampling2D, UpSampling3D, ZeroPadding1D, ZeroPadding2D, ZeroPadding3D, Cropping1D, Cropping2D, Cropping3D
from .normalize import InstanceNormalization, GroupNormalization
from .conv import _GroupConv, _get_macro_conv

from .. import compat
if compat.COMPATIBLE_MODE:
    from tensorflow.python.keras.engine.base_layer import InputSpec
else:
    from tensorflow.python.keras.engine.input_spec import InputSpec

_check_dl_func = lambda a: all(ai==1 for ai in a)

class NACUnit(Layer):
    """Norm-Actv-Conv Unit.
    Abstract nD convolution unit (private, used as implementation base).
    `NACUnit` implements the operation:
    `output = conv( activation( normalization( x, gamma, beta ), alpha ), W)`
    Different from advanced convolutional layer, in this structure, the convolution
    is performed in the final step.
    Such a structure would be used to construct higher level blocks like residual
    network. We do not recommend users to use this structure to construct networks
    directly.
    Arguments for convolution:
        rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
        filters: Integer, the dimensionality of the output space (i.e. the number
            of filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            length of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        lgroups: Latent group number of group convolution. Only if set, use group
            convolution. The latent filter number of group convolution would
            be inferred by lfilters = filters // lgroups. Hence, filters should
            be a multiple of lgroups.
        padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
        data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, ..., channels)` while `channels_first` corresponds to
            inputs with shape `(batch, channels, ...)`.
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        kernel_initializer: An initializer for the convolution kernel.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The function
            must take as input the unprojected variable and must return the
            projected variable (which must have the same shape). Constraints are
            not safe to use when doing asynchronous distributed training.
        trainable: Boolean, if `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        name: A string, the name of the layer.
    Arguments for normalization:
        normalization: The normalization type, which could be
            (1) None:  do not use normalization and do not add biases.
            (2) bias:  apply biases instead of using normalization.
            (3) batch: use batch normalization.
            (4) inst : use instance normalization.
            (5) group: use group normalization.
            If using (2), the initializer, regularizer and constraint for
            beta would be applied to the bias of convolution.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
        groups (only for group normalization): Integer, the number of 
            groups for Group Normalization.
            Can be in the range [1, N] where N is the input dimension.
            The input dimension must be divisible by the number of groups.
    Arguments for activation:
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        activity_config: keywords for the parameters of activation
            function (only for lrelu).
    Arguments (others):
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
    """

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 lgroups=None,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 normalization='inst',
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 groups=32,
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 _high_activation=None,
                 _use_bias=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
          kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(NACUnit, self).__init__(trainable=trainable, name=name, activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        # Inherit from keras.layers._Conv
        self.rank = rank
        self.filters = filters
        self.lgroups = lgroups
        if (lgroups is not None) and (lgroups > 1):
            if filters % lgroups != 0:
                raise ValueError('To grouplize the output channels, the output channel number should be a multiple of group number (N*{0}), but given {1}'.format(self.lgroups, self.filters))
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        if (self.padding == 'causal' and not isinstance(self, AConv1D)):
            raise ValueError('Causal padding is only supported for `AConv1D`.')
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        # Inherit from mdnt.layers.normalize
        self.normalization = normalization
        if isinstance(normalization, str) and normalization in ('batch', 'inst', 'group'):
            self.use_bias = False
            self.gamma_initializer = initializers.get(gamma_initializer)
            self.gamma_regularizer = regularizers.get(gamma_regularizer)
            self.gamma_constraint = constraints.get(gamma_constraint)
        elif normalization:
            self.use_bias = True
            self.gamma_initializer = None
            self.gamma_regularizer = None
            self.gamma_constraint = None
        else:
            self.use_bias = False
            self.gamma_initializer = None
            self.gamma_regularizer = None
            self.gamma_constraint = None
        if _use_bias is not None:
            self._use_bias = _use_bias
        else:
            self._use_bias = self.use_bias
        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.groups = groups
        # Inherit from keras.engine.Layer
        if _high_activation is not None:
            activation = _high_activation
        self.high_activation = _high_activation
        self.use_plain_activation = False
        if isinstance(activation, str) and (activation.casefold() in ('prelu','lrelu')):
            self.activation = activations.get(None)
            self.high_activation = activation.casefold()
            self.activity_config = activity_config # dictionary passed to activation
            if activity_config is None:
                self.activity_config = dict()
        elif activation is not None:
            self.use_plain_activation = True
            self.activation = activations.get(activation)
            self.activity_config = None
        else:
            self.activation = activations.get(None)
            self.activity_config = None
        
        self.trainable = trainable
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        next_shape = input_shape
        # Normalization
        if self.normalization and (not self.use_bias):
            if self.normalization.casefold() == 'batch':
                self.layer_norm = BatchNormalization(gamma_initializer = self.gamma_initializer,
                                                     gamma_regularizer = self.gamma_regularizer,
                                                     gamma_constraint = self.gamma_constraint,
                                                     beta_initializer = self.beta_initializer,
                                                     beta_regularizer = self.beta_regularizer,
                                                     beta_constraint = self.beta_constraint,
                                                     trainable=self.trainable)
            elif self.normalization.casefold() == 'inst':
                self.layer_norm = InstanceNormalization(axis=-1,
                                                     gamma_initializer = self.gamma_initializer,
                                                     gamma_regularizer = self.gamma_regularizer,
                                                     gamma_constraint = self.gamma_constraint,
                                                     beta_initializer = self.beta_initializer,
                                                     beta_regularizer = self.beta_regularizer,
                                                     beta_constraint = self.beta_constraint,
                                                     trainable=self.trainable)
            elif self.normalization.casefold() == 'group':
                self.layer_norm = GroupNormalization(axis=-1, groups=self.groups,
                                                     gamma_initializer = self.gamma_initializer,
                                                     gamma_regularizer = self.gamma_regularizer,
                                                     gamma_constraint = self.gamma_constraint,
                                                     beta_initializer = self.beta_initializer,
                                                     beta_regularizer = self.beta_regularizer,
                                                     beta_constraint = self.beta_constraint,
                                                     trainable=self.trainable)
            self.layer_norm.build(next_shape)
            if compat.COMPATIBLE_MODE: # for compatibility
                self._trainable_weights.extend(self.layer_norm._trainable_weights)
            next_shape = self.layer_norm.compute_output_shape(next_shape)
        # Activation (if activation is a layer)
        if self.high_activation == 'prelu':
            shared_axes = tuple(range(1,self.rank+1))
            self.layer_actv = PReLU(shared_axes=shared_axes)
            self.layer_actv.build(next_shape)
            if compat.COMPATIBLE_MODE: # for compatibility
                self._trainable_weights.extend(self.layer_actv._trainable_weights)
            next_shape = self.layer_actv.compute_output_shape(next_shape)
        elif self.high_activation == 'lrelu':
            alpha = self.activity_config.get('alpha', 0.3)
            self.layer_actv = LeakyReLU(alpha=alpha)
            self.layer_actv.build(next_shape)
            next_shape = self.layer_actv.compute_output_shape(next_shape)
        # Perform convolution
        if self._use_bias:
            bias_initializer = self.beta_initializer
            bias_regularizer = self.beta_regularizer
            bias_constraint = self.beta_constraint
        else:
            bias_initializer = None
            bias_regularizer = None
            bias_constraint = None
        if (self.lgroups is not None) and (self.lgroups > 1):
            self.layer_conv = _GroupConv(rank=self.rank,
                                         lgroups=self.lgroups,
                                         lfilters=self.filters // self.lgroups,
                                         kernel_size=self.kernel_size,
                                         strides=self.strides,
                                         padding=self.padding,
                                         data_format=self.data_format,
                                         dilation_rate=self.dilation_rate,
                                         activation=None,
                                         use_bias=self.use_bias,
                                         bias_initializer=bias_initializer,
                                         bias_regularizer=bias_regularizer,
                                         bias_constraint=bias_constraint,
                                         kernel_initializer=self.kernel_initializer,
                                         kernel_regularizer=self.kernel_regularizer,
                                         kernel_constraint=self.kernel_constraint,
                                         trainable=self.trainable)
        else:
            self.layer_conv = Conv(rank = self.rank,
                                   filters = self.filters,
                                   kernel_size = self.kernel_size,
                                   strides = self.strides,
                                   padding = self.padding,
                                   data_format = self.data_format,
                                   dilation_rate = self.dilation_rate,
                                   activation = None,
                                   use_bias = self._use_bias,
                                   bias_initializer = bias_initializer,
                                   bias_regularizer = bias_regularizer,
                                   bias_constraint = bias_constraint,
                                   kernel_initializer = self.kernel_initializer,
                                   kernel_regularizer = self.kernel_regularizer,
                                   kernel_constraint = self.kernel_constraint,
                                   trainable=self.trainable)
        self.layer_conv.build(next_shape)
        if compat.COMPATIBLE_MODE: # for compatibility
            self._trainable_weights.extend(self.layer_conv._trainable_weights)
        super(NACUnit, self).build(input_shape)

    def call(self, inputs):
        outputs = inputs
        if self.normalization and (not self.use_bias):
            outputs = self.layer_norm(outputs)
        if self.high_activation in ('prelu', 'lrelu'):
            outputs = self.layer_actv(outputs)
        elif self.use_plain_activation:
            outputs = self.activation(outputs)  # pylint: disable=not-callable
        outputs = self.layer_conv(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        next_shape = input_shape
        if not self.use_bias:
            next_shape = self.layer_norm.compute_output_shape(next_shape)
        if self.high_activation in ('prelu', 'lrelu'):
            next_shape = self.layer_actv.compute_output_shape(next_shape)
        next_shape = self.layer_conv.compute_output_shape(next_shape)
        return next_shape
    
    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'lgroups': self.lgroups,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'normalization': self.normalization,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'groups': self.groups,
            'activation': activations.serialize(self.activation),
            'activity_config': self.activity_config,
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            '_high_activation': self.high_activation,
            '_use_bias': self._use_bias
        }
        base_config = super(NACUnit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class NACUnitTranspose(Layer):
    """Norm-Actv-ConvTrans Unit.
    Abstract nD transposed convolution unit (private, used as implementation base).
    `NACUnitTranspose` implements the operation:
    `output = convTranspose( activation( normalization( x, gamma, beta ), alpha ), W)`
    Different from advanced convolutional layer, in this structure, the transposed
    convolution is performed in the final step.
    Such a structure would be used to construct higher level blocks like residual
    network. We do not recommend users to use this structure to construct networks
    directly.
    Arguments for convolution:
        rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
        filters: Integer, the dimensionality of the output space (i.e. the number
            of filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            length of the convolution window.
        modenew: The realization mode of this layer, could be 
            (1) True: use upsampling-padding-conv work-flow to replace transposed 
                convolution.
            (2) False: use plain transposed convolution.
            Indeed, we recommend users to use this mode, however, users could
            deactivate this mode by switch the global switch in this module.
        strides: An integer or tuple/list of n integers,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        lgroups: Latent group number of group convolution. Only if set, use group
            convolution. The latent filter number of group convolution would
            be inferred by lfilters = filters // lgroups. Hence, filters should
            be a multiple of lgroups.
        padding: One of `"valid"`,  `"same"`.
        output_mshape: (Only avaliable for new-style API) An integer or tuple/list
            of the desired output shape. When setting this option, `output_padding`
            and `out_cropping` would be inferred from the input shape, which means
            users' options would be invalid for the following two options.
            A recommended method of using this method is applying such a scheme:
                `AConv(..., output_mshape=tensor.get_shape())`
        output_padding: An integer or tuple/list of n integers,
            specifying the amount of padding along the axes of the output tensor.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape would not be padded.
            (When using new-style API, the padding could be like ((a,b),(c,d),...) 
             so that you could be able to perform padding along different edges.)
        out_cropping: (Only avaliable for new-style API) An integer or tuple/list 
            of n integers, specifying the amount of cropping along the axes of the
            output tensor. The amount of output cropping along a given dimension must
            be lower than the stride along that same dimension.
            If set to `None` (default), the output shape would not be cropped.
            (Because this option only takes effect on new-style API, the cropping
             could be like ((a,b),(c,d),...) so that you could be able to perform
             cropping along different edges.)
        data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, ..., channels)` while `channels_first` corresponds to
            inputs with shape `(batch, channels, ...)`.
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        kernel_initializer: An initializer for the convolution kernel.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The function
            must take as input the unprojected variable and must return the
            projected variable (which must have the same shape). Constraints are
            not safe to use when doing asynchronous distributed training.
        trainable: Boolean, if `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        name: A string, the name of the layer.
    Arguments for normalization:
        normalization: The normalization type, which could be
            (1) None:  do not use normalization and do not add biases.
            (2) bias:  apply biases instead of using normalization.
            (3) batch: use batch normalization.
            (4) inst : use instance normalization.
            (5) group: use group normalization.
            If using (2), the initializer, regularizer and constraint for
            beta would be applied to the bias of convolution.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
        groups (only for group normalization): Integer, the number of 
            groups for Group Normalization.
            Can be in the range [1, N] where N is the input dimension.
            The input dimension must be divisible by the number of groups.
    Arguments for activation:
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        activity_config: keywords for the parameters of activation
            function (only for lrelu).
    Arguments (others):
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
    """

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 modenew=None,
                 lgroups=None,
                 strides=1,
                 padding='valid',
                 output_mshape=None,
                 output_padding=None,
                 output_cropping=None,
                 data_format=None,
                 dilation_rate=1,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 normalization='inst',
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 groups=32,
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 _high_activation=None,
                 _use_bias=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
          kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(NACUnitTranspose, self).__init__(trainable=trainable, name=name, activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        # Inherit from keras.layers._Conv
        self.rank = rank
        if modenew is not None:
            self.modenew = modenew
        else:
            self.modenew = _get_macro_conv()
        self.filters = filters
        self.lgroups = lgroups
        if (lgroups is not None) and (lgroups > 1):
            if not self.modenew:
                raise ValueError('Transposed group convolution does not support old API, please set modenew=True or configure the macro.')
            if filters % lgroups != 0:
                raise ValueError('To grouplize the output channels, the output channel number should be a multiple of group number (N*{0}), but given {1}'.format(self.lgroups, self.filters))
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        if (self.padding == 'causal' and not isinstance(self, AConv1D)):
            raise ValueError('Causal padding is only supported for `AConv1D`.')
        if output_padding is not None:
            if self.modenew:
                self.output_padding = output_padding
            else:
                self.output_padding = conv_utils.normalize_tuple(output_padding, rank, 'output_padding')
        else:
            self.output_padding = None
        self.output_mshape = None
        self.output_cropping = None
        if self.modenew:
            if output_mshape:
                if hasattr(output_mshape, 'as_list'):
                    self.output_mshape = output_mshape.as_list()
                else:
                    self.output_mshape = output_mshape
            if output_cropping:
                self.output_cropping = output_cropping
        self.data_format = conv_utils.normalize_data_format(data_format)
        if rank == 1 and self.data_format == 'channels_first':
            raise ValueError('Does not support channels_first data format for 1D case due to the limitation of upsampling method.')
        self.dilation_rate = conv_utils.normalize_tuple(
                dilation_rate, rank, 'dilation_rate')
        if (not _check_dl_func(self.dilation_rate)) and (not _check_dl_func(self.strides)):
            raise ValueError('Does not support dilation_rate when strides > 1.')
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        # Inherit from mdnt.layers.normalize
        self.normalization = normalization
        if isinstance(normalization, str) and normalization in ('batch', 'inst', 'group'):
            self.use_bias = False
            self.gamma_initializer = initializers.get(gamma_initializer)
            self.gamma_regularizer = regularizers.get(gamma_regularizer)
            self.gamma_constraint = constraints.get(gamma_constraint)
        elif normalization:
            self.use_bias = True
            self.gamma_initializer = None
            self.gamma_regularizer = None
            self.gamma_constraint = None
        else:
            self.use_bias = False
            self.gamma_initializer = None
            self.gamma_regularizer = None
            self.gamma_constraint = None
        if _use_bias is not None:
            self._use_bias = _use_bias
        else:
            self._use_bias = self.use_bias
        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.groups = groups
        # Inherit from keras.engine.Layer
        if _high_activation is not None:
            activation = _high_activation
        self.high_activation = _high_activation
        self.use_plain_activation = False
        if isinstance(activation, str) and (activation.casefold() in ('prelu','lrelu')):
            self.activation = activations.get(None)
            self.high_activation = activation.casefold()
            self.activity_config = activity_config
            if activity_config is None:
                self.activity_config = dict()
        elif activation is not None:
            self.use_plain_activation = True
            self.activation = activations.get(activation)
            self.activity_config = None
        else:
            self.activation = activations.get(None)
            self.activity_config = None
        
        self.trainable = trainable
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(self.rank + 2)
        next_shape = input_shape
        if self.normalization and (not self.use_bias):
            if self.normalization.casefold() == 'batch':
                self.layer_norm = BatchNormalization(gamma_initializer = self.gamma_initializer,
                                                     gamma_regularizer = self.gamma_regularizer,
                                                     gamma_constraint = self.gamma_constraint,
                                                     beta_initializer = self.beta_initializer,
                                                     beta_regularizer = self.beta_regularizer,
                                                     beta_constraint = self.beta_constraint,
                                                     trainable=self.trainable)
            elif self.normalization.casefold() == 'inst':
                self.layer_norm = InstanceNormalization(axis=-1,
                                                     gamma_initializer = self.gamma_initializer,
                                                     gamma_regularizer = self.gamma_regularizer,
                                                     gamma_constraint = self.gamma_constraint,
                                                     beta_initializer = self.beta_initializer,
                                                     beta_regularizer = self.beta_regularizer,
                                                     beta_constraint = self.beta_constraint,
                                                     trainable=self.trainable)
            elif self.normalization.casefold() == 'group':
                self.layer_norm = GroupNormalization(axis=-1, groups=self.groups,
                                                     gamma_initializer = self.gamma_initializer,
                                                     gamma_regularizer = self.gamma_regularizer,
                                                     gamma_constraint = self.gamma_constraint,
                                                     beta_initializer = self.beta_initializer,
                                                     beta_regularizer = self.beta_regularizer,
                                                     beta_constraint = self.beta_constraint,
                                                     trainable=self.trainable)
            self.layer_norm.build(next_shape)
            if compat.COMPATIBLE_MODE: # for compatibility
                self._trainable_weights.extend(self.layer_norm._trainable_weights)
            next_shape = self.layer_norm.compute_output_shape(next_shape)
        if self.high_activation == 'prelu':
            shared_axes = tuple(range(1,self.rank+1))
            self.layer_actv = PReLU(shared_axes=shared_axes)
            self.layer_actv.build(next_shape)
            if compat.COMPATIBLE_MODE: # for compatibility
                self._trainable_weights.extend(self.layer_actv._trainable_weights)
        elif self.high_activation == 'lrelu':
            alpha = self.activity_config.get('alpha', 0.3)
            self.layer_actv = LeakyReLU(alpha=alpha)
            self.layer_actv.build(next_shape)
        # Transposed convolution
        if self._use_bias:
            bias_initializer = self.beta_initializer
            bias_regularizer = self.beta_regularizer
            bias_constraint = self.beta_constraint
        else:
            bias_initializer = None
            bias_regularizer = None
            bias_constraint = None
        if self.modenew:
            # If setting output_mshape, need to infer output_padding & output_cropping
            if self.output_mshape is not None:
                if not isinstance(self.output_mshape, (list, tuple)):
                    l_output_mshape = self.output_mshape.as_list()
                else:
                    l_output_mshape = self.output_mshape
                l_output_mshape = l_output_mshape[1:-1]
                l_input_shape = input_shape.as_list()[1:-1]
                self.output_padding = []
                self.output_cropping = []
                for i in range(self.rank):
                    get_shape_diff = l_output_mshape[i] - l_input_shape[i]*self.strides[i]
                    if get_shape_diff > 0:
                        b_inf = get_shape_diff // 2
                        b_sup = b_inf + get_shape_diff % 2
                        self.output_padding.append((b_inf, b_sup))
                        self.output_cropping.append((0, 0))
                    elif get_shape_diff < 0:
                        get_shape_diff = -get_shape_diff
                        b_inf = get_shape_diff // 2
                        b_sup = b_inf + get_shape_diff % 2
                        self.output_cropping.append((b_inf, b_sup))
                        self.output_padding.append((0, 0))
                    else:
                        self.output_cropping.append((0, 0))
                        self.output_padding.append((0, 0))
                deFlag_padding = 0
                deFlag_cropping = 0
                for i in range(self.rank):
                    smp = self.output_padding[i]
                    if smp[0] == 0 and smp[1] == 0:
                        deFlag_padding += 1
                    smp = self.output_cropping[i]
                    if smp[0] == 0 and smp[1] == 0:
                        deFlag_cropping += 1
                if deFlag_padding >= self.rank:
                    self.output_padding = None
                else:
                    self.output_padding = tuple(self.output_padding)
                if deFlag_cropping >= self.rank:
                    self.output_cropping = None
                else:
                    self.output_cropping = tuple(self.output_cropping)
            if self.rank == 1:
                self.layer_uppool = UpSampling1D(size=self.strides[0])
                self.layer_uppool.build(next_shape)
                next_shape = self.layer_uppool.compute_output_shape(next_shape)
                if self.output_padding is not None:
                    self.layer_padding = ZeroPadding1D(padding=self.output_padding)[0] # Necessary for 1D case, because we need to pick (a,b) from ((a, b))
                    self.layer_padding.build(next_shape)
                    next_shape = self.layer_padding.compute_output_shape(next_shape)
                else:
                    self.layer_padding = None
            elif self.rank == 2:
                self.layer_uppool = UpSampling2D(size=self.strides, data_format=self.data_format)
                self.layer_uppool.build(next_shape)
                next_shape = self.layer_uppool.compute_output_shape(next_shape)
                if self.output_padding is not None:
                    self.layer_padding = ZeroPadding2D(padding=self.output_padding, data_format=self.data_format)
                    self.layer_padding.build(next_shape)
                    next_shape = self.layer_padding.compute_output_shape(next_shape)
                else:
                    self.layer_padding = None
            elif self.rank == 3:
                self.layer_uppool = UpSampling3D(size=self.strides, data_format=self.data_format)
                self.layer_uppool.build(next_shape)
                next_shape = self.layer_uppool.compute_output_shape(next_shape)
                if self.output_padding is not None:
                    self.layer_padding = ZeroPadding3D(padding=self.output_padding, data_format=self.data_format)
                    self.layer_padding.build(next_shape)
                    next_shape = self.layer_padding.compute_output_shape(next_shape)
                else:
                    self.layer_padding = None
            else:
                raise ValueError('Rank of the deconvolution should be 1, 2 or 3.')
            if (self.lgroups is not None) and (self.lgroups > 1):
                self.layer_conv = _GroupConv(rank=self.rank,
                                             lgroups=self.lgroups,
                                             lfilters=self.filters // self.lgroups,
                                             kernel_size=self.kernel_size,
                                             strides=1,
                                             padding=self.padding,
                                             data_format=self.data_format,
                                             dilation_rate=self.dilation_rate,
                                             activation=None,
                                             use_bias=self.use_bias,
                                             bias_initializer=bias_initializer,
                                             bias_regularizer=bias_regularizer,
                                             bias_constraint=bias_constraint,
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             trainable=self.trainable)
            else:
                self.layer_conv = Conv(rank = self.rank,
                                       filters = self.filters,
                                       kernel_size = self.kernel_size,
                                       strides = 1,
                                       padding = self.padding,
                                       data_format = self.data_format,
                                       dilation_rate = self.dilation_rate,
                                       activation = None,
                                       use_bias = self._use_bias,
                                       bias_initializer = bias_initializer,
                                       bias_regularizer = bias_regularizer,
                                       bias_constraint = bias_constraint,
                                       kernel_initializer = self.kernel_initializer,
                                       kernel_regularizer = self.kernel_regularizer,
                                       kernel_constraint = self.kernel_constraint,
                                       trainable=self.trainable)
            self.layer_conv.build(next_shape)
            if compat.COMPATIBLE_MODE: # for compatibility
                self._trainable_weights.extend(self.layer_conv._trainable_weights)
            next_shape = self.layer_conv.compute_output_shape(next_shape)
            if self.output_cropping is not None:
                if self.rank == 1:
                    self.layer_cropping = Cropping1D(cropping=self.output_cropping)[0]
                elif self.rank == 2:
                    self.layer_cropping = Cropping2D(cropping=self.output_cropping)
                elif self.rank == 3:
                    self.layer_cropping = Cropping3D(cropping=self.output_cropping)
                else:
                    raise ValueError('Rank of the deconvolution should be 1, 2 or 3.')
                self.layer_cropping.build(next_shape)
                next_shape = self.layer_cropping.compute_output_shape(next_shape)
            else:
                self.layer_cropping = None
        else:
            if self.rank == 1:
                next_shape = next_shape[:1].concatenate([1,]).concatenate(next_shape[1:])
                if self.output_padding is None:
                    output_padding = None
                else:
                    output_padding = (1, *self.output_padding)
                self.layer_deconv = Conv2DTranspose(filters = self.filters,
                                    kernel_size = (1, *self.kernel_size),
                                    strides = (1, *self.strides),
                                    padding = self.padding,
                                    output_padding = output_padding,
                                    data_format = self.data_format,
                                    dilation_rate = (1, *self.dilation_rate),
                                    activation = None,
                                    use_bias = self._use_bias,
                                    bias_initializer = bias_initializer,
                                    bias_regularizer = bias_regularizer,
                                    bias_constraint = bias_constraint,
                                    kernel_initializer = self.kernel_initializer,
                                    kernel_regularizer = self.kernel_regularizer,
                                    kernel_constraint = self.kernel_constraint,
                                    trainable=self.trainable)
            elif self.rank == 2:
                self.layer_deconv = Conv2DTranspose(filters = self.filters,
                                    kernel_size = self.kernel_size,
                                    strides = self.strides,
                                    padding = self.padding,
                                    output_padding = self.output_padding,
                                    data_format = self.data_format,
                                    dilation_rate = self.dilation_rate,
                                    activation = None,
                                    use_bias = self._use_bias,
                                    bias_initializer = bias_initializer,
                                    bias_regularizer = bias_regularizer,
                                    bias_constraint = bias_constraint,
                                    kernel_initializer = self.kernel_initializer,
                                    kernel_regularizer = self.kernel_regularizer,
                                    kernel_constraint = self.kernel_constraint,
                                    trainable=self.trainable)
            elif self.rank == 3:
                self.layer_deconv = Conv3DTranspose(filters = self.filters,
                                    kernel_size = self.kernel_size,
                                    strides = self.strides,
                                    padding = self.padding,
                                    output_padding = self.output_padding,
                                    data_format = self.data_format,
                                    activation = None,
                                    use_bias = self._use_bias,
                                    bias_initializer = bias_initializer,
                                    bias_regularizer = bias_regularizer,
                                    bias_constraint = bias_constraint,
                                    kernel_initializer = self.kernel_initializer,
                                    kernel_regularizer = self.kernel_regularizer,
                                    kernel_constraint = self.kernel_constraint,
                                    trainable=self.trainable)
            else:
                raise ValueError('Rank of the deconvolution should be 1, 2 or 3.')
            self.layer_deconv.build(next_shape)
            if compat.COMPATIBLE_MODE: # for compatibility
                self._trainable_weights.extend(self.layer_deconv._trainable_weights)
            next_shape = self.layer_deconv.compute_output_shape(next_shape)
            if self.rank == 1:
                next_shape = next_shape[:1].concatenate(next_shape[2:])
        
        super(NACUnitTranspose, self).build(input_shape)

    def call(self, inputs):
        outputs = inputs
        if self.normalization and (not self.use_bias):
            outputs = self.layer_norm(outputs)
        if self.high_activation in ('prelu', 'lrelu'):
            outputs = self.layer_actv(outputs)
        if self.use_plain_activation:
            return self.activation(outputs)  # pylint: disable=not-callable
        if self.modenew: # Apply new architecture
            outputs = self.layer_uppool(outputs)
            if self.layer_padding is not None:
                outputs = self.layer_padding(outputs)
            if self.layer_cropping is not None:
                outputs = self.layer_cropping(outputs)
            outputs = self.layer_conv(outputs)
        else: # Use classic method
            if self.rank == 1:
                outputs = array_ops.expand_dims(outputs, axis=1)
            outputs = self.layer_deconv(outputs)
            if self.rank == 1:
                outputs = array_ops.squeeze(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(self.rank + 2)
        next_shape = input_shape
        if not self.use_bias:
            next_shape = self.layer_norm.compute_output_shape(next_shape)
        if self.high_activation in ('prelu', 'lrelu'):
            next_shape = self.layer_actv.compute_output_shape(next_shape)
        if self.modenew: # Apply new architecture
            next_shape = self.layer_uppool.compute_output_shape(next_shape)
            if self.layer_padding is not None:
                next_shape = self.layer_padding.compute_output_shape(next_shape)
            if self.layer_cropping is not None:
                next_shape = self.layer_cropping.compute_output_shape(next_shape)
            next_shape = self.layer_conv.compute_output_shape(next_shape)
        else: # Use classic method
            if self.rank == 1:
                next_shape = next_shape[:1].concatenate([1,]).concatenate(next_shape[1:])
            next_shape = self.layer_conv.compute_output_shape(next_shape)
            if self.rank == 1:
                next_shape = next_shape[:1].concatenate(next_shape[2:])
        return next_shape
    
    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'lgroups': self.lgroups,
            'padding': self.padding,
            'output_mshape': self.output_mshape,
            'output_padding': self.output_padding,
            'output_cropping': self.output_cropping,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'normalization': self.normalization,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'groups': self.groups,
            'activation': activations.serialize(self.activation),
            'activity_config': self.activity_config,
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            '_high_activation': self.high_activation,
            '_use_bias': self._use_bias
        }
        base_config = super(NACUnitTranspose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))