'''
################################################################
# Layers - Inception-v4 blocks
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Inception blocks and their deconvolutional versions.
# The inception block is modified from the basic unit of Google-
# LeNet. Compared to the residual block, it is more concentrated
# on the low-pass features of the input layer. Hence, in some 
# applications where the network serves as the low-pass filter,
# the inception block should be more effective.
# As mentioned in title, inception block has been modified for
# several times. The newest version is v4. The theory could be
# referred here:
# Inception-v4, Inception-ResNet and the Impact of Residual 
# Connections on Learning
#   https://arxiv.org/abs/1602.07261
# In this module, we would implement the inception-v4 block and
# popularize it into more generic cases. The abstract structure
# of this block is:
#   -> Average Pool -> Conv x1              -> \
#   -> Conv x1                              -> |
#   -> Conv x1 -> Conv xM                   -> |-Cup
#   -> Conv x1 -> Conv xM -> Conv xM        -> |
#   -> Conv x1 -> Conv xM -> Conv xM -> ... -> /
# Inception-v4 also has residual structure. The macro architec-
# ture of such a scheme is as
#   Input + "Inception-v4 plain block"
# We have also implemented the InceptRes-v4 in this module.
# Version: 0.45 # 2019/6/12
# Comments:
# 1. Enable all layers in this module to work with dropout.
# 2. Strengthen the compatibility.
# Version: 0.40 # 2019/6/9
# Comments:
#   Propose a new "inception plus" layer (`Inceptplus`) in this
#   module.
# Version: 0.34 # 2019/6/5
# Comments:
#   Fix a bug that using repeating biases on inception-residual
#   layers.
# Version: 0.33 # 2019/6/4
# Comments:
#   Minor change for default settings for network parameters.
# Version: 0.32 # 2019/6/4
# Comments:
#   Improve the quality of the codes.
# Version: 0.30 # 2019/6/4
# Comments:
#   Finish the Inceptres layers.
# Version: 0.20 # 2019/6/1
# Comments:
#   Finish the basic Inception layers, the InceptRes layers still
#   requires to be finished.
# Version: 0.10 # 2019/5/31
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

from tensorflow.python.keras.layers.convolutional import Conv, UpSampling1D, UpSampling2D, UpSampling3D, ZeroPadding1D, ZeroPadding2D, ZeroPadding3D, Cropping1D, Cropping2D, Cropping3D
from tensorflow.python.keras.layers.pooling import MaxPooling1D, MaxPooling2D, MaxPooling3D, AveragePooling1D, AveragePooling2D, AveragePooling3D
from tensorflow.python.keras.layers.merge import Add, Concatenate, Subtract
from .unit import NACUnit
from .conv import _AConv

from .. import compat
if compat.COMPATIBLE_MODE:
    from tensorflow.python.keras.engine.base_layer import InputSpec
else:
    from tensorflow.python.keras.engine.input_spec import InputSpec

_check_dl_func = lambda a: all(ai==1 for ai in a)

class _Inception(Layer):
    """Modern inception layer.
    Abstract nD inception layer (private, used as implementation base).
    `_Inception` implements the operation:
        `output = concat (i=0~D) ConvBranch(D, input)`
    where `ConvBranch` means D-1 times convolutional layers.
    To be specific, when D=0, this branch is low-pass filtered by pooling layer.
    In some cases, the first term may not need to be convoluted.
    Such a structure is adapted from:
        Inception-v4, Inception-ResNet and the Impact of Residual 
        Connections on Learning
            https://arxiv.org/abs/1602.07261
    The implementation here is not exactly the same as original paper. The main 
    difference includes
        1. The main structure is only borrowed from inception-A block.
        2. The down sampling layer is also implemented in this class 
           (if set strides)
        3. We do not invoke the low-rank decomposition for the conv. kernels.
        4. We borrow the idea of residual-v2 block and change the order of some
           layers. 
    Arguments for inception block:
        rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of n integers, specifying the
            length of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
                 depth, ofilters,
                 kernel_size,
                 lfilters=None,
                 strides=1,
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
                 dropout=None,
                 dropout_rate=0.3,
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 _high_activation=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
          kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(_Inception, self).__init__(trainable=trainable, name=name, **kwargs)
        # Inherit from keras.layers._Conv
        self.rank = rank
        self.depth = depth
        if depth < 1:
            raise ValueError('The depth of the inception block should be >= 1.')
        if ofilters % (depth+1) != 0:
            raise ValueError('The output filter number should be the multiple of (depth+1), i.e. N * {0}'.format(depth+1))
        self.ofilters = ofilters // (self.depth + 1)
        self.lfilters = lfilters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')
        if (not _check_dl_func(self.dilation_rate)) and (not _check_dl_func(self.strides)):
            raise ValueError('Does not support dilation_rate when strides > 1.')
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        # Inherit from mdnt.layers.normalize
        self.normalization = normalization
        if isinstance(normalization, str) and normalization in ('batch', 'inst', 'group'):
            self.gamma_initializer = initializers.get(gamma_initializer)
            self.gamma_regularizer = regularizers.get(gamma_regularizer)
            self.gamma_constraint = constraints.get(gamma_constraint)
        else:
            self.gamma_initializer = None
            self.gamma_regularizer = None
            self.gamma_constraint = None
        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.groups = groups
        # Inherit from mdnt.layers.dropout
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        # Inherit from keras.engine.Layer
        if _high_activation is not None:
            activation = _high_activation
        self.high_activation = _high_activation
        if isinstance(activation, str) and (activation.casefold() in ('prelu','lrelu')):
            self.activation = activations.get(None)
            self.high_activation = activation.casefold()
            self.activity_config = activity_config # dictionary passed to activation
        elif activation is not None:
            self.activation = activations.get(activation)
            self.activity_config = None
        self.sub_activity_regularizer=regularizers.get(activity_regularizer)

        # Reserve for build()
        self.channelIn = None
        
        self.trainable = trainable
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(self.rank + 2)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        self.channelIn = int(input_shape[channel_axis])
        if self.lfilters is None:
            self.lfilters = max( 1, self.channelIn // 2 )
        # Consider the branch zero
        if not _check_dl_func(self.strides):
            if self.rank == 1:
                self.layer_branch_zero = MaxPooling1D(pool_size=self.kernel_size, strides=self.strides, padding='same', data_format=self.data_format)
            elif self.rank == 2:
                self.layer_branch_zero = MaxPooling2D(pool_size=self.kernel_size, strides=self.strides, padding='same', data_format=self.data_format)
            elif self.rank == 3:
                self.layer_branch_zero = MaxPooling3D(pool_size=self.kernel_size, strides=self.strides, padding='same', data_format=self.data_format)
            else:
                raise ValueError('Rank of the inception should be 1, 2 or 3.')
        else:
            if self.rank == 1:
                self.layer_branch_zero = AveragePooling1D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            elif self.rank == 2:
                self.layer_branch_zero = AveragePooling2D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            elif self.rank == 3:
                self.layer_branch_zero = AveragePooling3D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            else:
                raise ValueError('Rank of the inception should be 1, 2 or 3.')
        self.layer_branch_zero.build(input_shape)
        zero_shape = self.layer_branch_zero.compute_output_shape(input_shape)
        # If channel does not match, use linear conv.
        if self.channelIn != self.ofilters:
            self.layer_branch_zero_map = _AConv(rank = self.rank,
                        filters = self.ofilters,
                        kernel_size = 1,
                        strides = 1,
                        padding = 'same',
                        data_format = self.data_format,
                        dilation_rate = 1,
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        normalization=self.normalization,
                        beta_initializer=self.beta_initializer,
                        gamma_initializer=self.gamma_initializer,
                        beta_regularizer=self.beta_regularizer,
                        gamma_regularizer=self.gamma_regularizer,
                        beta_constraint=self.beta_constraint,
                        gamma_constraint=self.gamma_constraint,
                        groups=self.groups,
                        activation=None,
                        activity_config=None,
                        activity_regularizer=None,
                        _high_activation=None,
                        trainable=self.trainable)
            self.layer_branch_zero_map.build(zero_shape)
            compat.collect_properties(self, self.layer_branch_zero_map) # for compatibility
            zero_shape = self.layer_branch_zero_map.compute_output_shape(zero_shape)
        else:
            self.layer_branch_zero_map = None
        # Consider the branch one
        if (not _check_dl_func(self.strides)) or self.channelIn != self.ofilters:
            self.layer_branch_one = _AConv(rank = self.rank,
                        filters = self.ofilters,
                        kernel_size = 1,
                        strides = self.strides,
                        padding = 'same',
                        data_format = self.data_format,
                        dilation_rate = 1,
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        normalization=self.normalization,
                        beta_initializer=self.beta_initializer,
                        gamma_initializer=self.gamma_initializer,
                        beta_regularizer=self.beta_regularizer,
                        gamma_regularizer=self.gamma_regularizer,
                        beta_constraint=self.beta_constraint,
                        gamma_constraint=self.gamma_constraint,
                        groups=self.groups,
                        activation=None,
                        activity_config=None,
                        activity_regularizer=None,
                        _high_activation=None,
                        trainable=self.trainable)
            self.layer_branch_one.build(input_shape)
            compat.collect_properties(self, self.layer_branch_one) # for compatibility
            one_shape = self.layer_branch_one.compute_output_shape(input_shape)
        else:
            self.layer_branch_one = None
            one_shape = input_shape
        # Consider branches with depth, with dropout
        self.layer_dropout = return_dropout(self.dropout, self.dropout_rate, axis=channel_axis, rank=self.rank)
        if self.layer_dropout is not None:
            self.layer_dropout.build(input_shape)
            depth_shape = self.layer_dropout.compute_output_shape(input_shape)
        else:
            depth_shape = input_shape
        depth_shape_list = []
        for D in range(self.depth-1):
            layer_middle_first = NACUnit(rank = self.rank,
                          filters = self.lfilters,
                          kernel_size = 1,
                          strides = self.strides,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
            layer_middle_first.build(depth_shape)
            compat.collect_properties(self, layer_middle_first) # for compatibility
            branch_shape = layer_middle_first.compute_output_shape(depth_shape)
            setattr(self, 'layer_middle_D{0:02d}_00'.format(D+2), layer_middle_first)
            for i in range(D):
                layer_middle = NACUnit(rank = self.rank,
                          filters = self.lfilters,
                          kernel_size = self.kernel_size,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = self.dilation_rate,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
                layer_middle.build(branch_shape)
                compat.collect_properties(self, layer_middle) # for compatibility
                branch_shape = layer_middle.compute_output_shape(branch_shape)
                setattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, i+1), layer_middle)
            layer_middle_last = NACUnit(rank = self.rank,
                          filters = self.ofilters,
                          kernel_size = self.kernel_size,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = self.dilation_rate,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
            layer_middle_last.build(branch_shape)
            compat.collect_properties(self, layer_middle_last) # for compatibility
            branch_shape = layer_middle_last.compute_output_shape(branch_shape)
            setattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, D+1), layer_middle_last)
            depth_shape_list.append(branch_shape)
        if self.data_format == 'channels_first':
            self.layer_merge = Concatenate(axis=1)
        else:
            self.layer_merge = Concatenate()
        self.layer_merge.build([zero_shape, one_shape, *depth_shape_list])
        super(_Inception, self).build(input_shape)

    def call(self, inputs):
        branch_zero = self.layer_branch_zero(inputs)
        if self.layer_branch_zero_map is not None:
            branch_zero = self.layer_branch_zero_map(branch_zero)
        if self.layer_branch_one is not None:
            branch_one = self.layer_branch_one(inputs)
        else:
            branch_one = inputs
        if self.layer_dropout is not None:
            depth_input = self.layer_dropout(inputs)
        else:
            depth_input = inputs
        branch_middle_list = []
        for D in range(self.depth-1):
            layer_middle_first = getattr(self, 'layer_middle_D{0:02d}_00'.format(D+2))
            branch_middle = layer_middle_first(depth_input)
            for i in range(D):
                layer_middle = getattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, i+1))
                branch_middle = layer_middle(branch_middle)
            layer_middle_last = getattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, D+1))
            branch_middle = layer_middle_last(branch_middle)
            branch_middle_list.append(branch_middle)
        outputs = self.layer_merge([branch_zero, branch_one, *branch_middle_list])
        return outputs

    def compute_output_shape(self, input_shape):
        branch_zero_shape = self.layer_branch_zero.compute_output_shape(input_shape)
        if self.layer_branch_zero_map is not None:
            branch_zero_shape = self.layer_branch_zero_map.compute_output_shape(branch_zero_shape)
        if self.layer_branch_one is not None:
            branch_one_shape = self.layer_branch_one.compute_output_shape(input_shape)
        else:
            branch_one_shape = input_shape
        if self.layer_dropout is not None:
            depth_input_shape = self.layer_dropout.compute_output_shape(input_shape)
        else:
            depth_input_shape = input_shape
        branch_middle_shape_list = []
        for D in range(self.depth-1):
            layer_middle_first = getattr(self, 'layer_middle_D{0:02d}_00'.format(D+2))
            branch_middle_shape = layer_middle_first.compute_output_shape(depth_input_shape)
            for i in range(D):
                layer_middle = getattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, i+1))
                branch_middle_shape = layer_middle.compute_output_shape(branch_middle_shape)
            layer_middle_last = getattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, D+1))
            branch_middle_shape = layer_middle_last.compute_output_shape(branch_middle_shape)
            branch_middle_shape_list.append(branch_middle_shape)
        next_shape = self.layer_merge.compute_output_shape([branch_zero_shape, branch_one_shape, *branch_middle_shape_list])
        return next_shape
    
    def get_config(self):
        config = {
            'depth': self.depth,
            'ofilters': self.ofilters * (self.depth + 1),
            'lfilters': self.lfilters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
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
            'dropout': self.dropout,
            'dropout_rate': self.dropout_rate,
            'activation': activations.serialize(self.activation),
            'activity_config': self.activity_config,
            'activity_regularizer': regularizers.serialize(self.sub_activity_regularizer),
            '_high_activation': self.high_activation
        }
        base_config = super(_Inception, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
class Inception1D(_Inception):
    """1D inception layer.
    `Inception1D` implements the operation:
        `output = concat (i=0~D) ConvBranch1D(D, input)`
    where `ConvBranch1D` means D-1 times 1D convolutional layers.
    To be specific, when D=0, this branch is low-pass filtered by pooling layer.
    In some cases, the first term may not need to be convoluted.
    The implementation here is not exactly the same as original paper. The main 
    difference includes
        1. The main structure is only borrowed from inception-A block.
        2. The down sampling layer is also implemented in this class 
           (if set strides)
        3. We do not invoke the low-rank decomposition for the conv. kernels.
        4. We borrow the idea of residual-v2 block and change the order of some
           layers. 
    Arguments for inception block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        kernel_constraint: Constraint function applied to the kernel matrix.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        3D tensor with shape: `(batch_size, steps, input_dim)`
    Output shape:
        3D tensor with shape: `(batch_size, new_steps, filters)`
        `steps` value might have changed due to padding or strides.
    """

    def __init__(self,
               ofilters,
               kernel_size,
               lfilters=None,
               depth=3,
               strides=1,
               data_format='channels_last',
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
               dropout=None,
               dropout_rate=0.3,
               activation=None,
               activity_config=None,
               activity_regularizer=None,
               **kwargs):
        super(Inception1D, self).__init__(
            rank=1, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        
class Inception2D(_Inception):
    """2D inception layer (e.g. spatial convolution over images).
    `Inception2D` implements the operation:
        `output = concat (i=0~D) ConvBranch2D(D, input)`
    where `ConvBranch2D` means D-1 times 2D convolutional layers.
    To be specific, when D=0, this branch is low-pass filtered by pooling layer.
    In some cases, the first term may not need to be convoluted.
    The implementation here is not exactly the same as original paper. The main 
    difference includes
        1. The main structure is only borrowed from inception-A block.
        2. The down sampling layer is also implemented in this class 
           (if set strides)
        3. We do not invoke the low-rank decomposition for the conv. kernels.
        4. We borrow the idea of residual-v2 block and change the order of some
           layers. 
    Arguments for inception block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the height and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        kernel_constraint: Constraint function applied to the kernel matrix.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
               ofilters,
               kernel_size,
               lfilters=None,
               depth=3,
               strides=(1, 1),
               data_format='channels_last',
               dilation_rate=(1, 1),
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
               dropout=None,
               dropout_rate=0.3,
               activation=None,
               activity_config=None,
               activity_regularizer=None,
               **kwargs):
        super(Inception2D, self).__init__(
            rank=2, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        
class Inception3D(_Inception):
    """3D inception layer (e.g. spatial convolution over volumes).
    `Inception3D` implements the operation:
        `output = concat (i=0~D) ConvBranch3D(D, input)`
    where `ConvBranch3D` means D-1 times 3D convolutional layers.
    To be specific, when D=0, this branch is low-pass filtered by pooling layer.
    In some cases, the first term may not need to be convoluted.
    The implementation here is not exactly the same as original paper. The main 
    difference includes
        1. The main structure is only borrowed from inception-A block.
        2. The down sampling layer is also implemented in this class 
           (if set strides)
        3. We do not invoke the low-rank decomposition for the conv. kernels.
        4. We borrow the idea of residual-v2 block and change the order of some
           layers. 
    Arguments for inception block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            depth, height and width of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 3 integers,
            specifying the strides of the convolution along each spatial
            dimension.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `channels_first` corresponds to inputs with shape
            `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 3 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        kernel_constraint: Constraint function applied to the kernel matrix.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        5D tensor with shape:
        `(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if
        data_format='channels_first'
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if
        data_format='channels_last'.
    Output shape:
        5D tensor with shape:
        `(samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if
        data_format='channels_first'
        or 5D tensor with shape:
        `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` if
        data_format='channels_last'.
        `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have
        changed due to padding.
    """

    def __init__(self,
               ofilters,
               kernel_size,
               lfilters=None,
               depth=3,
               strides=(1, 1, 1),
               data_format='channels_last',
               dilation_rate=(1, 1, 1),
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
               dropout=None,
               dropout_rate=0.3,
               activation=None,
               activity_config=None,
               activity_regularizer=None,
               **kwargs):
        super(Inception3D, self).__init__(
            rank=3, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
            
class _InceptionTranspose(Layer):
    """Modern transposed inception layer (sometimes called inception deconvolution).
    Abstract nD inception transposed layer (private, used as implementation base).
    `_InceptionTranspose` implements the operation:
        `output = concat (i=0~D) ConvBranch(D, Upsamp(input))`
    where `ConvBranch` means D-1 times transposed convolutional layers.
    The main difference is, the input is up sampled.
    To be specific, when D=0, this branch is low-pass filtered by pooling layer.
    In some cases, the first term may not need to be convoluted.
    Arguments for inception block:
        rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of n integers, specifying the
            length of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
                 depth, ofilters,
                 kernel_size,
                 lfilters=None,
                 strides=1,
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
                 dropout=None,
                 dropout_rate=0.3,
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 _high_activation=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
          kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(_InceptionTranspose, self).__init__(trainable=trainable, name=name, **kwargs)
        # Inherit from keras.layers._Conv
        self.rank = rank
        self.depth = depth
        if depth < 1:
            raise ValueError('The depth of the inception block should be >= 1.')
        if ofilters % (depth+1) != 0:
            raise ValueError('The output filter number should be the multiple of (depth+1), i.e. N * {0}'.format(depth+1))
        self.ofilters = ofilters // (self.depth + 1)
        self.lfilters = lfilters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.output_padding = output_padding
        self.output_mshape = None
        self.output_cropping = None
        if output_mshape:
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
            self.gamma_initializer = initializers.get(gamma_initializer)
            self.gamma_regularizer = regularizers.get(gamma_regularizer)
            self.gamma_constraint = constraints.get(gamma_constraint)
        else:
            self.gamma_initializer = None
            self.gamma_regularizer = None
            self.gamma_constraint = None
        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.groups = groups
        # Inherit from mdnt.layers.dropout
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        # Inherit from keras.engine.Layer
        if _high_activation is not None:
            activation = _high_activation
        self.high_activation = _high_activation
        if isinstance(activation, str) and (activation.casefold() in ('prelu','lrelu')):
            self.activation = activations.get(None)
            self.high_activation = activation.casefold()
            self.activity_config = activity_config # dictionary passed to activation
        elif activation is not None:
            self.activation = activations.get(activation)
            self.activity_config = None
        self.sub_activity_regularizer=regularizers.get(activity_regularizer)

        # Reserve for build()
        self.channelIn = None
        
        self.trainable = trainable
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(self.rank + 2)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        self.channelIn = int(input_shape[channel_axis])
        if self.lfilters is None:
            self.lfilters = max( 1, self.channelIn // 2 )
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
                get_shape_diff = l_output_mshape[i] - l_input_shape[i]*max(self.strides[i], self.dilation_rate[i])
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
            self.layer_uppool.build(input_shape)
            next_shape = self.layer_uppool.compute_output_shape(input_shape)
            if self.output_padding is not None:
                self.layer_padding = ZeroPadding1D(padding=self.output_padding)[0] # Necessary for 1D case, because we need to pick (a,b) from ((a, b))
                self.layer_padding.build(next_shape)
                next_shape = self.layer_padding.compute_output_shape(next_shape)
            else:
                self.layer_padding = None
        elif self.rank == 2:
            self.layer_uppool = UpSampling2D(size=self.strides, data_format=self.data_format)
            self.layer_uppool.build(input_shape)
            next_shape = self.layer_uppool.compute_output_shape(input_shape)
            if self.output_padding is not None:
                self.layer_padding = ZeroPadding2D(padding=self.output_padding, data_format=self.data_format)
                self.layer_padding.build(next_shape)
                next_shape = self.layer_padding.compute_output_shape(next_shape)
            else:
                self.layer_padding = None
        elif self.rank == 3:
            self.layer_uppool = UpSampling3D(size=self.strides, data_format=self.data_format)
            self.layer_uppool.build(input_shape)
            next_shape = self.layer_uppool.compute_output_shape(input_shape)
            if self.output_padding is not None:
                self.layer_padding = ZeroPadding3D(padding=self.output_padding, data_format=self.data_format)
                self.layer_padding.build(next_shape)
                next_shape = self.layer_padding.compute_output_shape(next_shape)
            else:
                self.layer_padding = None
        else:
            raise ValueError('Rank of the deconvolution should be 1, 2 or 3.')
        # Consider the branch zero
        if not _check_dl_func(self.strides):
            if self.rank == 1:
                self.layer_branch_zero = MaxPooling1D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            elif self.rank == 2:
                self.layer_branch_zero = MaxPooling2D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            elif self.rank == 3:
                self.layer_branch_zero = MaxPooling3D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            else:
                raise ValueError('Rank of the inception should be 1, 2 or 3.')
        else:
            if self.rank == 1:
                self.layer_branch_zero = AveragePooling1D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            elif self.rank == 2:
                self.layer_branch_zero = AveragePooling2D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            elif self.rank == 3:
                self.layer_branch_zero = AveragePooling3D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            else:
                raise ValueError('Rank of the inception should be 1, 2 or 3.')
        self.layer_branch_zero.build(next_shape)
        zero_shape = self.layer_branch_zero.compute_output_shape(next_shape)
        # If channel does not match, use linear conv.
        if self.channelIn != self.ofilters:
            self.layer_branch_zero_map = _AConv(rank = self.rank,
                        filters = self.ofilters,
                        kernel_size = 1,
                        strides = 1,
                        padding = 'same',
                        data_format = self.data_format,
                        dilation_rate = 1,
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        normalization=self.normalization,
                        beta_initializer=self.beta_initializer,
                        gamma_initializer=self.gamma_initializer,
                        beta_regularizer=self.beta_regularizer,
                        gamma_regularizer=self.gamma_regularizer,
                        beta_constraint=self.beta_constraint,
                        gamma_constraint=self.gamma_constraint,
                        groups=self.groups,
                        activation=None,
                        activity_config=None,
                        activity_regularizer=None,
                        _high_activation=None,
                        trainable=self.trainable)
            self.layer_branch_zero_map.build(zero_shape)
            compat.collect_properties(self, self.layer_branch_zero_map) # for compatibility
            zero_shape = self.layer_branch_zero_map.compute_output_shape(zero_shape)
        else:
            self.layer_branch_zero_map = None
        # Consider the branch one
        if (not _check_dl_func(self.strides)) or self.channelIn != self.ofilters:
            self.layer_branch_one = _AConv(rank = self.rank,
                        filters = self.ofilters,
                        kernel_size = 1,
                        strides = 1,
                        padding = 'same',
                        data_format = self.data_format,
                        dilation_rate = 1,
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        normalization=self.normalization,
                        beta_initializer=self.beta_initializer,
                        gamma_initializer=self.gamma_initializer,
                        beta_regularizer=self.beta_regularizer,
                        gamma_regularizer=self.gamma_regularizer,
                        beta_constraint=self.beta_constraint,
                        gamma_constraint=self.gamma_constraint,
                        groups=self.groups,
                        activation=None,
                        activity_config=None,
                        activity_regularizer=None,
                        _high_activation=None,
                        trainable=self.trainable)
            self.layer_branch_one.build(next_shape)
            compat.collect_properties(self, self.layer_branch_one) # for compatibility
            one_shape = self.layer_branch_one.compute_output_shape(next_shape)
        else:
            self.layer_branch_one = None
            one_shape = next_shape
        # Consider branches with depth, with dropout
        self.layer_dropout = return_dropout(self.dropout, self.dropout_rate, axis=channel_axis, rank=self.rank)
        if self.layer_dropout is not None:
            self.layer_dropout.build(next_shape)
            depth_shape = self.layer_dropout.compute_output_shape(next_shape)
        else:
            depth_shape = next_shape
        depth_shape_list = []
        for D in range(self.depth-1):
            layer_middle_first = NACUnit(rank = self.rank,
                          filters = self.lfilters,
                          kernel_size = 1,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
            layer_middle_first.build(depth_shape)
            compat.collect_properties(self, layer_middle_first) # for compatibility
            branch_shape = layer_middle_first.compute_output_shape(depth_shape)
            setattr(self, 'layer_middle_D{0:02d}_00'.format(D+2), layer_middle_first)
            for i in range(D):
                layer_middle = NACUnit(rank = self.rank,
                          filters = self.lfilters,
                          kernel_size = self.kernel_size,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = self.dilation_rate,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
                layer_middle.build(branch_shape)
                compat.collect_properties(self, layer_middle) # for compatibility
                branch_shape = layer_middle.compute_output_shape(branch_shape)
                setattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, i+1), layer_middle)
            layer_middle_last = NACUnit(rank = self.rank,
                          filters = self.ofilters,
                          kernel_size = self.kernel_size,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = self.dilation_rate,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
            layer_middle_last.build(branch_shape)
            compat.collect_properties(self, layer_middle_last) # for compatibility
            branch_shape = layer_middle_last.compute_output_shape(branch_shape)
            setattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, D+1), layer_middle_last)
            depth_shape_list.append(branch_shape)
        if self.data_format == 'channels_first':
            self.layer_merge = Concatenate(axis=1)
        else:
            self.layer_merge = Concatenate()
        self.layer_merge.build([zero_shape, one_shape, *depth_shape_list])
        next_shape = self.layer_merge.compute_output_shape([zero_shape, one_shape, *depth_shape_list])
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
        super(_InceptionTranspose, self).build(next_shape)

    def call(self, inputs):
        outputs = self.layer_uppool(inputs)
        if self.layer_padding is not None:
            outputs = self.layer_padding(outputs)
        branch_zero = self.layer_branch_zero(outputs)
        if self.layer_branch_zero_map is not None:
            branch_zero = self.layer_branch_zero_map(branch_zero)
        if self.layer_branch_one is not None:
            branch_one = self.layer_branch_one(outputs)
        else:
            branch_one = outputs
        if self.layer_dropout is not None:
            depth_input = self.layer_dropout(outputs)
        else:
            depth_input = outputs
        branch_middle_list = []
        for D in range(self.depth-1):
            layer_middle_first = getattr(self, 'layer_middle_D{0:02d}_00'.format(D+2))
            branch_middle = layer_middle_first(depth_input)
            for i in range(D):
                layer_middle = getattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, i+1))
                branch_middle = layer_middle(branch_middle)
            layer_middle_last = getattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, D+1))
            branch_middle = layer_middle_last(branch_middle)
            branch_middle_list.append(branch_middle)
        outputs = self.layer_merge([branch_zero, branch_one, *branch_middle_list])
        if self.layer_cropping is not None:
            outputs = self.layer_cropping(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(self.rank + 2)
        next_shape = self.layer_uppool.compute_output_shape(input_shape)
        if self.layer_padding is not None:
            next_shape = self.layer_padding.compute_output_shape(next_shape)
        branch_zero_shape = self.layer_branch_zero.compute_output_shape(next_shape)
        if self.layer_branch_zero_map is not None:
            branch_zero_shape = self.layer_branch_zero_map.compute_output_shape(branch_zero_shape)
        if self.layer_branch_one is not None:
            branch_one_shape = self.layer_branch_one.compute_output_shape(next_shape)
        else:
            branch_one_shape = next_shape
        if self.layer_dropout is not None:
            depth_input_shape = self.layer_dropout.compute_output_shape(next_shape)
        else:
            depth_input_shape = next_shape
        branch_middle_shape_list = []
        for D in range(self.depth-1):
            layer_middle_first = getattr(self, 'layer_middle_D{0:02d}_00'.format(D+2))
            branch_middle_shape = layer_middle_first.compute_output_shape(depth_input_shape)
            for i in range(D):
                layer_middle = getattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, i+1))
                branch_middle_shape = layer_middle.compute_output_shape(branch_middle_shape)
            layer_middle_last = getattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, D+1))
            branch_middle_shape = layer_middle_last.compute_output_shape(branch_middle_shape)
            branch_middle_shape_list.append(branch_middle_shape)
        next_shape = self.layer_merge.compute_output_shape([branch_zero_shape, branch_one_shape, *branch_middle_shape_list])
        if self.layer_cropping is not None:
            next_shape = self.layer_cropping.compute_output_shape(next_shape)
        return next_shape
    
    def get_config(self):
        config = {
            'depth': self.depth,
            'ofilters': self.ofilters * (self.depth + 1),
            'lfilters': self.lfilters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
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
            'dropout': self.dropout,
            'dropout_rate': self.dropout_rate,
            'activation': activations.serialize(self.activation),
            'activity_config': self.activity_config,
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            '_high_activation': self.high_activation
        }
        base_config = super(_InceptionTranspose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
class Inception1DTranspose(_InceptionTranspose):
    """Modern transposed inception layer (sometimes called inception deconvolution).
    `Inception1DTranspose` implements the operation:
        `output = concat (i=0~D) Conv1DBranch(D, Upsamp(input))`
    where `Conv1DBranch` means D-1 times 1D convolutional layers.
    The main difference is, the input is up sampled.
    To be specific, when D=0, this branch is low-pass filtered by pooling layer.
    In some cases, the first term may not need to be convoluted.
    The upsampling is performed on the input layer. Previous works prove that the
    "transposed convolution" could be viewed as upsampling + plain convolution. Here
    we adopt such a technique to realize this upsampling architecture.
    Arguments for inception block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of n integers, specifying the
            length of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        output_mshape: (Only avaliable for new-style API) An integer or tuple/list
            of the desired output shape. When setting this option, `output_padding`
            and `out_cropping` would be inferred from the input shape, which means
            users' options would be invalid for the following two options.
            A recommended method of using this method is applying such a scheme:
                `AConv(..., output_mshape=tensor.get_shape())`
        output_padding: An integer or tuple/list of n integers,
            specifying the amount of padding along the height and width
            of the output tensor.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape would not be padded.
        out_cropping: (Only avaliable for new-style API) An integer or tuple/list 
            of n integers, specifying the amount of cropping along the axes of the
            output tensor. The amount of output cropping along a given dimension must
            be lower than the stride along that same dimension.
            If set to `None` (default), the output shape would not be cropped.
        data_format: A string, only support `channels_last` here:
            `channels_last` corresponds to inputs with shape
            `(batch, steps channels)`
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        3D tensor with shape: `(batch_size, steps, input_dim)`
    Output shape:
        3D tensor with shape: `(batch_size, new_steps, filters)`
        `steps` value might have changed due to padding or strides.
    """

    def __init__(self, ofilters,
                 kernel_size,
                 lfilters=None,
                 depth=3,
                 strides=1,
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
                 dropout=None,
                 dropout_rate=0.3,
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 **kwargs):
        super(Inception1DTranspose, self).__init__(
            rank=1, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            output_mshape=output_mshape,
            output_padding=output_padding,
            output_cropping=output_cropping,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
            
class Inception2DTranspose(_InceptionTranspose):
    """Modern transposed inception layer (sometimes called inception deconvolution).
    `Inception2DTranspose` implements the operation:
        `output = concat (i=0~D) Conv2DBranch(D, Upsamp(input))`
    where `Conv2DBranch` means D-1 times 2D convolutional layers.
    The main difference is, the input is up sampled.
    To be specific, when D=0, this branch is low-pass filtered by pooling layer.
    In some cases, the first term may not need to be convoluted.
    The upsampling is performed on the input layer. Previous works prove that the
    "transposed convolution" could be viewed as upsampling + plain convolution. Here
    we adopt such a technique to realize this upsampling architecture.
    Arguments for inception block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the height and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        output_mshape: (Only avaliable for new-style API) An integer or tuple/list
            of the desired output shape. When setting this option, `output_padding`
            and `out_cropping` would be inferred from the input shape, which means
            users' options would be invalid for the following two options.
            A recommended method of using this method is applying such a scheme:
                `AConv(..., output_mshape=tensor.get_shape())`
        output_padding: An integer or tuple/list of 2 integers,
            specifying the amount of padding along the height and width
            of the output tensor.
            Can be a single integer to specify the same value for all
            spatial dimensions.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape would not be padded.
        out_cropping: (Only avaliable for new-style API) An integer or tuple/list 
            of n integers, specifying the amount of cropping along the axes of the
            output tensor. The amount of output cropping along a given dimension must
            be lower than the stride along that same dimension.
            If set to `None` (default), the output shape would not be cropped.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self, ofilters,
                 kernel_size,
                 lfilters=None,
                 depth=3,
                 strides=(1, 1),
                 output_mshape=None,
                 output_padding=None,
                 output_cropping=None,
                 data_format=None,
                 dilation_rate=(1, 1),
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
                 dropout=None,
                 dropout_rate=0.3,
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 **kwargs):
        super(Inception2DTranspose, self).__init__(
            rank=2, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            output_mshape=output_mshape,
            output_padding=output_padding,
            output_cropping=output_cropping,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
            
class Inception3DTranspose(_InceptionTranspose):
    """Modern transposed inception layer (sometimes called inception deconvolution).
    `Inception3DTranspose` implements the operation:
        `output = concat (i=0~D) Conv3DBranch(D, Upsamp(input))`
    where `Conv3DBranch` means D-1 times 3D convolutional layers.
    The main difference is, the input is up sampled.
    To be specific, when D=0, this branch is low-pass filtered by pooling layer.
    In some cases, the first term may not need to be convoluted.
    The upsampling is performed on the input layer. Previous works prove that the
    "transposed convolution" could be viewed as upsampling + plain convolution. Here
    we adopt such a technique to realize this upsampling architecture.
    Arguments for inception block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            depth, height and width of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 3 integers,
            specifying the strides of the convolution along the depth, height
            and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        output_mshape: (Only avaliable for new-style API) An integer or tuple/list
            of the desired output shape. When setting this option, `output_padding`
            and `out_cropping` would be inferred from the input shape, which means
            users' options would be invalid for the following two options.
            A recommended method of using this method is applying such a scheme:
                `AConv(..., output_mshape=tensor.get_shape())`
        output_padding: An integer or tuple/list of 3 integers,
            specifying the amount of padding along the depth, height, and
            width.
            Can be a single integer to specify the same value for all
            spatial dimensions.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape is inferred.
        out_cropping: (Only avaliable for new-style API) An integer or tuple/list 
            of n integers, specifying the amount of cropping along the axes of the
            output tensor. The amount of output cropping along a given dimension must
            be lower than the stride along that same dimension.
            If set to `None` (default), the output shape would not be cropped.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, depth, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, depth, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 3 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        5D tensor with shape:
        `(batch, channels, depth, rows, cols)` if data_format='channels_first'
        or 5D tensor with shape:
        `(batch, depth, rows, cols, channels)` if data_format='channels_last'.
    Output shape:
        5D tensor with shape:
        `(batch, filters, new_depth, new_rows, new_cols)` if
        data_format='channels_first'
        or 5D tensor with shape:
        `(batch, new_depth, new_rows, new_cols, filters)` if
        data_format='channels_last'.
        `depth` and `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self, ofilters,
                 kernel_size,
                 lfilters=None,
                 depth=3,
                 strides=(1, 1, 1),
                 output_mshape=None,
                 output_padding=None,
                 output_cropping=None,
                 data_format=None,
                 dilation_rate=(1, 1, 1),
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
                 dropout=None,
                 dropout_rate=0.3,
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 **kwargs):
        super(Inception3DTranspose, self).__init__(
            rank=3, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            output_mshape=output_mshape,
            output_padding=output_padding,
            output_cropping=output_cropping,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)

class _Inceptres(Layer):
    """Modern inception residual layer.
    Abstract nD inception-residual layer (private, used as implementation base).
    `_Inceptres` implements the operation:
        `output = Conv(input) + Conv( concat (i=0~D) ConvBranch(D, input) )`
    Certainly, this structure could be viewed as a combination of residual block
    and inception block. The first linear convolution is not necessary to exist,
    because it is used for channel mapping. The second convolution is required
    in most cases, it maps the boosted channels of inception block into the ori-
    ginal channel space.
    The inception-residual structure is adapted from:
        Inception-v4, Inception-ResNet and the Impact of Residual 
        Connections on Learning
            https://arxiv.org/abs/1602.07261
    The implementation here is not exactly the same as original paper. The main 
    difference includes
        1. We do not take off the D0 branch from the inception part.
        2. The down sampling layer is also implemented in this class 
           (if set strides), the down sampling is performed by conv. with strides
           rather than max pooling layer.
        3. We borrow the idea of residual-v2 block and change the order of some
           layers. 
    Arguments for inception residual block:
        rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of n integers, specifying the
            length of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
                 depth, ofilters,
                 kernel_size,
                 lfilters=None,
                 strides=1,
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
                 dropout=None,
                 dropout_rate=0.3,
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 _high_activation=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
          kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(_Inceptres, self).__init__(trainable=trainable, name=name, **kwargs)
        # Inherit from keras.layers._Conv
        self.rank = rank
        self.depth = depth
        if depth < 1:
            raise ValueError('The depth of the inception block should be >= 1.')
        self.ofilters = ofilters
        self.lfilters = lfilters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')
        if (not _check_dl_func(self.dilation_rate)) and (not _check_dl_func(self.strides)):
            raise ValueError('Does not support dilation_rate when strides > 1.')
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        # Inherit from mdnt.layers.normalize
        self.normalization = normalization
        if isinstance(normalization, str) and normalization in ('batch', 'inst', 'group'):
            self.gamma_initializer = initializers.get(gamma_initializer)
            self.gamma_regularizer = regularizers.get(gamma_regularizer)
            self.gamma_constraint = constraints.get(gamma_constraint)
        else:
            self.gamma_initializer = None
            self.gamma_regularizer = None
            self.gamma_constraint = None
        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.groups = groups
        # Inherit from mdnt.layers.dropout
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        # Inherit from keras.engine.Layer
        if _high_activation is not None:
            activation = _high_activation
        self.high_activation = _high_activation
        if isinstance(activation, str) and (activation.casefold() in ('prelu','lrelu')):
            self.activation = activations.get(None)
            self.high_activation = activation.casefold()
            self.activity_config = activity_config # dictionary passed to activation
        elif activation is not None:
            self.activation = activations.get(activation)
            self.activity_config = None
        self.sub_activity_regularizer=regularizers.get(activity_regularizer)

        # Reserve for build()
        self.channelIn = None
        
        self.trainable = trainable
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(self.rank + 2)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        self.channelIn = int(input_shape[channel_axis])
        if self.lfilters is None:
            self.lfilters = max( 1, self.channelIn // 2 )
        # Here we define the left branch
        last_use_bias = True
        if _check_dl_func(self.strides) and self.ofilters == self.channelIn:
            self.layer_branch_left = None
            left_shape = input_shape
        else:
            last_use_bias = False
            self.layer_branch_left = _AConv(rank = self.rank,
                          filters = self.ofilters,
                          kernel_size = 1,
                          strides = self.strides,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=None,
                          activity_config=None,
                          activity_regularizer=None,
                          _high_activation=None,
                          trainable=self.trainable)
            self.layer_branch_left.build(input_shape)
            compat.collect_properties(self, self.layer_branch_left) # for compatibility
            left_shape = self.layer_branch_left.compute_output_shape(input_shape)
        # Here we define the right branch
        # Consider the branch zero
        if not _check_dl_func(self.strides):
            if self.rank == 1:
                self.layer_branch_zero = MaxPooling1D(pool_size=self.kernel_size, strides=self.strides, padding='same', data_format=self.data_format)
            elif self.rank == 2:
                self.layer_branch_zero = MaxPooling2D(pool_size=self.kernel_size, strides=self.strides, padding='same', data_format=self.data_format)
            elif self.rank == 3:
                self.layer_branch_zero = MaxPooling3D(pool_size=self.kernel_size, strides=self.strides, padding='same', data_format=self.data_format)
            else:
                raise ValueError('Rank of the inception should be 1, 2 or 3.')
        else:
            if self.rank == 1:
                self.layer_branch_zero = AveragePooling1D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            elif self.rank == 2:
                self.layer_branch_zero = AveragePooling2D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            elif self.rank == 3:
                self.layer_branch_zero = AveragePooling3D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            else:
                raise ValueError('Rank of the inception should be 1, 2 or 3.')
        self.layer_branch_zero.build(input_shape)
        zero_shape = self.layer_branch_zero.compute_output_shape(input_shape)
        # Depth 0.
        if self.channelIn != self.lfilters:
            self.layer_branch_zero_map = NACUnit(rank = self.rank,
                          filters = self.lfilters,
                          kernel_size = 1,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
            self.layer_branch_zero_map.build(zero_shape)
            compat.collect_properties(self, self.layer_branch_zero_map) # for compatibility
            zero_shape = self.layer_branch_zero_map.compute_output_shape(zero_shape)
        else:
            self.layer_branch_zero_map = None
        # Consider the branch one
        if (not _check_dl_func(self.strides)) or self.channelIn != self.lfilters:
            self.layer_branch_one = NACUnit(rank = self.rank,
                          filters = self.lfilters,
                          kernel_size = 1,
                          strides = self.strides,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
            self.layer_branch_one.build(input_shape)
            compat.collect_properties(self, self.layer_branch_one) # for compatibility
            one_shape = self.layer_branch_one.compute_output_shape(input_shape)
        else:
            self.layer_branch_one = None
            one_shape = input_shape
        # Consider branches with depth, with dropout
        self.layer_dropout = return_dropout(self.dropout, self.dropout_rate, axis=channel_axis, rank=self.rank)
        if self.layer_dropout is not None:
            self.layer_dropout.build(input_shape)
            depth_shape = self.layer_dropout.compute_output_shape(input_shape)
        else:
            depth_shape = input_shape
        depth_shape_list = []
        for D in range(self.depth-1):
            layer_middle_first = NACUnit(rank = self.rank,
                          filters = self.lfilters,
                          kernel_size = 1,
                          strides = self.strides,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
            layer_middle_first.build(depth_shape)
            compat.collect_properties(self, layer_middle_first) # for compatibility
            branch_shape = layer_middle_first.compute_output_shape(depth_shape)
            setattr(self, 'layer_middle_D{0:02d}_00'.format(D+2), layer_middle_first)
            for i in range(D+1):
                layer_middle = NACUnit(rank = self.rank,
                          filters = self.lfilters,
                          kernel_size = self.kernel_size,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = self.dilation_rate,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
                layer_middle.build(branch_shape)
                compat.collect_properties(self, layer_middle) # for compatibility
                branch_shape = layer_middle.compute_output_shape(branch_shape)
                setattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, i+1), layer_middle)
            depth_shape_list.append(branch_shape)
        # Merge the right branch by concatnation.
        if self.data_format == 'channels_first':
            self.layer_branch_right = Concatenate(axis=1)
        else:
            self.layer_branch_right = Concatenate()
        self.layer_branch_right.build([zero_shape, one_shape, *depth_shape_list])
        right_shape = self.layer_branch_right.compute_output_shape([zero_shape, one_shape, *depth_shape_list])
        self.layer_branch_right_map = NACUnit(rank = self.rank,
                          filters = self.ofilters,
                          kernel_size = 1,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          _use_bias=last_use_bias,
                          trainable=self.trainable)
        self.layer_branch_right_map.build(right_shape)
        compat.collect_properties(self, self.layer_branch_right_map) # for compatibility
        right_shape = self.layer_branch_right_map.compute_output_shape(right_shape)
        # Merge the residual block
        self.layer_merge = Add()
        self.layer_merge.build([left_shape, right_shape])
        super(_Inceptres, self).build(input_shape)

    def call(self, inputs):
        # Left branch
        if self.layer_branch_left is not None:
            branch_left = self.layer_branch_left(inputs)
        else:
            branch_left = inputs
        # Right branch
        branch_zero = self.layer_branch_zero(inputs)
        if self.layer_branch_zero_map is not None:
            branch_zero = self.layer_branch_zero_map(branch_zero)
        if self.layer_branch_one is not None:
            branch_one = self.layer_branch_one(inputs)
        else:
            branch_one = inputs
        if self.layer_dropout is not None:
            depth_input = self.layer_dropout(inputs)
        else:
            depth_input = inputs
        branch_middle_list = []
        for D in range(self.depth-1):
            layer_middle_first = getattr(self, 'layer_middle_D{0:02d}_00'.format(D+2))
            branch_middle = layer_middle_first(depth_input)
            for i in range(D+1):
                layer_middle = getattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, i+1))
                branch_middle = layer_middle(branch_middle)
            branch_middle_list.append(branch_middle)
        branch_right = self.layer_branch_right([branch_zero, branch_one, *branch_middle_list])
        branch_right = self.layer_branch_right_map(branch_right)
        outputs = self.layer_merge([branch_left, branch_right])
        return outputs

    def compute_output_shape(self, input_shape):
        if self.layer_branch_left is not None:
            branch_left_shape = self.layer_branch_left.compute_output_shape(input_shape)
        else:
            branch_left_shape = input_shape
        branch_zero_shape = self.layer_branch_zero.compute_output_shape(input_shape)
        if self.layer_branch_zero_map is not None:
            branch_zero_shape = self.layer_branch_zero_map.compute_output_shape(branch_zero_shape)
        if self.layer_branch_one is not None:
            branch_one_shape = self.layer_branch_one.compute_output_shape(input_shape)
        else:
            branch_one_shape = input_shape
        if self.layer_dropout is not None:
            depth_input_shape = self.layer_dropout.compute_output_shape(input_shape)
        else:
            depth_input_shape = input_shape
        branch_middle_shape_list = []
        for D in range(self.depth-1):
            layer_middle_first = getattr(self, 'layer_middle_D{0:02d}_00'.format(D+2))
            branch_middle_shape = layer_middle_first.compute_output_shape(depth_input_shape)
            for i in range(D+1):
                layer_middle = getattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, i+1))
                branch_middle_shape = layer_middle.compute_output_shape(branch_middle_shape)
            branch_middle_shape_list.append(branch_middle_shape)
        branch_right_shape = self.layer_branch_right.compute_output_shape([branch_zero_shape, branch_one_shape, *branch_middle_shape_list])
        branch_right_shape = self.layer_branch_right_map.compute_output_shape(branch_right_shape)
        next_shape = self.layer_merge.compute_output_shape([branch_left_shape, branch_right_shape])
        return next_shape
    
    def get_config(self):
        config = {
            'depth': self.depth,
            'ofilters': self.ofilters,
            'lfilters': self.lfilters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
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
            'dropout': self.dropout,
            'dropout_rate': self.dropout_rate,
            'activation': activations.serialize(self.activation),
            'activity_config': self.activity_config,
            'activity_regularizer': regularizers.serialize(self.sub_activity_regularizer),
            '_high_activation': self.high_activation
        }
        base_config = super(_Inceptres, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
class Inceptres1D(_Inceptres):
    """1D inception residual layer.
    `Inceptres1D` implements the operation:
        `output = Conv1D(input) + Conv1D( concat (i=0~D) Conv1DBranch(D, input) )`
    Certainly, this structure could be viewed as a combination of residual block
    and inception block. The first linear convolution is not necessary to exist,
    because it is used for channel mapping. The second convolution is required
    in most cases, it maps the boosted channels of inception block into the ori-
    ginal channel space.
    The implementation here is not exactly the same as original paper. The main 
    difference includes
        1. We do not take off the D0 branch from the inception part.
        2. The down sampling layer is also implemented in this class 
           (if set strides), the down sampling is performed by conv. with strides
           rather than max pooling layer.
        3. We borrow the idea of residual-v2 block and change the order of some
           layers. 
    Arguments for inception residual block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        kernel_constraint: Constraint function applied to the kernel matrix.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        3D tensor with shape: `(batch_size, steps, input_dim)`
    Output shape:
        3D tensor with shape: `(batch_size, new_steps, filters)`
        `steps` value might have changed due to padding or strides.
    """

    def __init__(self,
               ofilters,
               kernel_size,
               lfilters=None,
               depth=3,
               strides=1,
               data_format='channels_last',
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
               dropout=None,
               dropout_rate=0.3,
               activation=None,
               activity_config=None,
               activity_regularizer=None,
               **kwargs):
        super(Inceptres1D, self).__init__(
            rank=1, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        
class Inceptres2D(_Inceptres):
    """2D inception residual layer (e.g. spatial convolution over images).
    `Inceptres2D` implements the operation:
        `output = Conv2D(input) + Conv2D( concat (i=0~D) Conv2DBranch(D, input) )`
    Certainly, this structure could be viewed as a combination of residual block
    and inception block. The first linear convolution is not necessary to exist,
    because it is used for channel mapping. The second convolution is required
    in most cases, it maps the boosted channels of inception block into the ori-
    ginal channel space.
    The implementation here is not exactly the same as original paper. The main 
    difference includes
        1. We do not take off the D0 branch from the inception part.
        2. The down sampling layer is also implemented in this class 
           (if set strides), the down sampling is performed by conv. with strides
           rather than max pooling layer.
        3. We borrow the idea of residual-v2 block and change the order of some
           layers. 
    Arguments for inception residual block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the height and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        kernel_constraint: Constraint function applied to the kernel matrix.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
               ofilters,
               kernel_size,
               lfilters=None,
               depth=3,
               strides=(1, 1),
               data_format='channels_last',
               dilation_rate=(1, 1),
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
               dropout=None,
               dropout_rate=0.3,
               activation=None,
               activity_config=None,
               activity_regularizer=None,
               **kwargs):
        super(Inceptres2D, self).__init__(
            rank=2, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        
class Inceptres3D(_Inceptres):
    """3D inception residual layer (e.g. spatial convolution over volumes).
    `Inceptres3D` implements the operation:
        `output = Conv3D(input) + Conv3D( concat (i=0~D) Conv3DBranch(D, input) )`
    Certainly, this structure could be viewed as a combination of residual block
    and inception block. The first linear convolution is not necessary to exist,
    because it is used for channel mapping. The second convolution is required
    in most cases, it maps the boosted channels of inception block into the ori-
    ginal channel space.
    The implementation here is not exactly the same as original paper. The main 
    difference includes
        1. We do not take off the D0 branch from the inception part.
        2. The down sampling layer is also implemented in this class 
           (if set strides), the down sampling is performed by conv. with strides
           rather than max pooling layer.
        3. We borrow the idea of residual-v2 block and change the order of some
           layers. 
    Arguments for inception residual block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            depth, height and width of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 3 integers,
            specifying the strides of the convolution along each spatial
            dimension.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `channels_first` corresponds to inputs with shape
            `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 3 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        kernel_constraint: Constraint function applied to the kernel matrix.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        5D tensor with shape:
        `(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if
        data_format='channels_first'
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if
        data_format='channels_last'.
    Output shape:
        5D tensor with shape:
        `(samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if
        data_format='channels_first'
        or 5D tensor with shape:
        `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` if
        data_format='channels_last'.
        `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have
        changed due to padding.
    """

    def __init__(self,
               ofilters,
               kernel_size,
               lfilters=None,
               depth=3,
               strides=(1, 1, 1),
               data_format='channels_last',
               dilation_rate=(1, 1, 1),
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
               dropout=None,
               dropout_rate=0.3,
               activation=None,
               activity_config=None,
               activity_regularizer=None,
               **kwargs):
        super(Inceptres3D, self).__init__(
            rank=3, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
            
class _InceptresTranspose(Layer):
    """Modern transposed inception residual layer (sometimes called inceptres
                                                   deconvolution).
    Abstract nD transposed inception residual layer (private, used as implementation base).
    `_InceptresTranspose` implements the operation:
        `output = Conv(Upsamp(input)) + Conv( concat (i=0~D) ConvBranch(D, Upsamp(input)) )`
    where `ConvBranch` means D-1 times convolutional layers.
    The upsampling is performed on the input layer. Previous works prove that the
    "transposed convolution" could be viewed as upsampling + plain convolution. Here
    we adopt such a technique to realize this upsampling architecture.
    Arguments for inception residual block:
        rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of n integers, specifying the
            length of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
                 depth, ofilters,
                 kernel_size,
                 lfilters=None,
                 strides=1,
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
                 dropout=None,
                 dropout_rate=0.3,
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 _high_activation=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
          kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(_InceptresTranspose, self).__init__(trainable=trainable, name=name, **kwargs)
        # Inherit from keras.layers._Conv
        self.rank = rank
        self.depth = depth
        if depth < 1:
            raise ValueError('The depth of the inception block should be >= 1.')
        self.ofilters = ofilters
        self.lfilters = lfilters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.output_padding = output_padding
        self.output_mshape = None
        self.output_cropping = None
        if output_mshape:
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
            self.gamma_initializer = initializers.get(gamma_initializer)
            self.gamma_regularizer = regularizers.get(gamma_regularizer)
            self.gamma_constraint = constraints.get(gamma_constraint)
        else:
            self.gamma_initializer = None
            self.gamma_regularizer = None
            self.gamma_constraint = None
        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.groups = groups
        # Inherit from mdnt.layers.dropout
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        # Inherit from keras.engine.Layer
        if _high_activation is not None:
            activation = _high_activation
        self.high_activation = _high_activation
        if isinstance(activation, str) and (activation.casefold() in ('prelu','lrelu')):
            self.activation = activations.get(None)
            self.high_activation = activation.casefold()
            self.activity_config = activity_config # dictionary passed to activation
        elif activation is not None:
            self.activation = activations.get(activation)
            self.activity_config = None
        self.sub_activity_regularizer=regularizers.get(activity_regularizer)

        # Reserve for build()
        self.channelIn = None
        
        self.trainable = trainable
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(self.rank + 2)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        self.channelIn = int(input_shape[channel_axis])
        if self.lfilters is None:
            self.lfilters = max( 1, self.channelIn // 2 )
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
                get_shape_diff = l_output_mshape[i] - l_input_shape[i]*max(self.strides[i], self.dilation_rate[i])
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
            self.layer_uppool.build(input_shape)
            next_shape = self.layer_uppool.compute_output_shape(input_shape)
            if self.output_padding is not None:
                self.layer_padding = ZeroPadding1D(padding=self.output_padding)[0] # Necessary for 1D case, because we need to pick (a,b) from ((a, b))
                self.layer_padding.build(next_shape)
                next_shape = self.layer_padding.compute_output_shape(next_shape)
            else:
                self.layer_padding = None
        elif self.rank == 2:
            self.layer_uppool = UpSampling2D(size=self.strides, data_format=self.data_format)
            self.layer_uppool.build(input_shape)
            next_shape = self.layer_uppool.compute_output_shape(input_shape)
            if self.output_padding is not None:
                self.layer_padding = ZeroPadding2D(padding=self.output_padding, data_format=self.data_format)
                self.layer_padding.build(next_shape)
                next_shape = self.layer_padding.compute_output_shape(next_shape)
            else:
                self.layer_padding = None
        elif self.rank == 3:
            self.layer_uppool = UpSampling3D(size=self.strides, data_format=self.data_format)
            self.layer_uppool.build(input_shape)
            next_shape = self.layer_uppool.compute_output_shape(input_shape)
            if self.output_padding is not None:
                self.layer_padding = ZeroPadding3D(padding=self.output_padding, data_format=self.data_format)
                self.layer_padding.build(next_shape)
                next_shape = self.layer_padding.compute_output_shape(next_shape)
            else:
                self.layer_padding = None
        else:
            raise ValueError('Rank of the deconvolution should be 1, 2 or 3.')
        # Here we define the left branch
        last_use_bias = True
        if self.ofilters == self.channelIn:
            self.layer_branch_left = None
            left_shape = next_shape
        else:
            last_use_bias = False
            self.layer_branch_left = _AConv(rank = self.rank,
                          filters = self.ofilters,
                          kernel_size = 1,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=None,
                          activity_config=None,
                          activity_regularizer=None,
                          _high_activation=None,
                          trainable=self.trainable)
            self.layer_branch_left.build(next_shape)
            compat.collect_properties(self, self.layer_branch_left) # for compatibility
            left_shape = self.layer_branch_left.compute_output_shape(next_shape)
        # Here we define the right branch
        # Consider the branch zero
        if not _check_dl_func(self.strides):
            if self.rank == 1:
                self.layer_branch_zero = MaxPooling1D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            elif self.rank == 2:
                self.layer_branch_zero = MaxPooling2D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            elif self.rank == 3:
                self.layer_branch_zero = MaxPooling3D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            else:
                raise ValueError('Rank of the inception should be 1, 2 or 3.')
        else:
            if self.rank == 1:
                self.layer_branch_zero = AveragePooling1D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            elif self.rank == 2:
                self.layer_branch_zero = AveragePooling2D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            elif self.rank == 3:
                self.layer_branch_zero = AveragePooling3D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            else:
                raise ValueError('Rank of the inception should be 1, 2 or 3.')
        self.layer_branch_zero.build(next_shape)
        zero_shape = self.layer_branch_zero.compute_output_shape(next_shape)
        # Depth 0.
        if self.channelIn != self.lfilters:
            self.layer_branch_zero_map = NACUnit(rank = self.rank,
                          filters = self.lfilters,
                          kernel_size = 1,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
            self.layer_branch_zero_map.build(zero_shape)
            compat.collect_properties(self, self.layer_branch_zero_map) # for compatibility
            zero_shape = self.layer_branch_zero_map.compute_output_shape(zero_shape)
        else:
            self.layer_branch_zero_map = None
        # Consider the branch one
        if self.channelIn != self.lfilters:
            self.layer_branch_one = NACUnit(rank = self.rank,
                          filters = self.lfilters,
                          kernel_size = 1,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
            self.layer_branch_one.build(next_shape)
            compat.collect_properties(self, self.layer_branch_one) # for compatibility
            one_shape = self.layer_branch_one.compute_output_shape(next_shape)
        else:
            self.layer_branch_one = None
            one_shape = next_shape
        # Consider branches with depth, with dropout
        self.layer_dropout = return_dropout(self.dropout, self.dropout_rate, axis=channel_axis, rank=self.rank)
        if self.layer_dropout is not None:
            self.layer_dropout.build(next_shape)
            depth_shape = self.layer_dropout.compute_output_shape(next_shape)
        else:
            depth_shape = next_shape
        depth_shape_list = []
        for D in range(self.depth-1):
            layer_middle_first = NACUnit(rank = self.rank,
                          filters = self.lfilters,
                          kernel_size = 1,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
            layer_middle_first.build(depth_shape)
            compat.collect_properties(self, layer_middle_first) # for compatibility
            branch_shape = layer_middle_first.compute_output_shape(depth_shape)
            setattr(self, 'layer_middle_D{0:02d}_00'.format(D+2), layer_middle_first)
            for i in range(D+1):
                layer_middle = NACUnit(rank = self.rank,
                          filters = self.lfilters,
                          kernel_size = self.kernel_size,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = self.dilation_rate,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
                layer_middle.build(branch_shape)
                compat.collect_properties(self, layer_middle) # for compatibility
                branch_shape = layer_middle.compute_output_shape(branch_shape)
                setattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, i+1), layer_middle)
            depth_shape_list.append(branch_shape)
        # Merge the right branch by concatnation.
        if self.data_format == 'channels_first':
            self.layer_branch_right = Concatenate(axis=1)
        else:
            self.layer_branch_right = Concatenate()
        self.layer_branch_right.build([zero_shape, one_shape, *depth_shape_list])
        right_shape = self.layer_branch_right.compute_output_shape([zero_shape, one_shape, *depth_shape_list])
        self.layer_branch_right_map = NACUnit(rank = self.rank,
                          filters = self.ofilters,
                          kernel_size = 1,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          _use_bias=last_use_bias,
                          trainable=self.trainable)
        self.layer_branch_right_map.build(right_shape)
        compat.collect_properties(self, self.layer_branch_right_map) # for compatibility
        right_shape = self.layer_branch_right_map.compute_output_shape(right_shape)
        # Merge the residual block
        self.layer_merge = Add()
        self.layer_merge.build([left_shape, right_shape])
        next_shape = self.layer_merge.compute_output_shape([left_shape, right_shape])
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
        super(_InceptresTranspose, self).build(next_shape)

    def call(self, inputs):
        outputs = self.layer_uppool(inputs)
        if self.layer_padding is not None:
            outputs = self.layer_padding(outputs)
        # Left branch
        if self.layer_branch_left is not None:
            branch_left = self.layer_branch_left(outputs)
        else:
            branch_left = outputs
        # Right branch
        branch_zero = self.layer_branch_zero(outputs)
        if self.layer_branch_zero_map is not None:
            branch_zero = self.layer_branch_zero_map(branch_zero)
        if self.layer_branch_one is not None:
            branch_one = self.layer_branch_one(outputs)
        else:
            branch_one = outputs
        if self.layer_dropout is not None:
            depth_input = self.layer_dropout(outputs)
        else:
            depth_input = outputs
        branch_middle_list = []
        for D in range(self.depth-1):
            layer_middle_first = getattr(self, 'layer_middle_D{0:02d}_00'.format(D+2))
            branch_middle = layer_middle_first(depth_input)
            for i in range(D+1):
                layer_middle = getattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, i+1))
                branch_middle = layer_middle(branch_middle)
            branch_middle_list.append(branch_middle)
        branch_right = self.layer_branch_right([branch_zero, branch_one, *branch_middle_list])
        branch_right = self.layer_branch_right_map(branch_right)
        outputs = self.layer_merge([branch_left, branch_right])
        if self.layer_cropping is not None:
            outputs = self.layer_cropping(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(self.rank + 2)
        next_shape = self.layer_uppool.compute_output_shape(input_shape)
        if self.layer_padding is not None:
            next_shape = self.layer_padding.compute_output_shape(next_shape)
        if self.layer_branch_left is not None:
            branch_left_shape = self.layer_branch_left.compute_output_shape(next_shape)
        else:
            branch_left_shape = next_shape
        branch_zero_shape = self.layer_branch_zero.compute_output_shape(next_shape)
        if self.layer_branch_zero_map is not None:
            branch_zero_shape = self.layer_branch_zero_map.compute_output_shape(branch_zero_shape)
        if self.layer_branch_one is not None:
            branch_one_shape = self.layer_branch_one.compute_output_shape(next_shape)
        else:
            branch_one_shape = next_shape
        if self.layer_dropout is not None:
            depth_input_shape = self.layer_dropout.compute_output_shape(next_shape)
        else:
            depth_input_shape = next_shape
        branch_middle_shape_list = []
        for D in range(self.depth-1):
            layer_middle_first = getattr(self, 'layer_middle_D{0:02d}_00'.format(D+2))
            branch_middle_shape = layer_middle_first.compute_output_shape(depth_input_shape)
            for i in range(D+1):
                layer_middle = getattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+2, i+1))
                branch_middle_shape = layer_middle.compute_output_shape(branch_middle_shape)
            branch_middle_shape_list.append(branch_middle_shape)
        branch_right_shape = self.layer_branch_right.compute_output_shape([branch_zero_shape, branch_one_shape, *branch_middle_shape_list])
        branch_right_shape = self.layer_branch_right_map.compute_output_shape(branch_right_shape)
        next_shape = self.layer_merge.compute_output_shape([branch_left_shape, branch_right_shape])
        if self.layer_cropping is not None:
            next_shape = self.layer_cropping.compute_output_shape(next_shape)
        return next_shape
    
    def get_config(self):
        config = {
            'depth': self.depth,
            'ofilters': self.ofilters,
            'lfilters': self.lfilters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
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
            'dropout': self.dropout,
            'dropout_rate': self.dropout_rate,
            'activation': activations.serialize(self.activation),
            'activity_config': self.activity_config,
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            '_high_activation': self.high_activation
        }
        base_config = super(_InceptresTranspose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
class Inceptres1DTranspose(_InceptresTranspose):
    """Modern transposed inception residual layer (sometimes called inceptres
                                                   deconvolution).
    `Inceptres1DTranspose` implements the operation:
        `output = Conv1D(Upsamp(input)) + Conv1D( concat (i=0~D) Conv1DBranch(D, Upsamp(input)) )`
    where `Conv1DBranch` means D-1 times convolutional layers.
    The upsampling is performed on the input layer. Previous works prove that the
    "transposed convolution" could be viewed as upsampling + plain convolution. Here
    we adopt such a technique to realize this upsampling architecture.
    Arguments for inception residual block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of n integers, specifying the
            length of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        output_mshape: (Only avaliable for new-style API) An integer or tuple/list
            of the desired output shape. When setting this option, `output_padding`
            and `out_cropping` would be inferred from the input shape, which means
            users' options would be invalid for the following two options.
            A recommended method of using this method is applying such a scheme:
                `AConv(..., output_mshape=tensor.get_shape())`
        output_padding: An integer or tuple/list of n integers,
            specifying the amount of padding along the height and width
            of the output tensor.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape would not be padded.
        out_cropping: (Only avaliable for new-style API) An integer or tuple/list 
            of n integers, specifying the amount of cropping along the axes of the
            output tensor. The amount of output cropping along a given dimension must
            be lower than the stride along that same dimension.
            If set to `None` (default), the output shape would not be cropped.
        data_format: A string, only support `channels_last` here:
            `channels_last` corresponds to inputs with shape
            `(batch, steps channels)`
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        3D tensor with shape: `(batch_size, steps, input_dim)`
    Output shape:
        3D tensor with shape: `(batch_size, new_steps, filters)`
        `steps` value might have changed due to padding or strides.
    """

    def __init__(self, ofilters,
                 kernel_size,
                 lfilters=None,
                 depth=3,
                 strides=1,
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
                 dropout=None,
                 dropout_rate=0.3,
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 **kwargs):
        super(Inceptres1DTranspose, self).__init__(
            rank=1, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            output_mshape=output_mshape,
            output_padding=output_padding,
            output_cropping=output_cropping,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
            
class Inceptres2DTranspose(_InceptresTranspose):
    """Modern transposed inception residual layer (sometimes called inceptres
                                                   deconvolution).
    `Inceptres2DTranspose` implements the operation:
        `output = Conv2D(Upsamp(input)) + Conv2D( concat (i=0~D) Conv2DBranch(D, Upsamp(input)) )`
    where `Conv2DBranch` means D-1 times convolutional layers.
    The upsampling is performed on the input layer. Previous works prove that the
    "transposed convolution" could be viewed as upsampling + plain convolution. Here
    we adopt such a technique to realize this upsampling architecture.
    Arguments for inception residual block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the height and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        output_mshape: (Only avaliable for new-style API) An integer or tuple/list
            of the desired output shape. When setting this option, `output_padding`
            and `out_cropping` would be inferred from the input shape, which means
            users' options would be invalid for the following two options.
            A recommended method of using this method is applying such a scheme:
                `AConv(..., output_mshape=tensor.get_shape())`
        output_padding: An integer or tuple/list of 2 integers,
            specifying the amount of padding along the height and width
            of the output tensor.
            Can be a single integer to specify the same value for all
            spatial dimensions.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape would not be padded.
        out_cropping: (Only avaliable for new-style API) An integer or tuple/list 
            of n integers, specifying the amount of cropping along the axes of the
            output tensor. The amount of output cropping along a given dimension must
            be lower than the stride along that same dimension.
            If set to `None` (default), the output shape would not be cropped.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self, ofilters,
                 kernel_size,
                 lfilters=None,
                 depth=3,
                 strides=(1, 1),
                 output_mshape=None,
                 output_padding=None,
                 output_cropping=None,
                 data_format=None,
                 dilation_rate=(1, 1),
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
                 dropout=None,
                 dropout_rate=0.3,
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 **kwargs):
        super(Inceptres2DTranspose, self).__init__(
            rank=2, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            output_mshape=output_mshape,
            output_padding=output_padding,
            output_cropping=output_cropping,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
            
class Inceptres3DTranspose(_InceptresTranspose):
    """Modern transposed inception residual layer (sometimes called inceptres
                                                   deconvolution).
    `Inceptres3DTranspose` implements the operation:
        `output = Conv3D(Upsamp(input)) + Conv3D( concat (i=0~D) Conv3DBranch(D, Upsamp(input)) )`
    where `Conv3DBranch` means D-1 times convolutional layers.
    The upsampling is performed on the input layer. Previous works prove that the
    "transposed convolution" could be viewed as upsampling + plain convolution. Here
    we adopt such a technique to realize this upsampling architecture.
    Arguments for inception residual block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            depth, height and width of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 3 integers,
            specifying the strides of the convolution along the depth, height
            and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        output_mshape: (Only avaliable for new-style API) An integer or tuple/list
            of the desired output shape. When setting this option, `output_padding`
            and `out_cropping` would be inferred from the input shape, which means
            users' options would be invalid for the following two options.
            A recommended method of using this method is applying such a scheme:
                `AConv(..., output_mshape=tensor.get_shape())`
        output_padding: An integer or tuple/list of 3 integers,
            specifying the amount of padding along the depth, height, and
            width.
            Can be a single integer to specify the same value for all
            spatial dimensions.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape is inferred.
        out_cropping: (Only avaliable for new-style API) An integer or tuple/list 
            of n integers, specifying the amount of cropping along the axes of the
            output tensor. The amount of output cropping along a given dimension must
            be lower than the stride along that same dimension.
            If set to `None` (default), the output shape would not be cropped.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, depth, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, depth, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 3 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        5D tensor with shape:
        `(batch, channels, depth, rows, cols)` if data_format='channels_first'
        or 5D tensor with shape:
        `(batch, depth, rows, cols, channels)` if data_format='channels_last'.
    Output shape:
        5D tensor with shape:
        `(batch, filters, new_depth, new_rows, new_cols)` if
        data_format='channels_first'
        or 5D tensor with shape:
        `(batch, new_depth, new_rows, new_cols, filters)` if
        data_format='channels_last'.
        `depth` and `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self, ofilters,
                 kernel_size,
                 lfilters=None,
                 depth=3,
                 strides=(1, 1, 1),
                 output_mshape=None,
                 output_padding=None,
                 output_cropping=None,
                 data_format=None,
                 dilation_rate=(1, 1, 1),
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
                 dropout=None,
                 dropout_rate=0.3,
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 **kwargs):
        super(Inceptres3DTranspose, self).__init__(
            rank=3, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            output_mshape=output_mshape,
            output_padding=output_padding,
            output_cropping=output_cropping,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)

class _Inceptplus(Layer):
    """Modern inception plus layer.
    Abstract nD inception-plus layer (private, used as implementation base).
    `_Inceptplus` implements the operation:
        `output = Conv(input) + Conv( concat (i=1~D) ConvBranch( D, input - Average(D, input) ) )`
    This module is adapted from inception-residual layer. This structure is based
    on the theory about reception field. Both average filter and convolution have
    reception field. Stacking several layers would expand the reception field.
    Therefore, this structure could be viewed as convolution performed on the res-
    idual between input and the average in the same reception field.
    The experience on dictionary learning shows that learn the residual rather
    than input may reduce the bias brought by low-pass data. In this structure,
    we implement the similar technique and let the inception-residual block be 
    able to learn residuals in different reception fields.
    Arguments for inception plus block:
        rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of n integers, specifying the
            length of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
                 depth, ofilters,
                 kernel_size,
                 lfilters=None,
                 strides=1,
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
                 dropout=None,
                 dropout_rate=0.3,
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 _high_activation=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
          kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(_Inceptplus, self).__init__(trainable=trainable, name=name, **kwargs)
        # Inherit from keras.layers._Conv
        self.rank = rank
        self.depth = depth
        if depth < 1:
            raise ValueError('The depth of the inception block should be >= 1.')
        self.ofilters = ofilters
        self.lfilters = lfilters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')
        if (not _check_dl_func(self.dilation_rate)) and (not _check_dl_func(self.strides)):
            raise ValueError('Does not support dilation_rate when strides > 1.')
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        # Inherit from mdnt.layers.normalize
        self.normalization = normalization
        if isinstance(normalization, str) and normalization in ('batch', 'inst', 'group'):
            self.gamma_initializer = initializers.get(gamma_initializer)
            self.gamma_regularizer = regularizers.get(gamma_regularizer)
            self.gamma_constraint = constraints.get(gamma_constraint)
        else:
            self.gamma_initializer = None
            self.gamma_regularizer = None
            self.gamma_constraint = None
        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.groups = groups
        # Inherit from mdnt.layers.dropout
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        # Inherit from keras.engine.Layer
        if _high_activation is not None:
            activation = _high_activation
        self.high_activation = _high_activation
        if isinstance(activation, str) and (activation.casefold() in ('prelu','lrelu')):
            self.activation = activations.get(None)
            self.high_activation = activation.casefold()
            self.activity_config = activity_config # dictionary passed to activation
        elif activation is not None:
            self.activation = activations.get(activation)
            self.activity_config = None
        self.sub_activity_regularizer=regularizers.get(activity_regularizer)

        # Reserve for build()
        self.channelIn = None
        
        self.trainable = trainable
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(self.rank + 2)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        self.channelIn = int(input_shape[channel_axis])
        if self.lfilters is None:
            self.lfilters = max( 1, self.channelIn // 2 )
        # Here we define the left branch
        last_use_bias = True
        if _check_dl_func(self.strides) and self.ofilters == self.channelIn:
            self.layer_branch_left = None
            left_shape = input_shape
        else:
            last_use_bias = False
            self.layer_branch_left = _AConv(rank = self.rank,
                          filters = self.ofilters,
                          kernel_size = 1,
                          strides = self.strides,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=None,
                          activity_config=None,
                          activity_regularizer=None,
                          _high_activation=None,
                          trainable=self.trainable)
            self.layer_branch_left.build(input_shape)
            compat.collect_properties(self, self.layer_branch_left) # for compatibility
            left_shape = self.layer_branch_left.compute_output_shape(input_shape)
        # Here we define the right branch, with dropout
        self.layer_dropout = return_dropout(self.dropout, self.dropout_rate, axis=channel_axis, rank=self.rank)
        if self.layer_dropout is not None:
            self.layer_dropout.build(input_shape)
            depth_shape = self.layer_dropout.compute_output_shape(input_shape)
        else:
            depth_shape = input_shape
        # First, need to calculate N depth average filter
        depth_shape_list = []
        branch_avg_shape = depth_shape
        # The branch (depth) D (D>=1) average filter
        for D in range(self.depth):
            # First, create average layer
            if self.rank == 1:
                layer_avg = AveragePooling1D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            elif self.rank == 2:
                layer_avg = AveragePooling2D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            elif self.rank == 3:
                layer_avg = AveragePooling3D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            else:
                raise ValueError('Rank of the inception should be 1, 2 or 3.')
            layer_avg.build(branch_avg_shape)
            branch_avg_shape = layer_avg.compute_output_shape(branch_avg_shape)
            setattr(self, 'layer_avg_D{0:02d}'.format(D+1), layer_avg)
            # Second use substract layer to perform input - avg
            layer_middle_input = Subtract()
            layer_middle_input.build([depth_shape, branch_avg_shape])
            branch_shape = layer_middle_input.compute_output_shape([depth_shape, branch_avg_shape])
            setattr(self, 'layer_middle_D{0:02d}_inp'.format(D+1), layer_middle_input)
            # Third, use 1 x 1 filter perform strides and channel mapping
            layer_middle_first = NACUnit(rank = self.rank,
                          filters = self.lfilters,
                          kernel_size = 1,
                          strides = self.strides,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
            layer_middle_first.build(branch_shape)
            compat.collect_properties(self, layer_middle_first) # for compatibility
            branch_shape = layer_middle_first.compute_output_shape(branch_shape)
            setattr(self, 'layer_middle_D{0:02d}_00'.format(D+1), layer_middle_first)
            for i in range(D+1):
                layer_middle = NACUnit(rank = self.rank,
                          filters = self.lfilters,
                          kernel_size = self.kernel_size,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = self.dilation_rate,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
                layer_middle.build(branch_shape)
                compat.collect_properties(self, layer_middle) # for compatibility
                branch_shape = layer_middle.compute_output_shape(branch_shape)
                setattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+1, i+1), layer_middle)
            depth_shape_list.append(branch_shape)
        # Merge the right branch by concatnation.
        if self.data_format == 'channels_first':
            self.layer_branch_right = Concatenate(axis=1)
        else:
            self.layer_branch_right = Concatenate()
        self.layer_branch_right.build(depth_shape_list)
        right_shape = self.layer_branch_right.compute_output_shape(depth_shape_list)
        self.layer_branch_right_map = NACUnit(rank = self.rank,
                          filters = self.ofilters,
                          kernel_size = 1,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          _use_bias=last_use_bias,
                          trainable=self.trainable)
        self.layer_branch_right_map.build(right_shape)
        compat.collect_properties(self, self.layer_branch_right_map) # for compatibility
        right_shape = self.layer_branch_right_map.compute_output_shape(right_shape)
        # Merge the residual block
        self.layer_merge = Subtract()
        self.layer_merge.build([left_shape, right_shape])
        super(_Inceptplus, self).build(input_shape)

    def call(self, inputs):
        # Left branch
        if self.layer_branch_left is not None:
            branch_left = self.layer_branch_left(inputs)
        else:
            branch_left = inputs
        # Right branch
        if self.layer_dropout is not None:
            depth_input = self.layer_dropout(inputs)
        else:
            depth_input = inputs
        branch_middle_list = []
        branch_avg = depth_input
        for D in range(self.depth):
            layer_avg = getattr(self, 'layer_avg_D{0:02d}'.format(D+1))
            branch_avg = layer_avg(branch_avg)
            layer_middle_input = getattr(self, 'layer_middle_D{0:02d}_inp'.format(D+1))
            branch_middle = layer_middle_input([depth_input, branch_avg])
            layer_middle_first = getattr(self, 'layer_middle_D{0:02d}_00'.format(D+1))
            branch_middle = layer_middle_first(branch_middle)
            for i in range(D+1):
                layer_middle = getattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+1, i+1))
                branch_middle = layer_middle(branch_middle)
            branch_middle_list.append(branch_middle)
        branch_right = self.layer_branch_right(branch_middle_list)
        branch_right = self.layer_branch_right_map(branch_right)
        outputs = self.layer_merge([branch_left, branch_right])
        return outputs

    def compute_output_shape(self, input_shape):
        if self.layer_branch_left is not None:
            branch_left_shape = self.layer_branch_left.compute_output_shape(input_shape)
        else:
            branch_left_shape = input_shape
        if self.layer_dropout is not None:
            depth_input_shape = self.layer_dropout.compute_output_shape(input_shape)
        else:
            depth_input_shape = input_shape
        branch_middle_shape_list = []
        branch_avg_shape = depth_input_shape
        for D in range(self.depth):
            layer_avg = getattr(self, 'layer_avg_D{0:02d}'.format(D+1))
            branch_avg_shape = layer_avg.compute_output_shape(branch_avg_shape)
            layer_middle_input = getattr(self, 'layer_middle_D{0:02d}_inp'.format(D+1))
            branch_middle_shape = layer_middle_input.compute_output_shape([depth_input_shape, branch_avg_shape])
            layer_middle_first = getattr(self, 'layer_middle_D{0:02d}_00'.format(D+1))
            branch_middle_shape = layer_middle_first.compute_output_shape(branch_middle_shape)
            for i in range(D+1):
                layer_middle = getattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+1, i+1))
                branch_middle_shape = layer_middle.compute_output_shape(branch_middle_shape)
            branch_middle_shape_list.append(branch_middle_shape)
        branch_right_shape = self.layer_branch_right.compute_output_shape(branch_middle_shape_list)
        branch_right_shape = self.layer_branch_right_map.compute_output_shape(branch_right_shape)
        next_shape = self.layer_merge.compute_output_shape([branch_left_shape, branch_right_shape])
        return next_shape
    
    def get_config(self):
        config = {
            'depth': self.depth,
            'ofilters': self.ofilters,
            'lfilters': self.lfilters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
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
            'dropout': self.dropout,
            'dropout_rate': self.dropout_rate,
            'activation': activations.serialize(self.activation),
            'activity_config': self.activity_config,
            'activity_regularizer': regularizers.serialize(self.sub_activity_regularizer),
            '_high_activation': self.high_activation
        }
        base_config = super(_Inceptplus, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
class Inceptplus1D(_Inceptplus):
    """1D inception plus layer.
    `Inceptplus1D` implements the operation:
        `output = Conv1D(input) + Conv1D( concat (i=1~D) Conv1DBranch( D, input - Average(D, input) ) )`
    where `Average` means D-depth average filter. Since we apply average filters
    in all branches, the zero branch in inception block is removed.
    This module is adapted from inception-residual layer. This structure is based
    on the theory about reception field. Both average filter and convolution have
    reception field. Stacking several layers would expand the reception field.
    Therefore, this structure could be viewed as convolution performed on the res-
    idual between input and the average in the same reception field.
    The experience on dictionary learning shows that learn the residual rather
    than input may reduce the bias brought by low-pass data. In this structure,
    we implement the similar technique and let the inception-residual block be 
    able to learn residuals in different reception fields.
    Arguments for inception plus block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        kernel_constraint: Constraint function applied to the kernel matrix.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        3D tensor with shape: `(batch_size, steps, input_dim)`
    Output shape:
        3D tensor with shape: `(batch_size, new_steps, filters)`
        `steps` value might have changed due to padding or strides.
    """

    def __init__(self,
               ofilters,
               kernel_size,
               lfilters=None,
               depth=2,
               strides=1,
               data_format='channels_last',
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
               dropout=None,
               dropout_rate=0.3,
               activation=None,
               activity_config=None,
               activity_regularizer=None,
               **kwargs):
        super(Inceptplus1D, self).__init__(
            rank=1, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        
class Inceptplus2D(_Inceptplus):
    """2D inception residual layer (e.g. spatial convolution over images).
    `Inceptplus2D` implements the operation:
        `output = Conv2D(input) + Conv2D( concat (i=1~D) Conv2DBranch( D, input - Average(D, input) ) )`
    where `Average` means D-depth average filter. Since we apply average filters
    in all branches, the zero branch in inception block is removed.
    This module is adapted from inception-residual layer. This structure is based
    on the theory about reception field. Both average filter and convolution have
    reception field. Stacking several layers would expand the reception field.
    Therefore, this structure could be viewed as convolution performed on the res-
    idual between input and the average in the same reception field.
    The experience on dictionary learning shows that learn the residual rather
    than input may reduce the bias brought by low-pass data. In this structure,
    we implement the similar technique and let the inception-residual block be 
    able to learn residuals in different reception fields.
    Arguments for inception plus block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the height and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        kernel_constraint: Constraint function applied to the kernel matrix.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
               ofilters,
               kernel_size,
               lfilters=None,
               depth=2,
               strides=(1, 1),
               data_format='channels_last',
               dilation_rate=(1, 1),
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
               dropout=None,
               dropout_rate=0.3,
               activation=None,
               activity_config=None,
               activity_regularizer=None,
               **kwargs):
        super(Inceptplus2D, self).__init__(
            rank=2, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        
class Inceptplus3D(_Inceptplus):
    """3D inception residual layer (e.g. spatial convolution over volumes).
    `Inceptplus3D` implements the operation:
        `output = Conv3D(input) + Conv3D( concat (i=1~D) Conv3DBranch( D, input - Average(D, input) ) )`
    where `Average` means D-depth average filter. Since we apply average filters
    in all branches, the zero branch in inception block is removed.
    This module is adapted from inception-residual layer. This structure is based
    on the theory about reception field. Both average filter and convolution have
    reception field. Stacking several layers would expand the reception field.
    Therefore, this structure could be viewed as convolution performed on the res-
    idual between input and the average in the same reception field.
    The experience on dictionary learning shows that learn the residual rather
    than input may reduce the bias brought by low-pass data. In this structure,
    we implement the similar technique and let the inception-residual block be 
    able to learn residuals in different reception fields.
    Arguments for inception plus block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            depth, height and width of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 3 integers,
            specifying the strides of the convolution along each spatial
            dimension.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `channels_first` corresponds to inputs with shape
            `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 3 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        kernel_constraint: Constraint function applied to the kernel matrix.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        5D tensor with shape:
        `(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if
        data_format='channels_first'
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if
        data_format='channels_last'.
    Output shape:
        5D tensor with shape:
        `(samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if
        data_format='channels_first'
        or 5D tensor with shape:
        `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` if
        data_format='channels_last'.
        `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have
        changed due to padding.
    """

    def __init__(self,
               ofilters,
               kernel_size,
               lfilters=None,
               depth=2,
               strides=(1, 1, 1),
               data_format='channels_last',
               dilation_rate=(1, 1, 1),
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
               dropout=None,
               dropout_rate=0.3,
               activation=None,
               activity_config=None,
               activity_regularizer=None,
               **kwargs):
        super(Inceptplus3D, self).__init__(
            rank=3, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
            
class _InceptplusTranspose(Layer):
    """Modern transposed inception plus layer (sometimes called inceptplus
                                                             deconvolution).
    Abstract nD transposed inception plus layer (private, used as implementation base).
    `_InceptplusTranspose` implements the operation:
        ```output = Conv(Upsamp(input)) + Conv( concat (i=1~D)
                    ConvBranch( D, Upsamp(input) - Average(D, Upsamp(input)) ) )```
    where `ConvBranch` means D-1 times convolutional layers and `Average` means 
    D-depth average filter. Since we apply average filters in all branches, the 
    zero branch in inception block is removed.
    The upsampling is performed on the input layer. Previous works prove that the
    "transposed convolution" could be viewed as upsampling + plain convolution. Here
    we adopt such a technique to realize this upsampling architecture.
    Arguments for inception plus block:
        rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of n integers, specifying the
            length of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
                 depth, ofilters,
                 kernel_size,
                 lfilters=None,
                 strides=1,
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
                 dropout=None,
                 dropout_rate=0.3,
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 _high_activation=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
          kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(_InceptplusTranspose, self).__init__(trainable=trainable, name=name, **kwargs)
        # Inherit from keras.layers._Conv
        self.rank = rank
        self.depth = depth
        if depth < 1:
            raise ValueError('The depth of the inception block should be >= 1.')
        self.ofilters = ofilters
        self.lfilters = lfilters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.output_padding = output_padding
        self.output_mshape = None
        self.output_cropping = None
        if output_mshape:
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
            self.gamma_initializer = initializers.get(gamma_initializer)
            self.gamma_regularizer = regularizers.get(gamma_regularizer)
            self.gamma_constraint = constraints.get(gamma_constraint)
        else:
            self.gamma_initializer = None
            self.gamma_regularizer = None
            self.gamma_constraint = None
        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.groups = groups
        # Inherit from mdnt.layers.dropout
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        # Inherit from keras.engine.Layer
        if _high_activation is not None:
            activation = _high_activation
        self.high_activation = _high_activation
        if isinstance(activation, str) and (activation.casefold() in ('prelu','lrelu')):
            self.activation = activations.get(None)
            self.high_activation = activation.casefold()
            self.activity_config = activity_config # dictionary passed to activation
        elif activation is not None:
            self.activation = activations.get(activation)
            self.activity_config = None
        self.sub_activity_regularizer=regularizers.get(activity_regularizer)

        # Reserve for build()
        self.channelIn = None
        
        self.trainable = trainable
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(self.rank + 2)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        self.channelIn = int(input_shape[channel_axis])
        if self.lfilters is None:
            self.lfilters = max( 1, self.channelIn // 2 )
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
                get_shape_diff = l_output_mshape[i] - l_input_shape[i]*max(self.strides[i], self.dilation_rate[i])
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
            self.layer_uppool.build(input_shape)
            next_shape = self.layer_uppool.compute_output_shape(input_shape)
            if self.output_padding is not None:
                self.layer_padding = ZeroPadding1D(padding=self.output_padding)[0] # Necessary for 1D case, because we need to pick (a,b) from ((a, b))
                self.layer_padding.build(next_shape)
                next_shape = self.layer_padding.compute_output_shape(next_shape)
            else:
                self.layer_padding = None
        elif self.rank == 2:
            self.layer_uppool = UpSampling2D(size=self.strides, data_format=self.data_format)
            self.layer_uppool.build(input_shape)
            next_shape = self.layer_uppool.compute_output_shape(input_shape)
            if self.output_padding is not None:
                self.layer_padding = ZeroPadding2D(padding=self.output_padding, data_format=self.data_format)
                self.layer_padding.build(next_shape)
                next_shape = self.layer_padding.compute_output_shape(next_shape)
            else:
                self.layer_padding = None
        elif self.rank == 3:
            self.layer_uppool = UpSampling3D(size=self.strides, data_format=self.data_format)
            self.layer_uppool.build(input_shape)
            next_shape = self.layer_uppool.compute_output_shape(input_shape)
            if self.output_padding is not None:
                self.layer_padding = ZeroPadding3D(padding=self.output_padding, data_format=self.data_format)
                self.layer_padding.build(next_shape)
                next_shape = self.layer_padding.compute_output_shape(next_shape)
            else:
                self.layer_padding = None
        else:
            raise ValueError('Rank of the deconvolution should be 1, 2 or 3.')
        # Here we define the left branch
        last_use_bias = True
        if self.ofilters == self.channelIn:
            self.layer_branch_left = None
            left_shape = next_shape
        else:
            last_use_bias = False
            self.layer_branch_left = _AConv(rank = self.rank,
                          filters = self.ofilters,
                          kernel_size = 1,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=None,
                          activity_config=None,
                          activity_regularizer=None,
                          _high_activation=None,
                          trainable=self.trainable)
            self.layer_branch_left.build(next_shape)
            compat.collect_properties(self, self.layer_branch_left) # for compatibility
            left_shape = self.layer_branch_left.compute_output_shape(next_shape)
        # Here we define the right branch, with dropout
        self.layer_dropout = return_dropout(self.dropout, self.dropout_rate, axis=channel_axis, rank=self.rank)
        if self.layer_dropout is not None:
            self.layer_dropout.build(next_shape)
            depth_shape = self.layer_dropout.compute_output_shape(next_shape)
        else:
            depth_shape = next_shape
        # First, need to calculate N depth average filter
        depth_shape_list = []
        branch_avg_shape = depth_shape
        # The branch (depth) D (D>=1) average filter
        for D in range(self.depth):
            # First, create average layer
            if self.rank == 1:
                layer_avg = AveragePooling1D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            elif self.rank == 2:
                layer_avg = AveragePooling2D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            elif self.rank == 3:
                layer_avg = AveragePooling3D(pool_size=self.kernel_size, strides=1, padding='same', data_format=self.data_format)
            else:
                raise ValueError('Rank of the inception should be 1, 2 or 3.')
            layer_avg.build(branch_avg_shape)
            branch_avg_shape = layer_avg.compute_output_shape(branch_avg_shape)
            setattr(self, 'layer_avg_D{0:02d}'.format(D+1), layer_avg)
            # Second use substract layer to perform input - avg
            layer_middle_input = Subtract()
            layer_middle_input.build([depth_shape, branch_avg_shape])
            branch_shape = layer_middle_input.compute_output_shape([depth_shape, branch_avg_shape])
            setattr(self, 'layer_middle_D{0:02d}_inp'.format(D+1), layer_middle_input)
            # Third, use 1 x 1 filter perform strides and channel mapping
            layer_middle_first = NACUnit(rank = self.rank,
                          filters = self.lfilters,
                          kernel_size = 1,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
            layer_middle_first.build(branch_shape)
            compat.collect_properties(self, layer_middle_first) # for compatibility
            branch_shape = layer_middle_first.compute_output_shape(branch_shape)
            setattr(self, 'layer_middle_D{0:02d}_00'.format(D+1), layer_middle_first)
            for i in range(D+1):
                layer_middle = NACUnit(rank = self.rank,
                          filters = self.lfilters,
                          kernel_size = self.kernel_size,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = self.dilation_rate,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          trainable=self.trainable)
                layer_middle.build(branch_shape)
                compat.collect_properties(self, layer_middle) # for compatibility
                branch_shape = layer_middle.compute_output_shape(branch_shape)
                setattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+1, i+1), layer_middle)
            depth_shape_list.append(branch_shape)
        # Merge the right branch by concatnation.
        if self.data_format == 'channels_first':
            self.layer_branch_right = Concatenate(axis=1)
        else:
            self.layer_branch_right = Concatenate()
        self.layer_branch_right.build(depth_shape_list)
        right_shape = self.layer_branch_right.compute_output_shape(depth_shape_list)
        self.layer_branch_right_map = NACUnit(rank = self.rank,
                          filters = self.ofilters,
                          kernel_size = 1,
                          strides = 1,
                          padding = 'same',
                          data_format = self.data_format,
                          dilation_rate = 1,
                          kernel_initializer=self.kernel_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          kernel_constraint=self.kernel_constraint,
                          normalization=self.normalization,
                          beta_initializer=self.beta_initializer,
                          gamma_initializer=self.gamma_initializer,
                          beta_regularizer=self.beta_regularizer,
                          gamma_regularizer=self.gamma_regularizer,
                          beta_constraint=self.beta_constraint,
                          gamma_constraint=self.gamma_constraint,
                          groups=self.groups,
                          activation=self.activation,
                          activity_config=self.activity_config,
                          activity_regularizer=self.sub_activity_regularizer,
                          _high_activation=self.high_activation,
                          _use_bias=last_use_bias,
                          trainable=self.trainable)
        self.layer_branch_right_map.build(right_shape)
        compat.collect_properties(self, self.layer_branch_right_map) # for compatibility
        right_shape = self.layer_branch_right_map.compute_output_shape(right_shape)
        # Merge the residual block
        self.layer_merge = Subtract()
        self.layer_merge.build([left_shape, right_shape])
        next_shape = self.layer_merge.compute_output_shape([left_shape, right_shape])
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
        super(_InceptplusTranspose, self).build(next_shape)

    def call(self, inputs):
        outputs = self.layer_uppool(inputs)
        if self.layer_padding is not None:
            outputs = self.layer_padding(outputs)
        # Left branch
        if self.layer_branch_left is not None:
            branch_left = self.layer_branch_left(outputs)
        else:
            branch_left = outputs
        # Right branch
        if self.layer_dropout is not None:
            depth_input = self.layer_dropout(outputs)
        else:
            depth_input = outputs
        branch_middle_list = []
        branch_avg = depth_input
        for D in range(self.depth):
            layer_avg = getattr(self, 'layer_avg_D{0:02d}'.format(D+1))
            branch_avg = layer_avg(branch_avg)
            layer_middle_input = getattr(self, 'layer_middle_D{0:02d}_inp'.format(D+1))
            branch_middle = layer_middle_input([depth_input, branch_avg])
            layer_middle_first = getattr(self, 'layer_middle_D{0:02d}_00'.format(D+1))
            branch_middle = layer_middle_first(branch_middle)
            for i in range(D+1):
                layer_middle = getattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+1, i+1))
                branch_middle = layer_middle(branch_middle)
            branch_middle_list.append(branch_middle)
        branch_right = self.layer_branch_right(branch_middle_list)
        branch_right = self.layer_branch_right_map(branch_right)
        outputs = self.layer_merge([branch_left, branch_right])
        if self.layer_cropping is not None:
            outputs = self.layer_cropping(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(self.rank + 2)
        next_shape = self.layer_uppool.compute_output_shape(input_shape)
        if self.layer_padding is not None:
            next_shape = self.layer_padding.compute_output_shape(next_shape)
        if self.layer_branch_left is not None:
            branch_left_shape = self.layer_branch_left.compute_output_shape(next_shape)
        else:
            branch_left_shape = next_shape
        if self.layer_dropout is not None:
            depth_input_shape = self.layer_dropout.compute_output_shape(next_shape)
        else:
            depth_input_shape = next_shape
        branch_middle_shape_list = []
        branch_avg_shape = depth_input_shape
        for D in range(self.depth):
            layer_avg = getattr(self, 'layer_avg_D{0:02d}'.format(D+1))
            branch_avg_shape = layer_avg.compute_output_shape(branch_avg_shape)
            layer_middle_input = getattr(self, 'layer_middle_D{0:02d}_inp'.format(D+1))
            branch_middle_shape = layer_middle_input.compute_output_shape([depth_input_shape, branch_avg_shape])
            layer_middle_first = getattr(self, 'layer_middle_D{0:02d}_00'.format(D+1))
            branch_middle_shape = layer_middle_first.compute_output_shape(branch_middle_shape)
            for i in range(D+1):
                layer_middle = getattr(self, 'layer_middle_D{0:02d}_{1:02d}'.format(D+1, i+1))
                branch_middle_shape = layer_middle.compute_output_shape(branch_middle_shape)
            branch_middle_shape_list.append(branch_middle_shape)
        branch_right_shape = self.layer_branch_right.compute_output_shape(branch_middle_shape_list)
        branch_right_shape = self.layer_branch_right_map.compute_output_shape(branch_right_shape)
        next_shape = self.layer_merge.compute_output_shape([branch_left_shape, branch_right_shape])
        if self.layer_cropping is not None:
            next_shape = self.layer_cropping.compute_output_shape(next_shape)
        return next_shape
    
    def get_config(self):
        config = {
            'depth': self.depth,
            'ofilters': self.ofilters,
            'lfilters': self.lfilters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
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
            'dropout': self.dropout,
            'dropout_rate': self.dropout_rate,
            'activation': activations.serialize(self.activation),
            'activity_config': self.activity_config,
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            '_high_activation': self.high_activation
        }
        base_config = super(_InceptplusTranspose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
class Inceptplus1DTranspose(_InceptplusTranspose):
    """Modern transposed inception residual layer (sometimes called inceptres
                                                   deconvolution).
    `Inceptplus1DTranspose` implements the operation:
        ```output = Conv1D(Upsamp(input)) + Conv1D( concat (i=1~D)
                    Conv1DBranch( D, Upsamp(input) - Average(D, Upsamp(input)) ) )```
    where `Conv1DBranch` means D-1 times convolutional layers and `Average` means 
    D-depth average filter. Since we apply average filters in all branches, the 
    zero branch in inception block is removed.
    The upsampling is performed on the input layer. Previous works prove that the
    "transposed convolution" could be viewed as upsampling + plain convolution. Here
    we adopt such a technique to realize this upsampling architecture.
    Arguments for inception plus block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of n integers, specifying the
            length of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        output_mshape: (Only avaliable for new-style API) An integer or tuple/list
            of the desired output shape. When setting this option, `output_padding`
            and `out_cropping` would be inferred from the input shape, which means
            users' options would be invalid for the following two options.
            A recommended method of using this method is applying such a scheme:
                `AConv(..., output_mshape=tensor.get_shape())`
        output_padding: An integer or tuple/list of n integers,
            specifying the amount of padding along the height and width
            of the output tensor.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape would not be padded.
        out_cropping: (Only avaliable for new-style API) An integer or tuple/list 
            of n integers, specifying the amount of cropping along the axes of the
            output tensor. The amount of output cropping along a given dimension must
            be lower than the stride along that same dimension.
            If set to `None` (default), the output shape would not be cropped.
        data_format: A string, only support `channels_last` here:
            `channels_last` corresponds to inputs with shape
            `(batch, steps channels)`
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        3D tensor with shape: `(batch_size, steps, input_dim)`
    Output shape:
        3D tensor with shape: `(batch_size, new_steps, filters)`
        `steps` value might have changed due to padding or strides.
    """

    def __init__(self, ofilters,
                 kernel_size,
                 lfilters=None,
                 depth=2,
                 strides=1,
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
                 dropout=None,
                 dropout_rate=0.3,
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 **kwargs):
        super(Inceptplus1DTranspose, self).__init__(
            rank=1, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            output_mshape=output_mshape,
            output_padding=output_padding,
            output_cropping=output_cropping,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
            
class Inceptplus2DTranspose(_InceptplusTranspose):
    """Modern transposed inception residual layer (sometimes called inceptres
                                                   deconvolution).
    `Inceptplus2DTranspose` implements the operation:
        ```output = Conv2D(Upsamp(input)) + Conv2D( concat (i=1~D)
                    Conv2DBranch( D, Upsamp(input) - Average(D, Upsamp(input)) ) )```
    where `Conv2DBranch` means D-1 times convolutional layers and `Average` means 
    D-depth average filter. Since we apply average filters in all branches, the 
    zero branch in inception block is removed.
    The upsampling is performed on the input layer. Previous works prove that the
    "transposed convolution" could be viewed as upsampling + plain convolution. Here
    we adopt such a technique to realize this upsampling architecture.
    Arguments for inception plus block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the height and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        output_mshape: (Only avaliable for new-style API) An integer or tuple/list
            of the desired output shape. When setting this option, `output_padding`
            and `out_cropping` would be inferred from the input shape, which means
            users' options would be invalid for the following two options.
            A recommended method of using this method is applying such a scheme:
                `AConv(..., output_mshape=tensor.get_shape())`
        output_padding: An integer or tuple/list of 2 integers,
            specifying the amount of padding along the height and width
            of the output tensor.
            Can be a single integer to specify the same value for all
            spatial dimensions.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape would not be padded.
        out_cropping: (Only avaliable for new-style API) An integer or tuple/list 
            of n integers, specifying the amount of cropping along the axes of the
            output tensor. The amount of output cropping along a given dimension must
            be lower than the stride along that same dimension.
            If set to `None` (default), the output shape would not be cropped.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    Output shape:
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self, ofilters,
                 kernel_size,
                 lfilters=None,
                 depth=2,
                 strides=(1, 1),
                 output_mshape=None,
                 output_padding=None,
                 output_cropping=None,
                 data_format=None,
                 dilation_rate=(1, 1),
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
                 dropout=None,
                 dropout_rate=0.3,
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 **kwargs):
        super(Inceptplus2DTranspose, self).__init__(
            rank=2, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            output_mshape=output_mshape,
            output_padding=output_padding,
            output_cropping=output_cropping,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
            
class Inceptplus3DTranspose(_InceptplusTranspose):
    """Modern transposed inception residual layer (sometimes called inceptres
                                                   deconvolution).
    `Inceptplus3DTranspose` implements the operation:
        ```output = Conv3D(Upsamp(input)) + Conv3D( concat (i=1~D)
                    Conv3DBranch( D, Upsamp(input) - Average(D, Upsamp(input)) ) )```
    where `Conv3DBranch` means D-1 times convolutional layers and `Average` means 
    D-depth average filter. Since we apply average filters in all branches, the 
    zero branch in inception block is removed.
    The upsampling is performed on the input layer. Previous works prove that the
    "transposed convolution" could be viewed as upsampling + plain convolution. Here
    we adopt such a technique to realize this upsampling architecture.
    Arguments for inception plus block:
        depth: An integer, indicates the number of network branches.
        ofilters: Integer, the dimensionality of the output space (i.e. the number
            of filters of output).
        lfilters: Integer, the dimensionality of the lattent space (i.e. the number
            of filters in the convolution branch).
    Arguments for convolution:
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            depth, height and width of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 3 integers,
            specifying the strides of the convolution along the depth, height
            and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        output_mshape: (Only avaliable for new-style API) An integer or tuple/list
            of the desired output shape. When setting this option, `output_padding`
            and `out_cropping` would be inferred from the input shape, which means
            users' options would be invalid for the following two options.
            A recommended method of using this method is applying such a scheme:
                `AConv(..., output_mshape=tensor.get_shape())`
        output_padding: An integer or tuple/list of 3 integers,
            specifying the amount of padding along the depth, height, and
            width.
            Can be a single integer to specify the same value for all
            spatial dimensions.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape is inferred.
        out_cropping: (Only avaliable for new-style API) An integer or tuple/list 
            of n integers, specifying the amount of cropping along the axes of the
            output tensor. The amount of output cropping along a given dimension must
            be lower than the stride along that same dimension.
            If set to `None` (default), the output shape would not be cropped.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, depth, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, depth, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 3 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
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
    Arguments for dropout: (drop out would be only applied on the entrance
                            of conv. branch.)
        dropout: The dropout type, which could be
            (1) None:    do not use dropout.
            (2) plain:   use tf.keras.layers.Dropout.
            (3) add:     use scale-invariant addictive noise.
                         (mdnt.layers.InstanceGaussianNoise)
            (4) mul:     use multiplicative noise.
                         (tf.keras.layers.GaussianDropout)
            (5) alpha:   use alpha dropout. (tf.keras.layers.AlphaDropout)
            (6) spatial: use spatial dropout (tf.keras.layers.SpatialDropout)
        dropout_rate: The drop probability. In `add` mode, it is used as
            maximal std. To learn more, please see the docstrings of each
            method.
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
    Input shape:
        5D tensor with shape:
        `(batch, channels, depth, rows, cols)` if data_format='channels_first'
        or 5D tensor with shape:
        `(batch, depth, rows, cols, channels)` if data_format='channels_last'.
    Output shape:
        5D tensor with shape:
        `(batch, filters, new_depth, new_rows, new_cols)` if
        data_format='channels_first'
        or 5D tensor with shape:
        `(batch, new_depth, new_rows, new_cols, filters)` if
        data_format='channels_last'.
        `depth` and `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self, ofilters,
                 kernel_size,
                 lfilters=None,
                 depth=2,
                 strides=(1, 1, 1),
                 output_mshape=None,
                 output_padding=None,
                 output_cropping=None,
                 data_format=None,
                 dilation_rate=(1, 1, 1),
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
                 dropout=None,
                 dropout_rate=0.3,
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 **kwargs):
        super(Inceptplus3DTranspose, self).__init__(
            rank=3, depth=depth, ofilters=ofilters,
            kernel_size=kernel_size,
            lfilters=lfilters,
            strides=strides,
            output_mshape=output_mshape,
            output_padding=output_padding,
            output_cropping=output_cropping,
            data_format=data_format,
            dilation_rate=dilation_rate,
            kernel_initializer=initializers.get(kernel_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            normalization=normalization,
            beta_initializer=initializers.get(beta_initializer),
            gamma_initializer=initializers.get(gamma_initializer),
            beta_regularizer=regularizers.get(beta_regularizer),
            gamma_regularizer=regularizers.get(gamma_regularizer),
            beta_constraint=constraints.get(beta_constraint),
            gamma_constraint=constraints.get(gamma_constraint),
            groups=groups,
            dropout=dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)