'''
################################################################
# Layers - Modern convolutional layers
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# A modern convolutional layer could be written as:
#   PRelu ( gamma * [( conv(x, W) - mu ) / sigma ] + beta )
# which indicates that it should contain:
#   1. A convolutional kernel.
#   2. A normalization layer.
#   3. Activation.
# We recommend to use instance normalization and PRelu in most
# cases. This idea is introduced in
#   https://arxiv.org/abs/1502.03167v3
# To learn transposed convolution, see
#   https://arxiv.org/abs/1603.07285v1
#   http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf
# We recommend users to use new work-flow for transposed 
# convolutional layers, if user want to switch back to old
# style of Keras, please set this macro:
#   mdnt.layers.conv.NEW_CONV_TRANSPOSE = False
# Here we also implement some tied convolutional layers, note
# that it is necessary to set name scope if using them in multi-
# models.
# Version: 0.20 # 2019/3/26
# Comments:
#   Add transposed convolutional layers to this handle.
# Version: 0.10 # 2019/3/25
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
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import array_ops

from tensorflow.keras.layers import BatchNormalization, LeakyReLU, PReLU
from tensorflow.python.keras.layers.convolutional import Conv, Conv2DTranspose, Conv3DTranspose, UpSampling1D, UpSampling2D, UpSampling3D, ZeroPadding1D, ZeroPadding2D, ZeroPadding3D
from .normalize import InstanceNormalization, GroupNormalization

NEW_CONV_TRANSPOSE = True

def _get_macro():
    return NEW_CONV_TRANSPOSE

class _AConv(Layer):
    """Modern convolutional layer.
    Abstract nD convolution layer (private, used as implementation base).
    `_AConv` implements the operation:
    `output = activation( normalization( conv(x, W), gamma, beta ), alpha )`
    This layer is a stack of convolution, normalization and activation.
    As an extension, we allow users to use activating layers with parameters
    like PRelu.
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
            function.
    Arguments (others):
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
    """

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
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
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
          kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(_AConv, self).__init__(trainable=trainable, name=name, activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        # Inherit from keras.layers._Conv
        self.rank = rank
        self.filters = filters
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
        elif not normalization:
            self.use_bias = False
            self.gamma_initializer = None
            self.gamma_regularizer = None
            self.gamma_constraint = None
        else:
            self.use_bias = True
            self.gamma_initializer = None
            self.gamma_regularizer = None
            self.gamma_constraint = None
        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.groups = groups
        # Inherit from keras.engine.Layer
        self.high_activation = _high_activation
        self.use_plain_activation = False
        if isinstance(activation, str) and (activation.casefold() in ('prelu','lrelu')):
            self.activation = activations.get(None)
            self.high_activation = activation.casefold()
            self.activity_config = activity_config
        elif activation is not None:
            self.use_plain_activation = True
            self.activation = activations.get(activation)
            self.activity_config = activity_config # dictionary passed to activation
        else:
            self.activation = activations.get(None)
            self.activity_config = None
        
        self.trainable = trainable
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.use_bias:
            bias_initializer = self.beta_initializer
            bias_regularizer = self.beta_regularizer
            bias_constraint = self.beta_constraint
        else:
            bias_initializer = None
            bias_regularizer = None
            bias_constraint = None
        self.layer_conv = Conv(rank = self.rank,
                          filters = self.filters,
                          kernel_size = self.kernel_size,
                          strides = self.strides,
                          padding = self.padding,
                          data_format = self.data_format,
                          dilation_rate = self.dilation_rate,
                          activation = None,
                          use_bias = self.use_bias,
                          bias_initializer = bias_initializer,
                          bias_regularizer = bias_regularizer,
                          bias_constraint = bias_constraint,
                          kernel_initializer = self.kernel_initializer,
                          kernel_regularizer = self.kernel_regularizer,
                          kernel_constraint = self.kernel_constraint,
                          trainable=self.trainable)
        self.layer_conv.build(input_shape)
        next_shape = self.layer_conv.compute_output_shape(input_shape)
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
            next_shape = self.layer_norm.compute_output_shape(next_shape)
        if self.high_activation == 'prelu':
            shared_axes = tuple(range(1,self.rank+1))
            self.layer_actv = PReLU(shared_axes=shared_axes)
            self.layer_actv.build(next_shape)
        elif self.high_activation == 'prelu':
            alpha = self.activity_config.get('alpha', 0.3)
            self.layer_actv = LeakyReLU(alpha=alpha)
            self.layer_actv.build(next_shape)
        super(_AConv, self).build(input_shape)

    def call(self, inputs):
        outputs = self.layer_conv(inputs)
        if self.normalization and (not self.use_bias):
            outputs = self.layer_norm(outputs)
        if self.high_activation in ('prelu', 'lrelu'):
            outputs = self.layer_actv(outputs)
        if self.use_plain_activation:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        next_shape = self.layer_conv.compute_output_shape(input_shape)
        if not self.use_bias:
            next_shape = self.layer_norm.compute_output_shape(next_shape)
        if self.high_activation in ('prelu', 'lrelu'):
            next_shape = self.layer_actv.compute_output_shape(next_shape)
        return next_shape
    
    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
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
            '_high_activation': self.high_activation
        }
        base_config = super(_AConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
class AConv1D(_AConv):
    """1D convolution layer (e.g. temporal convolution).
    This layer creates a convolution kernel that is convolved
    with the layer input over a single spatial (or temporal) dimension
    to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.
    When using this layer as the first layer in a model,
    provide an `input_shape` argument
    (tuple of integers or `None`, e.g.
    `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
    or `(None, 128)` for variable-length sequences of 128-dimensional vectors.
    The abstract architecture of AConv1D is:
    `output = activation( normalization( conv(x, W), gamma, beta ), alpha )`
    This layer is a stack of convolution, normalization and activation.
    As an extension, we allow users to use activating layers with parameters
    like PRelu.
    Arguments for convolution:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
            `"causal"` results in causal (dilated) convolutions, e.g. output[t]
            does not depend on input[t+1:]. Useful when modeling temporal data
            where the model should not violate the temporal order.
            See [WaveNet: A Generative Model for Raw Audio, section
            2.1](https://arxiv.org/abs/1609.03499).
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
    Arguments for activation:
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        activity_config: keywords for the parameters of activation
            function.
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
               filters,
               kernel_size,
               strides=1,
               padding='valid',
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
               activation=None,
               activity_config=None,
               activity_regularizer=None,
               **kwargs):
        super(AConv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
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
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)

    def call(self, inputs):
        if self.padding == 'causal':
            inputs = array_ops.pad(inputs, self._compute_causal_padding())
        return super(AConv1D, self).call(inputs)
        
class AConv2D(_AConv):
    """2D convolution layer (e.g. spatial convolution over images).
    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.
    The abstract architecture of AConv2D is:
    `output = activation( normalization( conv(x, W), gamma, beta ), alpha )`
    This layer is a stack of convolution, normalization and activation.
    As an extension, we allow users to use activating layers with parameters
    like PRelu.
    Arguments for convolution:
        filters: Integer, the dimensionality of the output space
          (i.e. the number of output filters in the convolution).
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
        padding: one of `"valid"` or `"same"` (case-insensitive).
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
    Arguments for activation:
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        activity_config: keywords for the parameters of activation
            function.
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
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
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
               activation=None,
               activity_config=None,
               activity_regularizer=None,
               **kwargs):
        super(AConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
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
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        
class AConv3D(_AConv):
    """3D convolution layer (e.g. spatial convolution over volumes).
    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 128, 1)` for 128x128x128 volumes
    with a single channel,
    in `data_format="channels_last"`.
    The abstract architecture of AConv3D is:
    `output = activation( normalization( conv(x, W), gamma, beta ), alpha )`
    This layer is a stack of convolution, normalization and activation.
    As an extension, we allow users to use activating layers with parameters
    like PRelu.
    Arguments for convolution:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
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
        padding: one of `"valid"` or `"same"` (case-insensitive).
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
    Arguments for activation:
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        activity_config: keywords for the parameters of activation
            function.
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
               filters,
               kernel_size,
               strides=(1, 1, 1),
               padding='valid',
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
               activation=None,
               activity_config=None,
               activity_regularizer=None,
               **kwargs):
        super(AConv3D, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
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
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
            
class _AConvTranspose(Layer):
    """Modern transposed convolution layer (sometimes called Deconvolution).
    Abstract nD transposed convolution layer (private, used as implementation base).
    `_AConvTranspose` implements the operation:
    `output = activation( normalization( convTranspose(x, W), gamma, beta ), alpha )`
    This layer is a stack of transposed convolution, normalization and activation.
    As an extension, we allow users to use activating layers with parameters
    like PRelu.
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
        padding: One of `"valid"`,  `"same"`.
        output_padding: An integer or tuple/list of n integers,
            specifying the amount of padding along the axes of the output tensor.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape is inferred.
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
            function.
    Arguments (others):
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
    References:
        - [A guide to convolution arithmetic for deep
            learning](https://arxiv.org/abs/1603.07285v1)
        - [Deconvolutional
            Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
    """

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 modenew=None,
                 strides=1,
                 padding='valid',
                 output_padding=None,
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
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
          kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(_AConvTranspose, self).__init__(trainable=trainable, name=name, activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        # Inherit from keras.layers._Conv
        self.rank = rank
        if modenew is not None:
            self.modenew = modenew
        else:
            self.modenew = _get_macro()
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        if (self.padding == 'causal' and not isinstance(self, AConv1D)):
            raise ValueError('Causal padding is only supported for `AConv1D`.')
        if output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(output_padding, rank, 'output_padding')
        else:
            self.output_padding = None
        self.data_format = conv_utils.normalize_data_format(data_format)
        if rank == 1 and self.data_format == 'channels_first':
            raise ValueError('Does not support channels_first data format for 1D case due to the limitation of upsampling method.')
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
        elif not normalization:
            self.use_bias = False
            self.gamma_initializer = None
            self.gamma_regularizer = None
            self.gamma_constraint = None
        else:
            self.use_bias = True
            self.gamma_initializer = None
            self.gamma_regularizer = None
            self.gamma_constraint = None
        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.groups = groups
        # Inherit from keras.engine.Layer
        self.high_activation = _high_activation
        self.use_plain_activation = False
        if isinstance(activation, str) and (activation.casefold() in ('prelu','lrelu')):
            self.activation = activations.get(None)
            self.high_activation = activation.casefold()
            self.activity_config = activity_config
        elif activation is not None:
            self.use_plain_activation = True
            self.activation = activations.get(activation)
            self.activity_config = activity_config # dictionary passed to activation
        else:
            self.activation = activations.get(None)
            self.activity_config = None
        
        self.trainable = trainable
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(self.rank + 2)
        if self.use_bias:
            bias_initializer = self.beta_initializer
            bias_regularizer = self.beta_regularizer
            bias_constraint = self.beta_constraint
        else:
            bias_initializer = None
            bias_regularizer = None
            bias_constraint = None
        if self.modenew:
            if self.rank == 1:
                self.layer_uppool = UpSampling1D(size=self.strides[0])
                self.layer_uppool.build(input_shape)
                next_shape = self.layer_uppool.compute_output_shape(input_shape)
                if self.output_padding is not None:
                    self.layer_padding = ZeroPadding1D(padding=self.output_padding[0])
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
            self.layer_conv = Conv(rank = self.rank,
                              filters = self.filters,
                              kernel_size = self.kernel_size,
                              strides = 1,
                              padding = self.padding,
                              data_format = self.data_format,
                              dilation_rate = self.dilation_rate,
                              activation = None,
                              use_bias = self.use_bias,
                              bias_initializer = bias_initializer,
                              bias_regularizer = bias_regularizer,
                              bias_constraint = bias_constraint,
                              kernel_initializer = self.kernel_initializer,
                              kernel_regularizer = self.kernel_regularizer,
                              kernel_constraint = self.kernel_constraint,
                              trainable=self.trainable)
            self.layer_conv.build(next_shape)
            next_shape = self.layer_conv.compute_output_shape(next_shape)
        else:
            if self.rank == 1:
                input_shape = input_shape[:1].concatenate([1,]).concatenate(input_shape[1:])
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
                                    use_bias = self.use_bias,
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
                                    use_bias = self.use_bias,
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
                                    dilation_rate = self.dilation_rate,
                                    activation = None,
                                    use_bias = self.use_bias,
                                    bias_initializer = bias_initializer,
                                    bias_regularizer = bias_regularizer,
                                    bias_constraint = bias_constraint,
                                    kernel_initializer = self.kernel_initializer,
                                    kernel_regularizer = self.kernel_regularizer,
                                    kernel_constraint = self.kernel_constraint,
                                    trainable=self.trainable)
            else:
                raise ValueError('Rank of the deconvolution should be 1, 2 or 3.')
            self.layer_deconv.build(input_shape)
            next_shape = self.layer_deconv.compute_output_shape(input_shape)
            if self.rank == 1:
                next_shape = next_shape[:1].concatenate(next_shape[2:])
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
            next_shape = self.layer_norm.compute_output_shape(next_shape)
        if self.high_activation == 'prelu':
            shared_axes = tuple(range(1,self.rank+1))
            self.layer_actv = PReLU(shared_axes=shared_axes)
            self.layer_actv.build(next_shape)
        elif self.high_activation == 'prelu':
            alpha = self.activity_config.get('alpha', 0.3)
            self.layer_actv = LeakyReLU(alpha=alpha)
            self.layer_actv.build(next_shape)
        super(_AConvTranspose, self).build(input_shape)

    def call(self, inputs):
        if self.modenew: # Apply new architecture
            outputs = self.layer_uppool(inputs)
            if self.layer_padding is not None:
                outputs = self.layer_padding(outputs)
            outputs = self.layer_conv(outputs)
        else: # Use classic method
            if self.rank == 1:
                inputs = array_ops.expand_dims(inputs, axis=1)
            outputs = self.layer_deconv(inputs)
            if self.rank == 1:
                outputs = array_ops.squeeze(outputs, axis=1)
        if self.normalization and (not self.use_bias):
            outputs = self.layer_norm(outputs)
        if self.high_activation in ('prelu', 'lrelu'):
            outputs = self.layer_actv(outputs)
        if self.use_plain_activation:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(self.rank + 2)
        if self.modenew: # Apply new architecture
            next_shape = self.layer_uppool.compute_output_shape(input_shape)
            if self.layer_padding is not None:
                next_shape = self.layer_padding.compute_output_shape(next_shape)
            next_shape = self.layer_conv.compute_output_shape(next_shape)
        else: # Use classic method
            if self.rank == 1:
                next_shape = input_shape[:1].concatenate([1,]).concatenate(input_shape[1:])
            next_shape = self.layer_conv.compute_output_shape(next_shape)
            if self.rank == 1:
                next_shape = next_shape[:1].concatenate(next_shape[2:])
        if not self.use_bias:
            next_shape = self.layer_norm.compute_output_shape(next_shape)
        if self.high_activation in ('prelu', 'lrelu'):
            next_shape = self.layer_actv.compute_output_shape(next_shape)
        return next_shape
    
    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'output_padding': self.output_padding,
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
            '_high_activation': self.high_activation
        }
        base_config = super(_AConvTranspose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
class AConv1DTranspose(_AConvTranspose):
    """Modern transposed convolution layer (sometimes called Deconvolution).
    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.
    The abstract architecture of AConv1DTranspose is:
    `output = activation( normalization( convTranspose(x, W), gamma, beta ), alpha )`
    This layer is a stack of transposed convolution, normalization and activation.
    As an extension, we allow users to use activating layers with parameters
    like PRelu.
    Arguments for convolution:
        filters: Integer, the dimensionality of the output space (i.e. the number
            of filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            length of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"`,  `"same"`.
        output_padding: An integer or tuple/list of n integers,
            specifying the amount of padding along the height and width
            of the output tensor.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape is inferred.
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
    Arguments for activation:
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        activity_config: keywords for the parameters of activation
            function.
    Arguments (others):
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
    Input shape:
        3D tensor with shape: `(batch_size, steps, input_dim)`
    Output shape:
        3D tensor with shape: `(batch_size, new_steps, filters)`
        `steps` value might have changed due to padding or strides.
    References:
        - [A guide to convolution arithmetic for deep
            learning](https://arxiv.org/abs/1603.07285v1)
        - [Deconvolutional
            Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 output_padding=None,
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
                 **kwargs):
        super(AConv1DTranspose, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
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
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
            
class AConv2DTranspose(_AConvTranspose):
    """Modern transposed convolution layer (sometimes called Deconvolution).
    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.
    The abstract architecture of AConv1DTranspose is:
    `output = activation( normalization( convTranspose(x, W), gamma, beta ), alpha )`
    This layer is a stack of transposed convolution, normalization and activation.
    As an extension, we allow users to use activating layers with parameters
    like PRelu.
    Arguments for convolution:
        filters: Integer, the dimensionality of the output space
          (i.e. the number of output filters in the convolution).
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
        padding: one of `"valid"` or `"same"` (case-insensitive).
        output_padding: An integer or tuple/list of 2 integers,
            specifying the amount of padding along the height and width
            of the output tensor.
            Can be a single integer to specify the same value for all
            spatial dimensions.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape is inferred.
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
    Arguments for activation:
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        activity_config: keywords for the parameters of activation
            function.
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
    References:
        - [A guide to convolution arithmetic for deep
            learning](https://arxiv.org/abs/1603.07285v1)
        - [Deconvolutional
            Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 output_padding=None,
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
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 **kwargs):
        super(AConv2DTranspose, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
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
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
            
class AConv3DTranspose(_AConvTranspose):
    """Modern transposed convolution layer (sometimes called Deconvolution).
    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 128, 3)` for a 128x128x128 volume with 3 channels
    if `data_format="channels_last"`.
    The abstract architecture of AConv1DTranspose is:
    `output = activation( normalization( convTranspose(x, W), gamma, beta ), alpha )`
    This layer is a stack of transposed convolution, normalization and activation.
    As an extension, we allow users to use activating layers with parameters
    like PRelu.
    Arguments for convolution:
        filters: Integer, the dimensionality of the output space
          (i.e. the number of output filters in the convolution).
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
        padding: one of `"valid"` or `"same"` (case-insensitive).
        output_padding: An integer or tuple/list of 3 integers,
            specifying the amount of padding along the depth, height, and
            width.
            Can be a single integer to specify the same value for all
            spatial dimensions.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape is inferred.
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
    Arguments for activation:
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        activity_config: keywords for the parameters of activation
            function.
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
    References:
        - [A guide to convolution arithmetic for deep
            learning](https://arxiv.org/abs/1603.07285v1)
        - [Deconvolutional
            Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 output_padding=None,
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
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 **kwargs):
        super(AConv3DTranspose, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
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
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)