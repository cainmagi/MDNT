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
# Version: 0.58 # 2019/6/11
# Comments:
#   Fix a bug for normalization layers inside AConv when
#   channels_first.
# Version: 0.55 # 2019/6/6
# Comments:
#   A failed try for quick group convolution (QGroupConv), move
#   it to deprecated.
# Version: 0.5 # 2019/6/6
# Comments:
#   Enable the advanced convolutional layers (AConv) to support
#   group convolution.
# Version: 0.4 # 2019/6/5
# Comments:
#   Add group convolutional layers (`GroupConv`).
# Version: 0.35 # 2019/5/28
# Comments:
#   1. Change the order of Cropping layer for AConvTranspose.
#   2. Fix the bug of dilation_rate for AConvTranspose.
# Version: 0.30 # 2019/5/22
# Comments:
#   Enhance the transposed convolution to enable it to infer
#   the padding/cropping policy from desired output shape.
# Version: 0.23 # 2019/3/30
# Comments:
#   Fix a bug when using lrelu without giving configs.
# Version: 0.22 # 2019/3/28
# Comments:
#   Enable the transposed convolution to control output-padding
#   in both directions.
# Version: 0.21 # 2019/3/27
# Comments:
#   Add compatible support and fix a bug about activation.
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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables

from tensorflow.keras.layers import BatchNormalization, LeakyReLU, PReLU
from tensorflow.python.keras.layers.convolutional import Conv, Conv2DTranspose, Conv3DTranspose, UpSampling1D, UpSampling2D, UpSampling3D, ZeroPadding1D, ZeroPadding2D, ZeroPadding3D, Cropping1D, Cropping2D, Cropping3D
from .normalize import InstanceNormalization, GroupNormalization

from .. import compat
if compat.COMPATIBLE_MODE:
    from tensorflow.python.keras.engine.base_layer import InputSpec
else:
    from tensorflow.python.keras.engine.input_spec import InputSpec

NEW_CONV_TRANSPOSE = True

def _get_macro_conv():
    return NEW_CONV_TRANSPOSE

_check_dl_func = lambda a: all(ai==1 for ai in a)

class Conv1DTied(Conv2DTranspose):
    """Tied convolution layer (sometimes called Deconvolution).
    Compared to `Conv1DTranspose`, this implementation requires a `Conv1D`
    layer to provide kernel which would be used as the kernel for transpo-
    sed convolution. As a result, this implementation is a symmetric layer
    for the provided layer.
    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 3)` for 128 length vector
    in `data_format="channels_last"`.
    NOTE THAT ALTHOUGH WE HAVE SUCCESSED TO MAKE THIS LAYER SERIALIZABLE,
    IT MAY BE STILL PROBLEMATIC FOR TRAINING ALGORITHM. PLEASE BE CAREFUL
    WHEN USING SUCH KIND OF LAYERS.
    IN MULTIPLE MODELS, THIS INSTANCE MAY CAUSING CONFLICTS BECAUSE IT
    USES GLOBAL VARIABLE NAME TO SERIALIZE CROSSED LAYERS. IT IS
    RECOMMENDED TO SEPARATE NAME SCOPES WHEN USING MULTIPLE MODELS.
    Arguments:
        padding: one of `"valid"` or `"same"` (case-insensitive).
        output_padding: An integer or tuple/list of 1 integers,
            specifying the amount of padding along the height and width
            of the output tensor.
            Can be a single integer to specify the same value for all
            spatial dimensions.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape is inferred.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        bias_initializer: Initializer for the bias vector.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        bias_constraint: Constraint function applied to the bias vector.
    Reserved arguments:
        varName, filters, kernel_size, strides, data_format, 
            dilation_rate: only used when saving and restoring the layer.
    Input shape:
        3D tensor with shape:
        `(batch, channels, steps)` if data_format='channels_first'
        or 3D tensor with shape:
        `(batch, steps, channels)` if data_format='channels_last'.
    Output shape:
        3D tensor with shape:
        `(batch, filters, new_steps)` if data_format='channels_first'
        or 3D tensor with shape:
        `(batch, new_steps, filters)` if data_format='channels_last'.
        `new_steps` values might have changed due to padding.
    """

    def __init__(self,
                 tied_layer='',
                 padding='valid',
                 output_padding=None,
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 activity_regularizer=None,
                 bias_constraint=None,
                 varName='',
                 filters=None,
                 kernel_size=1,
                 strides=1,
                 data_format=None,
                 dilation_rate=1,
                 **kwargs):
        # Reserved variables
        if tied_layer != '':
            self.kernelFrom = tied_layer.kernel.name
            data_format = tied_layer.data_format
            strides = (1, *tied_layer.strides)
            dilation_rate = (1, *tied_layer.dilation_rate)
        super(Conv2DTranspose, self).__init__(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                activation=activations.get(activation),
                use_bias=use_bias,
                bias_initializer=initializers.get(bias_initializer),
                bias_regularizer=regularizers.get(bias_regularizer),
                activity_regularizer=regularizers.get(activity_regularizer),
                bias_constraint=constraints.get(bias_constraint),
                **kwargs)

        self.output_padding = output_padding
        if output_padding is not None:
            if isinstance(output_padding, (list, tuple)) and len(output_padding)==2:
                self.output_padding = output_padding
            else:    
                output_padding = conv_utils.normalize_tuple(output_padding, 1, 'output_padding')
                self.output_padding = (1, *output_padding)
        self.varName = varName
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 3:
            raise ValueError('Inputs should have rank 3. Received input shape: ' + str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
            self.get_channels_first = True
        else:
            channel_axis = -1
            self.get_channels_first = False
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
        
        if self.varName == '':
            kernelFrom = list(filter(lambda x:x.name==self.kernelFrom, [op for op in variables.global_variables(scope=None)]))[0]
        else:
            kernelFrom = list(filter(lambda x:x.name==self.varName, [op for op in variables.global_variables(scope=None)]))[0]
        self.kernel = K.expand_dims(kernelFrom, axis=0)
        # Save/Load information from tied layer (or database).
        if self.varName == '':
            kernel_shape = self.kernel.get_shape().as_list()
            self.varName = kernelFrom.name
            self.kernel_size = kernel_shape[:2]
            self.filters = kernel_shape[2]
        if self.output_padding is not None:
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) + ' must be '
                                     'greater than output padding ' +
                                     str(self.output_padding))
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        print(inputs)
        if self.get_channels_first:
            r2_inputs = K.expand_dims(inputs, axis=2)
            get_r2_out = super(Conv1DTied, self).call(r2_inputs)
            return K.squeeze(get_r2_out, axis=2)
        else:
            r2_inputs = K.expand_dims(inputs, axis=1)
            get_r2_out = super(Conv1DTied, self).call(r2_inputs)
            return K.squeeze(get_r2_out, axis=1)

    def get_config(self):
        config = super(Conv1DTied, self).get_config()
        config['varName'] = self.varName
        config['tied_layer'] = ''
        return config

class Conv2DTied(Conv2DTranspose):
    """Tied convolution layer (sometimes called Deconvolution).
    Compared to `Conv2DTranspose`, this implementation requires a `Conv2D`
    layer to provide kernel which would be used as the kernel for transpo-
    sed convolution. As a result, this implementation is a symmetric layer
    for the provided layer.
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
    NOTE THAT ALTHOUGH WE HAVE SUCCESSED TO MAKE THIS LAYER SERIALIZABLE,
    IT MAY BE STILL PROBLEMATIC FOR TRAINING ALGORITHM. PLEASE BE CAREFUL
    WHEN USING SUCH KIND OF LAYERS.
    IN MULTIPLE MODELS, THIS INSTANCE MAY CAUSING CONFLICTS BECAUSE IT
    USES GLOBAL VARIABLE NAME TO SERIALIZE CROSSED LAYERS. IT IS
    RECOMMENDED TO SEPARATE NAME SCOPES WHEN USING MULTIPLE MODELS.
    Arguments:
        padding: one of `"valid"` or `"same"` (case-insensitive).
        output_padding: An integer or tuple/list of 2 integers,
            specifying the amount of padding along the height and width
            of the output tensor.
            Can be a single integer to specify the same value for all
            spatial dimensions.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape is inferred.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        bias_initializer: Initializer for the bias vector.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        bias_constraint: Constraint function applied to the bias vector.
    Reserved arguments:
        varName, filters, kernel_size, strides, data_format, 
            dilation_rate: only used when saving and restoring the layer.
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

    def __init__(self,
                 tied_layer='',
                 padding='valid',
                 output_padding=None,
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 activity_regularizer=None,
                 bias_constraint=None,
                 varName='',
                 filters=None,
                 kernel_size=1,
                 strides=1,
                 data_format=None,
                 dilation_rate=1,
                 **kwargs):
        # Reserved variables
        if tied_layer != '':
            self.kernelFrom = tied_layer.kernel.name
            data_format = tied_layer.data_format
            strides = tied_layer.strides
            dilation_rate = tied_layer.dilation_rate
        super(Conv2DTranspose, self).__init__(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                activation=activations.get(activation),
                use_bias=use_bias,
                bias_initializer=initializers.get(bias_initializer),
                bias_regularizer=regularizers.get(bias_regularizer),
                activity_regularizer=regularizers.get(activity_regularizer),
                bias_constraint=constraints.get(bias_constraint),
                **kwargs)

        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(self.output_padding, 2, 'output_padding')
        self.varName = varName

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank 4. Received input shape: ' + str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        
        if self.varName == '':
            kernelFrom = list(filter(lambda x:x.name==self.kernelFrom, [op for op in variables.global_variables(scope=None)]))[0]
        else:
            kernelFrom = list(filter(lambda x:x.name==self.varName, [op for op in variables.global_variables(scope=None)]))[0]
        self.kernel = kernelFrom
        # Save/Load information from tied layer (or database).
        if self.varName == '':
            kernel_shape = self.kernel.get_shape().as_list()
            self.varName = kernelFrom.name
            self.kernel_size = kernel_shape[:2]
            self.filters = kernel_shape[2]
        if self.output_padding is not None:
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) + ' must be '
                                     'greater than output padding ' +
                                     str(self.output_padding))
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        self.built = True

    def get_config(self):
        config = super(Conv2DTied, self).get_config()
        config['varName'] = self.varName
        config['tied_layer'] = ''
        return config

class Conv3DTied(Conv3DTranspose):
    """Tied convolution layer (sometimes called Deconvolution).
    Compared to `Conv3DTranspose`, this implementation requires a `Conv3D`
    layer to provide kernel which would be used as the kernel for transpo-
    sed convolution. As a result, this implementation is a symmetric layer
    for the provided layer.
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
    NOTE THAT ALTHOUGH WE HAVE SUCCESSED TO MAKE THIS LAYER SERIALIZABLE,
    IT MAY BE STILL PROBLEMATIC FOR TRAINING ALGORITHM. PLEASE BE CAREFUL
    WHEN USING SUCH KIND OF LAYERS.
    IN MULTIPLE MODELS, THIS INSTANCE MAY CAUSING CONFLICTS BECAUSE IT
    USES GLOBAL VARIABLE NAME TO SERIALIZE CROSSED LAYERS. IT IS
    RECOMMENDED TO SEPARATE NAME SCOPES WHEN USING MULTIPLE MODELS.
    Arguments:
        padding: one of `"valid"` or `"same"` (case-insensitive).
        output_padding: An integer or tuple/list of 3 integers,
            specifying the amount of padding along the depth, height, and
            width.
            Can be a single integer to specify the same value for all
            spatial dimensions.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape is inferred.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    Reserved arguments:
        varName, filters, kernel_size, strides, data_format:
            only used when saving and restoring the layer.
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

    def __init__(self,
                 tied_layer='',
                 padding='valid',
                 output_padding=None,
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 activity_regularizer=None,
                 bias_constraint=None,
                 varName='',
                 filters=None,
                 kernel_size=1,
                 strides=1,
                 data_format=None,
                 **kwargs):
        # Reserved variables
        if tied_layer != '':
            self.kernelFrom = tied_layer.kernel.name
            data_format = tied_layer.data_format
            strides = tied_layer.strides
        super(Conv3DTranspose, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(self.output_padding, 3, 'output_padding')
        self.varName = varName

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 5:
            raise ValueError('Inputs should have rank 5, received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined, found None: ' + str(input_shape))
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=5, axes={channel_axis: input_dim})

        if self.varName == '':
            kernelFrom = list(filter(lambda x:x.name==self.kernelFrom, [op for op in variables.global_variables(scope=None)]))[0]
        else:
            kernelFrom = list(filter(lambda x:x.name==self.varName, [op for op in variables.global_variables(scope=None)]))[0]
        self.kernel = kernelFrom
        # Save/Load information from tied layer (or database).
        if self.varName == '':
            kernel_shape = self.kernel.get_shape().as_list()
            self.varName = kernelFrom.name
            self.kernel_size = kernel_shape[:2]
            self.filters = kernel_shape[2]
        if self.output_padding is not None:
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) + ' must be '
                                     'greater than output padding ' +
                                      str(self.output_padding))
        if self.use_bias:
            self.bias = self.add_weight(
                    'bias',
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    trainable=True,
                    dtype=self.dtype)
        else:
            self.bias = None
        self.built = True

    def get_config(self):
        config = super(Conv2DTied, self).get_config()
        config['varName'] = self.varName
        config['tied_layer'] = ''
        return config

class _GroupConv(Layer):
    """Abstract nD group convolution layer (private, used as implementation base).
    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.
    Different from trivial `Conv` layers, `GroupConv` divide the input channels into
    several groups, and apply trivial convolution (or called dense convolution) to
    each group. Inside each group, the convolution is trivial, however, between each
    two groups, the convolutions are independent.
    Arguments:
        rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
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

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        if input_dim % self.lgroups != 0:
            raise ValueError('To grouplize the input channels, the input channel number should be a multiple of group number (N*{0}), but given {1}'.format(self.lgroups, input_dim))
        self.group_input_dim = input_dim // self.lgroups
        kernel_shape = self.kernel_size + (self.group_input_dim, self.lfilters * self.lgroups)

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
            op_padding = 'valid'
        else:
            op_padding = self.padding
        # Create conv. op groups.
        if self.data_format == 'channels_first':
            group_input_shape = tensor_shape.TensorShape([input_shape[0], self.group_input_dim, *input_shape[2:]])
        else:
            group_input_shape = tensor_shape.TensorShape([*input_shape[:-1], self.group_input_dim])
        group_kernel_shape = tensor_shape.TensorShape([*kernel_shape[:-1], self.lfilters])
        self._convolution_op = nn_ops.Convolution(
                group_input_shape,
                filter_shape=group_kernel_shape,
                dilation_rate=self.dilation_rate,
                strides=self.strides,
                padding=op_padding.upper(),
                data_format=conv_utils.convert_data_format(self.data_format, self.rank + 2))
        self.built = True

    def call(self, inputs):
        outputs_list = []
        if self.data_format == 'channels_first':
            for i in range(self.lgroups):
                get_output = self._convolution_op(inputs[:,i*self.group_input_dim:(i+1)*self.group_input_dim, ...], self.kernel[..., i*self.lfilters:(i+1)*self.lfilters])
                outputs_list.append(get_output)
            outputs = array_ops.concat(outputs_list, 1)
        else:
            for i in range(self.lgroups):
                get_output = self._convolution_op(inputs[..., i*self.group_input_dim:(i+1)*self.group_input_dim], self.kernel[..., i*self.lfilters:(i+1)*self.lfilters])
                outputs_list.append(get_output)
            outputs = array_ops.concat(outputs_list, -1)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.lfilters * self.lgroups, 1))
                    outputs += bias
                if self.rank == 2:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
                if self.rank == 3:
                    # As of Mar 2017, direct addition is significantly slower than
                    # bias_add when computing gradients. To use bias_add, we collapse Z
                    # and Y into a single dimension to obtain a 4D input tensor.
                    outputs_shape = outputs.shape.as_list()
                    if outputs_shape[0] is None:
                        outputs_shape[0] = -1
                    outputs_4d = array_ops.reshape(outputs,
                                                   [outputs_shape[0], outputs_shape[1],
                                                   outputs_shape[2] * outputs_shape[3],
                                                   outputs_shape[4]])
                    outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
                    outputs = array_ops.reshape(outputs_4d, outputs_shape)
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                        space[i],
                        self.kernel_size[i],
                        padding=self.padding,
                        stride=self.strides[i],
                        dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0]] + new_space + [self.lfilters * self.lgroups])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                        space[i],
                        self.kernel_size[i],
                        padding=self.padding,
                        stride=self.strides[i],
                        dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0], self.lfilters * self.lgroups] + new_space)

    def get_config(self):
        config = {
            'lgroups': self.lgroups,
            'lfilters': self.lfilters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                    regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(_GroupConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _compute_causal_padding(self):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if self.data_format == 'channels_last':
            causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
        return causal_padding

class GroupConv1D(_GroupConv):
    """1D group convolution layer (e.g. temporal group convolution).
    This layer creates a convolution kernel that is convolved
    with the layer input over a single spatial (or temporal) dimension
    to produce a tensor of outputs.
    Different from trivial `Conv` layers, `GroupConv` divide the input 
    channels into several groups, and apply trivial convolution (or called 
    dense convolution) to each group. Inside each group, the convolution is
    trivial, however, between each two groups, the convolutions are independent.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.
    When using this layer as the first layer in a model,
    provide an `input_shape` argument
    (tuple of integers or `None`, e.g.
    `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
    or `(None, 128)` for variable-length sequences of 128-dimensional vectors.
    Arguments:
        lgroups: Integer, the group number of the latent convolution branch. The
            number of filters in the whole latent space is lgroups * lfilters.
        lfilters: Integer, the dimensionality in each the lattent group (i.e. the
            number of filters in each latent convolution branch).
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
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.
    Input shape:
        3D tensor with shape: `(batch_size, steps, input_dim)`
    Output shape:
        3D tensor with shape: `(batch_size, new_steps, filters)`
        `steps` value might have changed due to padding or strides.
    """

    def __init__(self,
                 lgroups,
                 lfilters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format='channels_last',
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
                 **kwargs):
        super(GroupConv1D, self).__init__(
            rank=1,
            lgroups=lgroups,
            lfilters=lfilters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

    def call(self, inputs):
        if self.padding == 'causal':
            inputs = array_ops.pad(inputs, self._compute_causal_padding())
        return super(GroupConv1D, self).call(inputs)

class GroupConv2D(_GroupConv):
    """2D group convolution layer (e.g. spatial group convolution over images).
    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of outputs.
    Different from trivial `Conv` layers, `GroupConv` divide the input 
    channels into several groups, and apply trivial convolution (or called 
    dense convolution) to each group. Inside each group, the convolution is
    trivial, however, between each two groups, the convolutions are independent. 
    If `use_bias` is True, a bias vector is created and added to the outputs. 
    Finally, if `activation` is not `None`, it is applied to the outputs as well.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.
    Arguments:
        lgroups: Integer, the group number of the latent convolution branch. The
            number of filters in the whole latent space is lgroups * lfilters.
        lfilters: Integer, the dimensionality in each the lattent group (i.e. the
            number of filters in each latent convolution branch).
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
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.
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
                 lgroups,
                 lfilters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GroupConv2D, self).__init__(
            rank=2,
            lgroups=lgroups,
            lfilters=lfilters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

class GroupConv3D(_GroupConv):
    """3D group convolution layer (e.g. spatial group convolution over volumes).
    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of outputs.
    Different from trivial `Conv` layers, `GroupConv` divide the input 
    channels into several groups, and apply trivial convolution (or called 
    dense convolution) to each group. Inside each group, the convolution is
    trivial, however, between each two groups, the convolutions are independent.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 128, 1)` for 128x128x128 volumes
    with a single channel,
    in `data_format="channels_last"`.
    Arguments:
        lgroups: Integer, the group number of the latent convolution branch. The
            number of filters in the whole latent space is lgroups * lfilters.
        lfilters: Integer, the dimensionality in each the lattent group (i.e. the
            number of filters in each latent convolution branch).
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
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.
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
                 lgroups,
                 lfilters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GroupConv3D, self).__init__(
            rank=3,
            lgroups=lgroups,
            lfilters=lfilters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

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
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(_AConv, self).__init__(trainable=trainable, name=name, activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
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
            self.use_bias = True
            self.gamma_initializer = None
            self.gamma_regularizer = None
            self.gamma_constraint = None
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
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if self.use_bias:
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
            self.layer_conv = Conv(rank=self.rank,
                                   filters=self.filters,
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
        self.layer_conv.build(input_shape)
        if compat.COMPATIBLE_MODE: # for compatibility
            self._trainable_weights.extend(self.layer_conv._trainable_weights)
        next_shape = self.layer_conv.compute_output_shape(input_shape)
        if self.normalization and (not self.use_bias):
            if self.normalization.casefold() == 'batch':
                self.layer_norm = BatchNormalization(axis=channel_axis,
                                                     gamma_initializer=self.gamma_initializer,
                                                     gamma_regularizer=self.gamma_regularizer,
                                                     gamma_constraint=self.gamma_constraint,
                                                     beta_initializer=self.beta_initializer,
                                                     beta_regularizer=self.beta_regularizer,
                                                     beta_constraint=self.beta_constraint,
                                                     trainable=self.trainable)
            elif self.normalization.casefold() == 'inst':
                self.layer_norm = InstanceNormalization(axis=channel_axis,
                                                        gamma_initializer=self.gamma_initializer,
                                                        gamma_regularizer=self.gamma_regularizer,
                                                        gamma_constraint=self.gamma_constraint,
                                                        beta_initializer=self.beta_initializer,
                                                        beta_regularizer=self.beta_regularizer,
                                                        beta_constraint=self.beta_constraint,
                                                        trainable=self.trainable)
            elif self.normalization.casefold() == 'group':
                self.layer_norm = GroupNormalization(axis=channel_axis, groups=self.groups,
                                                     gamma_initializer=self.gamma_initializer,
                                                     gamma_regularizer=self.gamma_regularizer,
                                                     gamma_constraint=self.gamma_constraint,
                                                     beta_initializer=self.beta_initializer,
                                                     beta_regularizer=self.beta_regularizer,
                                                     beta_constraint=self.beta_constraint,
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
        lgroups: Latent group number of group convolution. Only if set, use group
            convolution. The latent filter number of group convolution would
            be inferred by lfilters = filters // lgroups. Hence, filters should
            be a multiple of lgroups.
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
               filters,
               kernel_size,
               strides=1,
               lgroups=None,
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
            lgroups=lgroups,
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
        lgroups: Latent group number of group convolution. Only if set, use group
            convolution. The latent filter number of group convolution would
            be inferred by lfilters = filters // lgroups. Hence, filters should
            be a multiple of lgroups.
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
               filters,
               kernel_size,
               strides=(1, 1),
               lgroups=None,
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
            lgroups=lgroups,
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
        lgroups: Latent group number of group convolution. Only if set, use group
            convolution. The latent filter number of group convolution would
            be inferred by lfilters = filters // lgroups. Hence, filters should
            be a multiple of lgroups.
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
               filters,
               kernel_size,
               strides=(1, 1, 1),
               lgroups=None,
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
            lgroups=lgroups,
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
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
          kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(_AConvTranspose, self).__init__(trainable=trainable, name=name, activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
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
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if self.use_bias:
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
                self.layer_conv = Conv(rank=self.rank,
                                       filters=self.filters,
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
            if compat.COMPATIBLE_MODE: # for compatibility
                self._trainable_weights.extend(self.layer_deconv._trainable_weights)
            next_shape = self.layer_deconv.compute_output_shape(input_shape)
            if self.rank == 1:
                next_shape = next_shape[:1].concatenate(next_shape[2:])
        if self.normalization and (not self.use_bias):
            if self.normalization.casefold() == 'batch':
                self.layer_norm = BatchNormalization(axis=channel_axis,
                                                     gamma_initializer = self.gamma_initializer,
                                                     gamma_regularizer = self.gamma_regularizer,
                                                     gamma_constraint = self.gamma_constraint,
                                                     beta_initializer = self.beta_initializer,
                                                     beta_regularizer = self.beta_regularizer,
                                                     beta_constraint = self.beta_constraint,
                                                     trainable=self.trainable)
            elif self.normalization.casefold() == 'inst':
                self.layer_norm = InstanceNormalization(axis=channel_axis,
                                                     gamma_initializer = self.gamma_initializer,
                                                     gamma_regularizer = self.gamma_regularizer,
                                                     gamma_constraint = self.gamma_constraint,
                                                     beta_initializer = self.beta_initializer,
                                                     beta_regularizer = self.beta_regularizer,
                                                     beta_constraint = self.beta_constraint,
                                                     trainable=self.trainable)
            elif self.normalization.casefold() == 'group':
                self.layer_norm = GroupNormalization(axis=channel_axis, groups=self.groups,
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
        super(_AConvTranspose, self).build(input_shape)

    def call(self, inputs):
        if self.modenew: # Apply new architecture
            outputs = self.layer_uppool(inputs)
            if self.layer_padding is not None:
                outputs = self.layer_padding(outputs)
            outputs = self.layer_conv(outputs)
            if self.layer_cropping is not None:
                outputs = self.layer_cropping(outputs)
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
            if self.layer_cropping is not None:
                next_shape = self.layer_cropping.compute_output_shape(next_shape)
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
    References:
        - [A guide to convolution arithmetic for deep
            learning](https://arxiv.org/abs/1603.07285v1)
        - [Deconvolutional
            Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 lgroups=None,
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
                 **kwargs):
        super(AConv1DTranspose, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            lgroups=lgroups,
            padding=padding,
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
        lgroups: Latent group number of group convolution. Only if set, use group
            convolution. The latent filter number of group convolution would
            be inferred by lfilters = filters // lgroups. Hence, filters should
            be a multiple of lgroups.
        padding: one of `"valid"` or `"same"` (case-insensitive).
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
    References:
        - [A guide to convolution arithmetic for deep
            learning](https://arxiv.org/abs/1603.07285v1)
        - [Deconvolutional
            Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 lgroups=None,
                 padding='valid',
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
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 **kwargs):
        super(AConv2DTranspose, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            lgroups=lgroups,
            padding=padding,
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
        lgroups: Latent group number of group convolution. Only if set, use group
            convolution. The latent filter number of group convolution would
            be inferred by lfilters = filters // lgroups. Hence, filters should
            be a multiple of lgroups.
        padding: one of `"valid"` or `"same"` (case-insensitive).
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
    References:
        - [A guide to convolution arithmetic for deep
            learning](https://arxiv.org/abs/1603.07285v1)
        - [Deconvolutional
            Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 lgroups=None,
                 padding='valid',
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
                 activation=None,
                 activity_config=None,
                 activity_regularizer=None,
                 **kwargs):
        super(AConv3DTranspose, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            lgroups=lgroups,
            padding=padding,
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
            activation=activation,
            activity_config=activity_config,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
