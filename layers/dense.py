'''
################################################################
# Layers - Dense (deprecated)
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Extend the dense layer API with tied version.
# Version: 0.15 # 2019/6/24
# Comments:
# 1. Add the Ghost layer for implementing trainable input layer.
# 2. Fix a small bug for Ghost.
# Version: 0.11 # 2019/3/27
# Comments:
#   Add compatible support.
# Version: 0.10 # 2019/3/23
# Comments:
#   Create this submodule.
################################################################
'''

from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variables

from .. import compat
if compat.COMPATIBLE_MODE:
    from tensorflow.python.keras.engine.base_layer import InputSpec
else:
    from tensorflow.python.keras.engine.input_spec import InputSpec

class Ghost(Layer):
    """Ghost layer for setting tunable input
    Since tf-Keras does not support users to build a trainable input layer, we use
    an interesting trick, i.e. "Ghost" to realize the trainable input. Our Ghost
    layer is implemented like this:
        ouput = kernel * input + bias
    where both kernel and bias share the same shape of input tensor.
    There are two ways to build a tunable input layer. The first way is using
    kernel solely:
        input = Input(shape=shape) # feeding constant 1.0
        tunable_input = Ghost(use_kernel=True)(input) = kernel * 1.0 = kernel
    The second way is using bias solely:
        input = Input(shape=shape) # feeding constant 0.0
        tunable_input = Ghost(use_bias=True)(input) = bias + 0.0 = bias
    Because both kernel and bias are trainable, such a technique enables tf-Keras
    users to create a tunable input layer easily.
    It is not allowed to use kernel and bias in the same time, because in this
    case the solution for Ghost layer would become ill-posed.
    Arguments:
        use_kernel: Boolean, whether the layer uses multiplicative strategy to
            define the variable.
        use_bias: Boolean, whether the layer uses additive strategy to define
            the variable.
        var_initializer: Initializer for the tunable variable. The variable
            depends on setting use_kernel or setting use_bias.
        var_regularizer: Regularizer function applied to the tunable variable.
        var_constraint: Constraint function applied to the tunable variable.
    Input shape:
        Any shape. The shape should be totally known except the batch number.
    Output shape:
        The same as input shape.
    """
    def __init__(self,
                 use_kernel=False,
                 use_bias=False,
                 var_initializer='glorot_uniform',
                 var_regularizer=None,
                 var_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
          kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(Ghost, self).__init__(
            activity_regularizer=None, **kwargs)
        if not (use_kernel or use_bias):
            raise ValueError('Need to specify either "use_kernel" or "use_bias".')
        if use_kernel and use_bias:
            raise ValueError('Should not specify "use_kernel" and "use_bias" in the same time.')
        self.use_kernel = use_kernel
        self.use_bias = use_bias
        self.var_initializer = initializers.get(var_initializer)
        self.var_regularizer = regularizers.get(var_regularizer)
        self.var_constraint = constraints.get(var_constraint)
        self.supports_masking = True

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        for i in range(1, len(input_shape)):
            if tensor_shape.dimension_value(input_shape[i]) is None:
                raise ValueError('The input shape [1:] should be defined, but found element `None`.')
        if self.use_kernel:
            varName = 'kernel'
        elif self.use_bias:
            varName = 'bias'
        get_in = input_shape.as_list()[1:]
        self.get_var = self.add_weight(
            varName,
            shape=get_in,
            initializer=self.var_initializer,
            regularizer=self.var_regularizer,
            constraint=self.var_constraint,
            dtype=self.dtype,
            trainable=True)
        super(Ghost, self).build(input_shape)

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs)
        input_shape = K.int_shape(inputs)
        broadcast_shape = [1] + list(input_shape[1:])
        broadcast_var = K.reshape(self.get_var, broadcast_shape)
        if self.use_kernel:
            return broadcast_var * inputs
        elif self.use_bias:
            return broadcast_var + inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'use_kernel': self.use_kernel,
            'use_bias': self.use_bias,
            'var_initializer': initializers.serialize(self.var_initializer),
            'var_regularizer': regularizers.serialize(self.var_regularizer),
            'var_constraint': regularizers.serialize(self.var_constraint)
        }
        base_config = super(Ghost, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DenseTied(Layer):
    """Tied densely-connected NN layer.
    `DenseTied` implements the operation:
    `output = activation(dot(input, kernel.T) + bias)`
    where kernel comes from another Dense layer.
    NOTE THAT ALTHOUGH WE HAVE SUCCESSED TO MAKE THIS LAYER SERIALIZABLE,
    IT MAY BE STILL PROBLEMATIC FOR TRAINING ALGORITHM. PLEASE BE CAREFUL
    WHEN USING SUCH KIND OF LAYERS.
    IN MULTIPLE MODELS, THIS INSTANCE MAY CAUSING CONFLICTS BECAUSE IT
    USES GLOBAL VARIABLE NAME TO SERIALIZE CROSSED LAYERS. IT IS
    RECOMMENDED TO SEPARATE NAME SCOPES WHEN USING MULTIPLE MODELS.
    Arguments:
        tied_layer: A Dense layer instance where this layer is tied.
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
        varName, varShape: only used when saving and restoring the layer.
    Input shape:
        nD tensor with shape: `(batch_size, ..., output_dim_of_tied_layer)`.
        The most common situation would be
        a 2D input with shape `(batch_size, output_dim_of_tied_layer)`.
    Output shape:
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape 
            `(batch_size, input_dim_of_tied_layer)`,
        the output would have shape `(batch_size, input_dim_of_tied_layer)`.
    """
    def __init__(self,
                 tied_layer='',
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 activity_regularizer=None,
                 bias_constraint=None,
                 varName='', varShape=[],
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
          kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(DenseTied, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        if tied_layer != '':
            self.kernelFrom = tied_layer.kernel.name
        self.varName = varName
        self.varShape = varShape
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
          raise ValueError('The last dimension of the inputs to `Dense` '
                           'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: last_dim})
        if self.varName == '':
            kernelFrom = list(filter(lambda x:x.name==self.kernelFrom, [op for op in variables.global_variables(scope=None)]))[0]
            self.kernel = K.transpose(kernelFrom)
            self.o_shape = self.kernel.get_shape().as_list()
            self.varName = kernelFrom.name
            self.varShape = kernelFrom.get_shape().as_list()
        else:
            kernelFrom = list(filter(lambda x:x.name==self.varName, [op for op in variables.global_variables(scope=None)]))[0]
            self.kernel = K.transpose(kernelFrom)
            self.o_shape = self.kernel.get_shape().as_list()
        if self.use_bias:
          self.bias = self.add_weight(
              'bias',
              shape=[self.o_shape[-1],],
              initializer=self.bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint,
              dtype=self.dtype,
              trainable=True)
        else:
          self.bias = None
        self.built = True

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs)
        rank = common_shapes.rank(inputs)
        if rank > 2:
          # Broadcasting is required for the inputs.
          outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
          # Reshape the output back to the original ndim of the input.
          if not context.executing_eagerly():
            shape = inputs.get_shape().as_list()
            output_shape = shape[:-1] + [self.o_shape]
            outputs.set_shape(output_shape)
        else:
          outputs = gen_math_ops.mat_mul(inputs, self.kernel)
        if self.use_bias:
          outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
          return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
          raise ValueError(
              'The innermost dimension of input_shape must be defined, but saw: %s'
              % input_shape)
        return input_shape[:-1].concatenate(self.o_shape)
    
    def get_config(self):
        config = {
            'tied_layer': '',
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'varName': self.varName, 'varShape': self.varShape
        }
        base_config = super(DenseTied, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))