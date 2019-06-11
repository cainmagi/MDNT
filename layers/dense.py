'''
################################################################
# Layers - Dense (deprecated)
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Extend the dense layer API with tied version.
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