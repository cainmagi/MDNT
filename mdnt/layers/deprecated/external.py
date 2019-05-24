'''
################################################################
# Layers - External API layer (deprecated)
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# An abstract utility for introducing the outside API into the
# tf-keras architecture.
# Version: 0.20 # 2019/5/23
# Comments:
#   Add class 'External' to this submodule.
# Version: 0.10 # 2019/3/23
# Comments:
#   Create this submodule.
################################################################
'''

import uuid
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variables
from tensorflow.python.keras.utils import tf_utils

from .. import compat
if compat.COMPATIBLE_MODE:
    from tensorflow.python.keras.engine.base_layer import InputSpec
else:
    from tensorflow.python.keras.engine.input_spec import InputSpec

def dtype_serialize(input_dtypes):
    if isinstance(input_dtypes, list):
        return [dtypes.as_dtype(get_dt).as_datatype_enum() for get_dt in input_dtypes]
    else:
        return [dtypes.as_dtype(input_dtypes).as_datatype_enum()]

def dtype_get(input_serials):
    if isinstance(input_serials, list):
        return [dtypes.as_dtype(get_dt) for get_dt in input_serials]
    else:
        return [dtypes.as_dtype(input_serials)]

class External(Layer):
    """External API layer.
    `External` is used to introduce a non-parameter function from an 
    external library and enable it to participate the learning workflow.
    Therefore, this layer is requires users to provide:
        1. The forward propagation function `forward()`.
        2. The back propagation function `backward()`.
    Arguments:
        forward:      the forward propagating function.
        backward:     the back propagation function.
        Tin:          a list of input tf.DType.
        Tout:         a list of output tf.DType.
        stateful:     a bool flag used to define whether the forward/backward
                      function is calculated based on previous calculation.
        output_shape: a tf.TensorShape, a tuple/list or a function. It is
                      used for estimating the output shape fast.
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
                 forward,
                 backward,
                 Tin,
                 Tout,
                 output_shape=None,
                 id=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
          kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(External, self).__init__(**kwargs)
        self.forward = activations.get(forward)
        self.backward = activations.get(backward)
        self.Tin = dtype_get(Tin)
        self.Tout = dtype_get(Tout)

        if id is None: # id is used to tag the newly created instance
            self._id = 'PyExternal' + str(uuid.uuid4())
        else:
            self._id = id

        if output_shape is None:
            raise NotImplementedError('We could not automatically infer '
                                      'the static shape of the External\'s output.'
                                      ' Please specify the `output_shape` for'
                                      ' this External.')
        else:
            self._output_shape = activations.get(output_shape)

        self.supports_masking = True

    def backward_tensor(self, op, *grad):
        x = op.inputs
        return script_ops.py_func(self.backward, [*x, *grad], self.Tin, name=self.name+'Grad')

    def call(self, inputs):
        if isinstance(inputs, list):
            inputs = [ops.convert_to_tensor(one_input) for one_input in inputs]
        else:
            inputs = [ops.convert_to_tensor(inputs)]
        # Register and override the gradients
        ops.RegisterGradient(self._id)(self.backward_tensor)
        g = ops.get_default_graph()
        with g.gradient_override_map({"PyFunc": self._id, "pyfunc_0": self._id, "PyFuncStateless": self._id}):
            res = script_ops.py_func(self.forward, inputs, self.Tout, name=self.name)
            oshape = self._output_shape([inp.get_shape() for inp in inputs])
            if isinstance(res, list):
                for i in range(len(res)):
                    res[i].set_shape(oshape[i])
            return res

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self._output_shape is None:
            raise NotImplementedError('We could not automatically infer '
                                      'the static shape of the External\'s output.'
                                      ' Please specify the `output_shape` for'
                                      ' this External.')
        else:
            shape = self._output_shape(input_shape)
            if not isinstance(shape, (list, tuple)):
                raise ValueError(
                    '`output_shape` function must return a tuple or a list of tuples.')
            # List here can represent multiple outputs or single output.
            if isinstance(shape, list):
                # Convert list representing single output into a tuple.
                if isinstance(shape[0], (int, type(None))):
                    shape = tuple(shape)
                else:
                    return [
                        tensor_shape.TensorShape(single_shape) for single_shape in shape
                    ]
            return tensor_shape.TensorShape(shape)
    
    def get_config(self):
        config = {
            'forward': activations.serialize(self.forward),
            'backward': activations.serialize(self.backward),
            'Tin': dtype_serialize(self.Tin),
            'Tout': dtype_serialize(self.Tout),
            'output_shape': activations.serialize(self._output_shape),
            'id': self._id,
        }
        base_config = super(External, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
