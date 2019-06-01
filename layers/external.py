'''
################################################################
# Layers - External API layer
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

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import activations
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.keras.utils import tf_utils

def dtype_serialize(input_dtypes):
    if isinstance(input_dtypes, list):
        return [dtypes.as_dtype(get_dt).as_datatype_enum for get_dt in input_dtypes]
    else:
        return [dtypes.as_dtype(input_dtypes).as_datatype_enum]

def dtype_get(input_serials):
    if isinstance(input_serials, list):
        return [dtypes.as_dtype(get_dt) for get_dt in input_serials]
    else:
        return [dtypes.as_dtype(input_serials)]

class PyExternal(Layer):
    """External API layer for generic python function.
    `PyExternal` is used to introduce a non-parameter function from an 
    external library and enable it to participate the learning workflow.
    Therefore, this layer is requires users to provide:
        1. The forward propagation function `forward()`.
        2. The back propagation function `backward()`.
        3. The shape calculation function `output_shape()`.
    Arguments:
        forward:      the forward propagating function. It serves as 
                      `y=F(x)`, where `x` may be a list of multiple inputs.
        backward:     the back propagation function. It serves as
                      `dx=B(...)`, where the input of this function is
                      determined by `xEnable`, `yEnable`, `dyEnable`.
        Tin:          a list of input tf.DType.
        Tout:         a list of output tf.DType.
        output_shape: a tf.TensorShape, a tuple/list or a function. It is
                      used for estimating the output shape fast.
        xEnable,
        yEnable,
        dyEnable:     enable users to customize the input of the backward
                      function. If only the `xEnable` is `True`, the input
                      of the function would be `B(x)`, For another example,
                      if only both `yEnable` and `dyEnable` are `True`, the
                      input of the function would be `B(y, dy)`.
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
                 output_shape,
                 Tin,
                 Tout,
                 xEnable=True,
                 yEnable=False,
                 dyEnable=True,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
          kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(PyExternal, self).__init__(**kwargs)
        self.forward = activations.get(forward)
        self.backward = activations.get(backward)
        self.Tin = dtype_get(Tin)
        self.Tout = dtype_get(Tout)
        self.xEnable = xEnable
        self.yEnable = yEnable
        self.dyEnable = dyEnable

        if output_shape is None:
            raise NotImplementedError('We could not automatically infer '
                                      'the static shape of the PyExternal\'s output.'
                                      ' Please specify the `output_shape` for'
                                      ' this PyExternal.')
        else:
            self._output_shape = activations.get(output_shape)

        self.supports_masking = True

    def call(self, inputs):
        if isinstance(inputs, list):
            inputs = [ops.convert_to_tensor(one_input) for one_input in inputs]
        else:
            inputs = [ops.convert_to_tensor(inputs)]

        # Define ops with first-order gradients
        @custom_gradient.custom_gradient
        def _external_func(*x):
            y = script_ops.eager_py_func(self.forward, x, self.Tout)
            def _external_func_grad(*grad):
                iList = []
                if self.xEnable:
                    iList.extend(x)
                if self.yEnable:
                    if isinstance(y, (list, tuple)):
                        iList.extend(y)
                    else:
                        iList.append(y)
                if self.dyEnable:
                    iList.extend(grad)
                return script_ops.eager_py_func(self.backward, iList, self.Tin)
            return y, _external_func_grad

        res = _external_func(*inputs)
        oshape = self._output_shape([inp.get_shape() for inp in inputs])
        if isinstance(res, list):
            for i in range(len(res)):
                res[i].set_shape(oshape[i])
        return res

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
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
            'xEnable': self.xEnable,
            'yEnable': self.yEnable,
            'dyEnable': self.dyEnable
        }
        base_config = super(PyExternal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
