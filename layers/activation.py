'''
################################################################
# Layers - Activation
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Extend the activation layer APIs. It allows more useful
# functions for building a complicated network.
# Version: 0.17 # 2019/10/27
# Comments:
#   Remove the "c" variable (slack variable) for RestrictSub,
#   because this variable is useless.
# Version: 0.16 # 2019/10/23
# Comments:
#   Finish the ExpandDims layer.
# Version: 0.15 # 2019/10/22
# Comments:
#   Finish the Restrict and RestrictSub layers.
# Version: 0.12 # 2019/10/20
# Comments:
#   Finish the Slice layer.
# Version: 0.10 # 2019/10/19
# Comments:
#   Create this submodule.
################################################################
'''

import copy
import numpy as np
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn_ops

from .utils import normalize_abtuple, normalize_slices, slice_len_for

from .. import compat
if compat.COMPATIBLE_MODE:
    from tensorflow.python.keras.engine.base_layer import InputSpec
else:
    from tensorflow.python.keras.engine.input_spec import InputSpec

class ExpandDims(Layer):
    '''Expand a new dimension at the specific location.
    This is the layer version for using tf.expand_dims.
    Arguments:
        axis: The location where we insert the new dimension.
    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape:
        Same as the input shape except the newly inserted
        dimension.
    '''

    def __init__(self, axis, **kwargs):
        super(ExpandDims, self).__init__(**kwargs)
        self.axis = int(axis)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape = copy.copy(input_shape)
        output_shape.insert(self.axis, 1)
        return tensor_shape.TensorShape(output_shape)

    def call(self, inputs):
        return array_ops.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(ExpandDims, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RestrictSub(Layer):
    '''Restrict the last dimension into given sub-ranges
    Use rebias factors and softmax function to rebias the last dimension of a
    tensor into given sub-ranges. Different from Restrict layer, this layer 
    is used for restricting the sum of outputs in a specified range. Consider
    x_i is the ith slice along the last dimension, the restrict-sub operation
    could be formulated as
        `y_i = l + (h - l) * softmax(a * x + b)_i`
    `a` and `b` are learnable vectors. The  range of `y_i` would be restricted
    in `(l_i, h_i)`. Notice that there should be `l < h`.
    It is suggest to let `x` has N+1 length if you want to predict N-length
    values, because one element should be reserved as "blank area".
    Since the sum satisfies `0 < sum_i(softmax(x, c)) < 1`, this operation
    would ensure that the sum of outputs to be limited in (l, h).
    Arguments:
        low_bound: The lower bound of the restricted range. Need to be a
                   float (or could be converted to a float).
        sup_bound: The super bound of the restricted range. Need to be a
                   float (or could be converted to a float).
        with_sum: Whether to use cumsum to control the outputs. It supports
            3 modes:
            `i`: Increasing mode, where the outputs would be
                `y_i = y_{i-1} + l + (h - l) * softmax(a * x + b, c)_i`
                the lower bound `l` requires to be >=0.
            `d`: Decreasing mode, where the outputs would be
                `y_{i-1} = y_i + l + (h - l) * softmax(a * x + b, c)_i`
                the lower bound `l` requires to be >=0.
            `n`: Do not apply cumsum scheme.
        a_initializer: Initializer for the a weight.
        b_initializer: Initializer for the b weight.
        c_initializer: Initializer for the c weight.
        a_regularizer: Optional regularizer for the a weight.
        b_regularizer: Optional regularizer for the b weight.
        c_regularizer: Optional regularizer for the c weight.
        a_constraint: Optional constraint for the a weight.
        b_constraint: Optional constraint for the b weight.
        c_constraint: Optional constraint for the c weight.
    Input shape:
        Arbitrary, use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape:
        The same as input shape.
    '''
    def __init__(self,
                 low_bound,
                 sup_bound,
                 with_sum='n',
                 a_initializer='ones',
                 a_regularizer=None,
                 a_constraint=None,
                 b_initializer='zeros',
                 b_regularizer=None,
                 b_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
          kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(RestrictSub, self).__init__(
            activity_regularizer=None, **kwargs)
        self.low_bound, self.sup_bound = float(low_bound), float(sup_bound)
        if with_sum not in ('i', 'd', 'n'):
            raise ValueError('The input `with_sum` only supports 3 modes: i, d, n. But received ' + str(with_sum))
        self.with_sum = with_sum
        self.a_initializer = initializers.get(a_initializer)
        self.a_regularizer = regularizers.get(a_regularizer)
        self.a_constraint = constraints.get(a_constraint)
        self.b_initializer = initializers.get(b_initializer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.b_constraint = constraints.get(b_constraint)
        self.supports_masking = True

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                           'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.get_a = self.add_weight(
            'a',
            shape=(last_dim,),
            initializer=self.a_initializer,
            regularizer=self.a_regularizer,
            constraint=self.a_constraint,
            dtype=self.dtype,
            trainable=True)
        self.get_b = self.add_weight(
            'b',
            shape=(last_dim,),
            initializer=self.b_initializer,
            regularizer=self.b_regularizer,
            constraint=self.b_constraint,
            dtype=self.dtype,
            trainable=True)
        super(RestrictSub, self).build(input_shape)

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs)
        input_shape = K.int_shape(inputs)
        # Broadcast a, b
        broadcast_shape = [1] * (len(input_shape)-1) + [input_shape[-1]]
        broadcast_a = K.reshape(self.get_a, broadcast_shape)
        broadcast_b = K.reshape(self.get_b, broadcast_shape)
        broadcast_l = K.constant(self.low_bound, dtype=self.dtype)
        broadcast_s = K.constant(self.sup_bound - self.low_bound, dtype=self.dtype)
        y = nn_ops.softmax(broadcast_a * inputs + broadcast_b, axis=-1)
        if self.with_sum == 'i':
            y = math_ops.cumsum(y, axis=-1)
        elif self.with_sum == 'd':
            y = math_ops.cumsum(y, axis=-1, reverse=True)
        return broadcast_l + broadcast_s * y

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'low_bound': self.low_bound,
            'sup_bound': self.sup_bound,
            'with_sum': self.with_sum,
            'a_initializer': initializers.serialize(self.a_initializer),
            'a_regularizer': regularizers.serialize(self.a_regularizer),
            'a_constraint': constraints.serialize(self.a_constraint),
            'b_initializer': initializers.serialize(self.b_initializer),
            'b_regularizer': regularizers.serialize(self.b_regularizer),
            'b_constraint': constraints.serialize(self.b_constraint)
        }
        base_config = super(RestrictSub, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Restrict(Layer):
    '''Restrict the last dimension into a given range
    Use rebias factors and sigmoid function to rebias the last dimension of a
    tensor into a given range. Consider x_i is the ith slice along the last
    dimension, the restrict operation could be formulated as
        `y_i = l_i + (h_i - l_i) * sigmoid(a_i * x_i + b_i)`
    where `a_i` and `b_i` are learnable factors. The range of `y_i` would be
    restricted in `(l_i, h_i)`. Notice that there should be `l_i < h_i`
    Arguments:
        low_bound: The lower bound of the restricted range. If given a list,
            the restriction will be performed for each element independently.
            If given a scalar, the restriction will be performed along the
            whole axis (each element share the same restriction).
        sup_bound: The super bound of the restricted range. Use like the
            `low_bound`.
        with_sum: Whether to use cumsum to control the outputs. It supports
            3 modes:
            `i`: Increasing mode, where the outputs would be
                `y_i = y_{i-1} + l_i + (h_i - l_i) * sigmoid(a_i * x_i + b_i)`
                the lower bound `l_i` requires to be >=0.
            `d`: Decreasing mode, where the outputs would be
                `y_{i-1} = y_i + l_i + (h_i - l_i) * sigmoid(a_i * x_i + b_i)`
                the lower bound `l_i` requires to be >=0.
            `n`: Do not apply cumsum scheme.
        a_initializer: Initializer for the a weight.
        b_initializer: Initializer for the b weight.
        a_regularizer: Optional regularizer for the a weight.
        b_regularizer: Optional regularizer for the b weight.
        a_constraint: Optional constraint for the a weight.
        b_constraint: Optional constraint for the b weight.
    Input shape:
        Arbitrary, use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape:
        The same as input shape.
    '''
    def __init__(self,
                 low_bound,
                 sup_bound,
                 with_sum='n',
                 a_initializer='ones',
                 a_regularizer=None,
                 a_constraint=None,
                 b_initializer='zeros',
                 b_regularizer=None,
                 b_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
          kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(Restrict, self).__init__(
            activity_regularizer=None, **kwargs)
        try:
            low_bound, sup_bound = float(low_bound), float(sup_bound)
        except TypeError:
            pass
        check_l, check_s = isinstance(low_bound, float), isinstance(sup_bound, float)
        if (check_l and (not check_s)) or (check_s and (not check_l)):
            raise TypeError('The input `low_bound` and `sup_bound` does not share the same float type.')
        elif check_l and check_s:
            self.arg_array = False
        else:
            check_l, check_s = isinstance(low_bound, (list, tuple)), isinstance(sup_bound, (list, tuple))
            if (check_l and (not check_s)) or (check_s and (not check_l)):
                raise TypeError('The input `low_bound` and `sup_bound` does not share the same list/tuple type.')
            if len(low_bound) != len(sup_bound):
                raise TypeError('The input `low_bound` and `sup_bound` does not share the same length')
            for l,s in zip(low_bound, sup_bound):
                if l >= s:
                    raise ValueError('The input `low_bound` should be less than `sup_bound`, but received ' + 
                                     str(low_bound) + ' ' + str(sup_bound))
            if with_sum == 'i' and len(low_bound) > 1:
                for l in low_bound[1:]:
                    if l < 0:
                        raise ValueError('When set increasing mode, each element of `low_bound[1:]` should be '
                                         'non-negative, but received ' + str(low_bound))
            elif with_sum == 'd' and len(low_bound) > 1:
                for l in low_bound[:-1]:
                    if l < 0:
                        raise ValueError('When set increasing mode, each element of `low_bound[:-1]` should be '
                                         'non-negative, but received ' + str(low_bound))
            if check_l and check_s:
                self.arg_array = True
            else:
                raise TypeError('At least one of `low_bound` and `sup_bound` has wrong type.')
            low_bound = np.array(low_bound, dtype=np.float)
            sup_bound = np.array(sup_bound, dtype=np.float)
        self.low_bound = low_bound
        self.sup_bound = sup_bound
        if with_sum not in ('i', 'd', 'n'):
            raise ValueError('The input `with_sum` only supports 3 modes: i, d, n. But received ' + str(with_sum))
        self.with_sum = with_sum
        self.a_initializer = initializers.get(a_initializer)
        self.a_regularizer = regularizers.get(a_regularizer)
        self.a_constraint = constraints.get(a_constraint)
        self.b_initializer = initializers.get(b_initializer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.b_constraint = constraints.get(b_constraint)
        self.supports_masking = True

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                           'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if self.arg_array:
            L = len(self.low_bound)
            if L != last_dim:
                raise ValueError('The length of the bound vector does not correspond to the input shape.')
        else:
            L = 1
        self.get_a = self.add_weight(
            'a',
            shape=(L,),
            initializer=self.a_initializer,
            regularizer=self.a_regularizer,
            constraint=self.a_constraint,
            dtype=self.dtype,
            trainable=True)
        self.get_b = self.add_weight(
            'b',
            shape=(L,),
            initializer=self.b_initializer,
            regularizer=self.b_regularizer,
            constraint=self.b_constraint,
            dtype=self.dtype,
            trainable=True)
        super(Restrict, self).build(input_shape)

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs)
        input_shape = K.int_shape(inputs)
        if self.arg_array:
            broadcast_shape = [1] * (len(input_shape)-1) + [input_shape[-1]]
            broadcast_a = K.reshape(self.get_a, broadcast_shape)
            broadcast_b = K.reshape(self.get_b, broadcast_shape)
            broadcast_l = K.reshape(K.constant(self.low_bound, dtype=self.dtype), broadcast_shape)
            broadcast_s = K.reshape(K.constant(self.sup_bound - self.low_bound, dtype=self.dtype), broadcast_shape)
        else:
            broadcast_a = self.get_a
            broadcast_b = self.get_b
            broadcast_l = K.constant(self.low_bound, dtype=self.dtype)
            broadcast_s = K.constant(self.sup_bound - self.low_bound, dtype=self.dtype)
        y = broadcast_l + broadcast_s * math_ops.sigmoid(broadcast_a * inputs + broadcast_b)
        if self.with_sum == 'i':
            y = math_ops.cumsum(y, axis=-1)
        elif self.with_sum == 'd':
            y = math_ops.cumsum(y, axis=-1, reverse=True)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        if self.arg_array:
            serial_low_bound = self.low_bound.tolist()
            serial_sup_bound = self.sup_bound.tolist()
        else:
            serial_low_bound, serial_sup_bound = self.low_bound, self.sup_bound
        config = {
            'low_bound': serial_low_bound,
            'sup_bound': serial_sup_bound,
            'with_sum': self.with_sum,
            'a_initializer': initializers.serialize(self.a_initializer),
            'a_regularizer': regularizers.serialize(self.a_regularizer),
            'a_constraint': constraints.serialize(self.a_constraint),
            'b_initializer': initializers.serialize(self.b_initializer),
            'b_regularizer': regularizers.serialize(self.b_regularizer),
            'b_constraint': constraints.serialize(self.b_constraint)
        }
        base_config = super(Restrict, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Slice(Layer):
    '''Extract a slice from a given tensor
    This layer is implemented by tf.Tensor.getitem, which allows us to extract
    patches from an unknown-shape tensor. The given tensor could be in any
    dimension, so this layer is a full-functional interface of tensor slicing
    operator.
    Arguments:
        dims: The axes where we extract patches. It should be a tuple.
            If given an integer, it would be equivalent to `(dims,)`.
        slices: A tuple of N tuples of 2/3 ints, where N is the dimension
            number. there allows `None` in any int of the sub-tuple.
            This argument could be interpreted as
                `((begin_1, end_1), (begin_2 end_2, step_2), (begin_3, end_3) ...)`
            If given a tuple of 2/3 ints, it would be equivalent to
                `(slices,)`
    Input shape:
        Arbitrary, use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape:
        The sliced tensor shape along the picked dims.
    Example:
    ```python
        # the original API for slicing
        x = tf.zeros([10, 10, 10])
        x_s = x[:, 2:10:2, :]
        # the equivalent operation
        x_s2 = mdnt.layers.Slice(dims=1, slices=(2,10,2))(x)
    ```
    '''

    def __init__(self, dims, slices, **kwargs):
        super(Slice, self).__init__(**kwargs)
        if isinstance(dims, int):
            dims = (dims,)
        dims = normalize_abtuple(dims, 'dims', n=None)
        slices = normalize_slices(slices, 'slices')
        if len(dims) != len(slices):
            raise ValueError('`dims` and `slices` should share the same '
                             'length and correspond to each other.')
        self.dims = dims
        self.slices = slices

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        input_ndim = len(input_shape)
        norm_dims = tuple( input_ndim+d if d < 0 else d for d in self.dims)
        if len(norm_dims) > 1:
            sort_dims = sorted(norm_dims, key=lambda x:x)
            for a, b in zip(sort_dims[:-1], sort_dims[1:]):
                if a >= b:
                    raise ValueError('There is repeated dimensions in the input'
                                     ' `dims`, received: ' + str(norm_dims))
        self.ind_built = tuple(sorted(zip(norm_dims, map(lambda x:slice(*x), self.slices)), 
                                      key=lambda x:x[0]))
        self.input_ndim = input_ndim

        super(Slice, self).build(input_shape)

    def _compute_shape_one_axis(self, cur_dim, shape_one_axis):
        for d, s in self.ind_built:
            if cur_dim == d:
                return slice_len_for(s, shape_one_axis)
        return shape_one_axis

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape = []
        for i, isp in enumerate(input_shape):
            if isp is None:
                output_shape.append(None)
            else:
                output_shape.append(self._compute_shape_one_axis(i, isp))
        return tensor_shape.TensorShape(output_shape)

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs)
        get_slices = []
        for i in range(self.input_ndim):
            for d, s in self.ind_built:
                if i == d:
                    get_slices.append(s)
                    break
            else:
                get_slices.append(slice(None))
        return inputs[get_slices]

    def get_config(self):
        config = {
            'dims': self.dims,
            'slices': self.slices
        }
        base_config = super(Slice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))