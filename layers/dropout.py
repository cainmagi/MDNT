'''
################################################################
# Layers - Extended dropout and noise layers
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Extend the methods for adding dropouts and noises. Such
# methods may help the network avoid overfitting problems.
# Version: 0.10 # 2019/6/11
# Comments:
#   Create this submodule.
################################################################
'''

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer

from tensorflow.python.keras.layers.core import Dropout, SpatialDropout1D, SpatialDropout2D, SpatialDropout3D
from tensorflow.python.keras.layers.noise import GaussianDropout, AlphaDropout

from .. import compat
if compat.COMPATIBLE_MODE['1.12']:
    from tensorflow.python.keras.engine.base_layer import InputSpec
else:
    from tensorflow.python.keras.engine.input_spec import InputSpec

class InstanceGaussianNoise(Layer):
    """Apply additive zero-centered Gaussian noise.
    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process
    for real valued inputs.
    As it is a regularization layer, it is only active at training time.
    Different from tf.keras.layers.GaussianNoise, in this method, we
    add the noise in the instance normalized space:
        `output = std * ( (input-mean) / std + N(0, eps) ) + mean`.
    where `eps ~ U(0, alpha)`. So the noise strength would be scale-
    invariant to the input data.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceGaussianNoise`.
            Setting `axis=None` will normalize all values in each
            instance of the batch （Layer Normalization）.
            Axis 0 is the batch dimension. `axis` cannot be set to 0
            to avoid errors.
        alpha: float, maximal standard deviation of the noise 
            distribution. For example, when alpha = 0.3, it means
            the noise would be at most 30% of the input. 
        epsilon: Small float added to variance to avoid dividing by
            zero.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """
    def __init__(self,
                 axis=None,
                 alpha=0.3,
                 epsilon=1e-3,
                 **kwargs):
        super(InstanceGaussianNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.alpha = alpha
        self.epsilon = epsilon

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        def noised():
            eps = K.random_uniform(shape=[1], maxval=self.alpha)
            return inputs + K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=eps)
        get_noised = K.in_train_phase(noised, normed, training=training)

        retrived = stddev * get_noised + mean
        return retrived
        
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'axis': self.axis,
            'alpha': self.alpha
        }
        base_config = super(InstanceGaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def return_dropout(dropout_type, dropout_rate, axis=-1, rank=None):
    if dropout_type is None:
        return None
    elif dropout_type == 'plain':
        return Dropout(rate=dropout_rate)
    elif dropout_type == 'add':
        return InstanceGaussianNoise(axis=axis, alpha=dropout_rate)
    elif dropout_type == 'mul':
        return GaussianDropout(rate=dropout_rate)
    elif dropout_type == 'alpha':
        return AlphaDropout(rate=dropout_rate)
    elif dropout_type == 'spatial':
        if axis == 1:
            dformat = 'channels_first'
        else:
            dformat = 'channels_last'
        if rank == 1:
            return SpatialDropout1D(rate=dropout_rate)
        elif rank == 2:
            return SpatialDropout2D(rate=dropout_rate, data_format=dformat)
        elif rank == 3:
            return SpatialDropout3D(rate=dropout_rate, data_format=dformat)
        else:
            return None
    else:
        return None