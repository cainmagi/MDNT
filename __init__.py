'''
################################################################
# Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# This is a pakage for extending the tensorflow-keras to modern
# deep network design. It would introduce some state-of-art
# network blocks, data parsing utilities, logging modules and
# more extensions.
# Loading this module would not cause conflictions on other
# modules (if users do not use `from mdnt import *` to override
# utilites from other modules. However, it will provide some
# tools with the same/similar name and functions compared to
# plain tensorflow-keras.
# Version: 0.50-b # 2019/6/11
# Comments:
#   1. Fix a bug for implementing the channel_first mode for
#      AConv in .layers.
#   2. Finish InstanceGaussianNoise in .layers.
#   3. Prepare the test for adding dropout to residual layers
#      in .layers.
# Version: 0.50 # 2019/6/11
# Comments:
#   1. Finish Conv1DTied, Conv2DTied, Conv3DTied in .layers.
#   2. Switch back to the 0.48 version for .layers.DenseTied
#      APIs because testing show that the modification in
#      0.48-b will cause bugs.
# Version: 0.48-b # 2019/6/10
# Comments:
#   A Test on replacing the .layers.DenseTied APIs like 
#   tf.keras.layers.Wrappers.
# Version: 0.48 # 2019/6/9
# Comments:
# 1. Finish Inceptplus1D, Inceptplus2D, Inceptplus3D,
#           Inceptplus1DTranspose, Inceptplus2DTranspose,
#           Inceptplus3DTranspose in .layers.
# 2. Minor changes for docstrings and default settings in 
#    .layers.inception.
# Version: 0.45-b # 2019/6/7
# Comments:
# 1. Enable the ResNeXt to estimate the latent group and local 
#    filter number.
# 2. Make a failed try on implementing quick group convolution,
#    testing results show that using tf.nn.depthwise_conv2d
#    to replace multiple convND ops would cause the computation
#    to be even slower.
# Version: 0.45 # 2019/6/6
# Comments:
# 1. Enable Modern convolutional layers to work with group
#    convolution.
# 2. Reduce the memory consumption for network construction
#    when using ResNeXt layers in case of out of memory (OOM)
#    problems.
# 3. Fix a minor bug for group convolution.
# Version: 0.42 # 2019/6/5
# Comments:
# 1. Add GroupConv1D, GroupConv2D, GroupConv3D in .layers.
# 2. Fix the bugs in channel detections for residual and
#    inception layers.
# Version: 0.40 # 2019/6/5
# Comments:
# 1. Finish Resnext1D, Resnext2D, Resnext3D,
#           Resnext1DTranspose, Resnext2DTranspose,
#           Resnext3DTranspose in .layers.
# 2. Fix the repeating biases problems in inception-residual
#    layers.
# Version: 0.38 # 2019/6/4
# Comments:
# 1. Finish Inceptres1D, Inceptres2D, Inceptres3D, 
#           Inceptres1DTranspose, Inceptres2DTranspose,
#           Inceptres3DTranspose in .layers.
# 2. Fix some bugs and revise docstrings for .layers.residual and
#    .layers.inception.
# Version: 0.36 # 2019/6/1
# Comments:
#   Finish Inception1D, Inception2D, Inception3D, 
#          Inception1DTranspose, Inception2DTranspose,
#          Inception3DTranspose in .layers.
# Version: 0.32 # 2019/5/31
# Comments:
#   Finish Residual1D, Residual2D, Residual3D, Residual1DTranspose, 
#          Residual2DTranspose, Residual3DTranspose in .layers.
# Version: 0.28 # 2019/5/24
# Comments:
# 1. Fix the bug about padding for transposed dilation 
#    convolutional layers.
# 2. Add a new option output_mshape to help transposed 
#    convolutional layers to control the desired output shape.
# 3. Finish PyExternal in .layers.
# Version: 0.24 # 2019/3/31
# Comments:
#   Finish H5GCombiner in .data.
# Version: 0.23 # 2019/3/26
# Comments:
#   1. Use keras.Sequence() to redefine H5GParser and 
#      H5HGParser.
#   2. Add compatible check.
# Version: 0.22 # 2019/3/26
# Comments:
#   Adjust the .data.h5py module to make it more generalized.
# Version: 0.20 # 2019/3/26
# Comments:
#   Finish H5HGParser, H5SupSaver, H5GParser in .data.
#   Finish DenseTied, InstanceNormalization, GroupNormalization,
#          AConv1D, AConv2D, AConv3D, AConv1DTranspose, 
#          AConv2DTranspose, AConv3DTranspose in .layers.
# Version: 0.10 # 2019/3/23
# Comments:
#   Create this project.
################################################################
'''

# Import sub-modules
from . import optimizers
from . import layers
from . import data

__version__ = '0.50-b'

# Set this local module as the prefered one
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

# Merge custom objects from sub-modules
from tensorflow.python.keras.engine.saving import load_model as _load_model
customObjects = dict()
customObjects.update(layers.customObjects)
def load_model(filepath, custom_objects=None, compile=True, *args, **kwargs):
    if isinstance(custom_objects, dict):
        custom_objects.update(customObjects)
    else:
        custom_objects = customObjects
    return _load_model(filepath, custom_objects, compile, *args, **kwargs)
    
# Delete private sub-modules and objects
del extend_path