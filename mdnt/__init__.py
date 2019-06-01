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

__version__ = '0.36'

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