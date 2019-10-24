'''
################################################################
# Data
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Extended data parser for tf-K standard IO APIs.
# Version: 0.15 # 2019/3/30
# Comments:
#   Add `H5GCombiner` into this module.
# Version: 0.10 # 2019/3/26
# Comments:
#   Create this submodule.
################################################################
'''

# Import sub-modules
from .h5py import H5HGParser, H5SupSaver, H5GParser, H5GCombiner, H5VGParser

__all__ = ['H5HGParser', 'H5SupSaver', 'H5GParser', 'H5GCombiner', 'H5VGParser']

# Set this local module as the prefered one
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

# Delete private sub-modules
del extend_path