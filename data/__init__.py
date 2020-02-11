'''
################################################################
# Data
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Extended data parser for tf-K standard IO APIs.
# Version: 0.18 # 2020/02/10
# Comments:
#   Add `H5Converter` into this module.
# Version: 0.16 # 2019/10/23
# Comments:
#   Add `H5VGParser` into this module.
# Version: 0.15 # 2019/3/30
# Comments:
#   Add `H5GCombiner` into this module.
# Version: 0.10 # 2019/3/26
# Comments:
#   Create this submodule.
################################################################
'''

# Import sub-modules
from .h5py import H5HGParser, H5SupSaver, H5GParser, H5GCombiner, H5VGParser, H5Converter

__all__ = ['H5HGParser', 'H5SupSaver', 'H5GParser', 'H5GCombiner', 'H5VGParser', 'H5Converter']

# Set this local module as the prefered one
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

# Delete private sub-modules
del extend_path