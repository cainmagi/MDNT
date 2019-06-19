'''
################################################################
# Utilities
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Extended utilities for MDNT. This module includes useful tools
# that are not directly related to deep network architecture. 
# For example, it has callbacks for fitting a network, the pre-
# processing and postprocessing tools and APIs for drawing
# figures. 
# Version: 0.10 # 2019/6/16
# Comments:
#   Create this submodule.
################################################################
'''

# Import sub-modules
from . import callbacks
from ._default import save_model, load_model

__all__ = [
            'callbacks', 
            'save_model', 'load_model'
          ]

# Set this local module as the prefered one
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

# Delete private sub-modules
del extend_path