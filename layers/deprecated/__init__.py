'''
################################################################
# Layers (deprecated)
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Put the deprecated libs here.
# Version: 0.10 # 2019/5/23
# Comments:
#   Create this submodule.
################################################################
'''

# Import sub-modules
from .external import External
from .dense import DenseTied

__all__ = ['External', 'DenseTied']