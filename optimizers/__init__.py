'''
################################################################
# Optimizers
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Wrapping the optimizers in tf-K with default options. In this
# module, we would also try to propose some newly introduced 
# optimizers if need. 
# Version: 0.10 # 2019/3/23
# Comments:
#   Create this submodule.
################################################################
'''
# Import sub-modules
from ._default import optimizer as optimizer
from .mixture import Adam2SGD, Nadam2NSGD

# Set optimizer dictionaries
customObjects = {
    'Adam2SGD': Adam2SGD,
    'Nadam2NSGD': Nadam2NSGD
}

# Set this local module as the prefered one
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

__all__ = list(customObjects.keys())

# Delete private sub-modules and objects
del _default
del extend_path