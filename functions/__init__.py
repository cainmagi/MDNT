'''
################################################################
# Functions
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Extended functions for MDNT. Mainly including new losses and
# metrics. 
# Version: 0.10 # 2019/6/13
# Comments:
#   Create this submodule.
################################################################
'''

# Import sub-modules
from . import losses
from . import metrics

# Set layer dictionaries
customObjects = {
    'linear_jaccard_index': losses.linear_jaccard_loss,
    'lovasz_jaccard_loss': losses.lovasz_jaccard_loss,
    'signal_to_noise': metrics.signal_to_noise,
    'correlation': metrics.correlation,
    'jaccard_index': metrics.jaccard_index
}

__all__ = list(customObjects.keys())

# Set this local module as the prefered one
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

# Delete private sub-modules
del extend_path