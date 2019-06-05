'''
################################################################
# Layers
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Modern network layers. This sub-module would include some 
# effective network layers which are not introduced in tf-K.
# All of these modules are produced by standard tf-K APIs.
# Version: 0.10 # 2019/3/23
# Comments:
#   Create this submodule.
################################################################
'''

# Import sub-modules
from .dense import DenseTied
from .normalize import InstanceNormalization, GroupNormalization
from .conv import AConv1D, AConv2D, AConv3D, AConv1DTranspose, AConv2DTranspose, AConv3DTranspose#, AConv1DTied, AConv2DTied, AConv3DTied
from .external import PyExternal
from .residual import Residual1D, Residual1DTranspose, Residual2D, Residual2DTranspose, Residual3D, Residual3DTranspose, Resnext1D, Resnext1DTranspose, Resnext2D, Resnext2DTranspose, Resnext3D, Resnext3DTranspose
from .inception import Inception1D, Inception2D, Inception3D, Inception1DTranspose, Inception2DTranspose, Inception3DTranspose, Inceptres1D, Inceptres2D, Inceptres3D, Inceptres1DTranspose, Inceptres2DTranspose, Inceptres3DTranspose

# Set layer dictionaries
customObjects = {
    'DenseTied': DenseTied,
    'InstanceNormalization': InstanceNormalization,
    'GroupNormalization': GroupNormalization,
    'AConv1D': AConv1D,
    'AConv2D': AConv2D,
    'AConv3D': AConv3D,
    'AConv1DTranspose': AConv1DTranspose,
    'AConv2DTranspose': AConv2DTranspose,
    'AConv3DTranspose': AConv3DTranspose,
    'Residual1D': Residual1D,
    'Residual2D': Residual2D,
    'Residual3D': Residual3D,
    'Residual1DTranspose': Residual1DTranspose,
    'Residual2DTranspose': Residual2DTranspose,
    'Residual3DTranspose': Residual3DTranspose,
    'Resnext1D': Resnext1D,
    'Resnext2D': Resnext2D,
    'Resnext3D': Resnext3D,
    'Resnext1DTranspose': Resnext1DTranspose,
    'Resnext2DTranspose': Resnext2DTranspose,
    'Resnext3DTranspose': Resnext3DTranspose,
    'Inception1D': Inception1D,
    'Inception2D': Inception2D,
    'Inception3D': Inception3D,
    'Inception1DTranspose': Inception1DTranspose,
    'Inception2DTranspose': Inception2DTranspose,
    'Inception3DTranspose': Inception3DTranspose,
    'Inceptres1D': Inceptres1D,
    'Inceptres2D': Inceptres2D,
    'Inceptres3D': Inceptres3D,
    'Inceptres1DTranspose': Inceptres1DTranspose,
    'Inceptres2DTranspose': Inceptres2DTranspose,
    'Inceptres3DTranspose': Inceptres3DTranspose,
    'PyExternal': PyExternal
}

__all__ = list(customObjects.keys())

# Set alias
#res = residual

# Set this local module as the prefered one
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

# Delete private sub-modules
del extend_path