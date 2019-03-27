'''
################################################################
# Compatibility check
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Use this module to check whether we need to open the
# compatible mode.
# Version: 0.10 # 2019/3/27
# Comments:
#   Create this compatible module.
################################################################
'''

# Check compatibility
import tensorflow

if [int(i) for i in tensorflow.__version__.split('.')] < [1, 12, 0]:
    COMPATIBLE_MODE = True
else:
    COMPATIBLE_MODE = False