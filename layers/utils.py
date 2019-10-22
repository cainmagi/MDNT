'''
################################################################
# Layers - Utilities
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Utilities for some newly defined layers.
# Version: 0.10 # 2019/10/20
# Comments:
#   Create this submodule.
################################################################
'''

def normalize_slice(value):
    '''Transform an iterable of integers (or `None`) into a slice.
    Arguments:
        value: The input iterable object, which would be converted to a slice
               indicating tuple.
    Returns:
        A tuple of 2/3 integers or `None`.
    Raises:
        ValueError: If something else than an int/long or iterable thereof was
                    passed.
    '''
    try:
        value_tuple = tuple(value)
    except TypeError:
        raise ValueError('One input value object could not be converted into'
                         ' a slice: ' + str(value))
    if len(value_tuple) not in (2, 3):
        raise ValueError(str(value) + ' should has 2/3 integers or `None`, but'
                         ' actually has ' + str(len(value_tuple)) + ' elements.')
    for single_value in value_tuple:
        if single_value is None:
            continue
        try:
            int(single_value)
        except (ValueError, TypeError):
            raise ValueError(str(value) + ' should only include integers or `None`'
                             ', but actually includes element ' + str(single_value) +
                             ' of type ' + str(type(single_value)))
    return value_tuple

def normalize_slices(value, name):
    '''Transforms an iterable of tuples into a slice tuple.
    Arguments:
        value: The input iterable object, which would be converted to a slice
               indicating tuple.
        name:  The name of the argument being validated.
    Returns:
        A tuple of n tuples, where n is inferred by input value.
    Raises:
        ValueError: If something else than an int/long or iterable thereof was
                    passed.
    '''
    try:
        value_tuple = (normalize_slice(value),)
    except ValueError:
        value_tuple = []
        try:
            value_tuple = tuple(map(normalize_slice, value))
        except ValueError as e:
            raise ValueError('The `' + name + '` argument must be a tuple of slices' +
                             '. Received: ' + str(value) + ' including element with' +
                             ' error: ' + str(e))
    return value_tuple
    
def normalize_abtuple(value, name, n=None):
    '''Transforms a single integer or iterable of integers into an integer tuple
       with an arbitrary length.
    Arguments:
        value: The value to validate and convert. Could an int, or any iterable
        of ints.
        n: The size of the tuple to be returned, if set `None`, the tuple would
           have an arbitrary length.
        name: The name of the argument being validated, e.g. "strides" or
              "dims". This is only used to format error messages.
    Returns:
        A tuple of n integers. If n is None, the tuple length is inferred by
        input value.
    Raises:
        ValueError: If something else than an int/long or iterable thereof was
                    passed.
    '''
    str_n = ('a tuple of ' + str(n) + 'integers') if n is None else ('a tuple')
    if isinstance(value, int):
        if n is None:
            n = 1
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError('The `' + name + '` argument must be ' + str_n +
                             '. Received: ' + str(value))
        if n is not None and len(value_tuple) != n:
            raise ValueError('The `' + name + '` argument must be ' + str_n +
                             '. Received: ' + str(value))
    for single_value in value_tuple:
        try:
            int(single_value)
        except (ValueError, TypeError):
            raise ValueError('The `' + name + '` argument must be ' + str_n +
                             '. Received: ' + str(value) + ' including element ' +
                             str(single_value) + ' of type ' + 
                             str(type(single_value)))
    return value_tuple

def slice_len_for(slc, seqlen):
    '''
    Infer the length of a slice object
        slc:    Slice object.
        seqlen: Where the slice is applied.
    '''
    start, stop, step = slc.indices(seqlen)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)