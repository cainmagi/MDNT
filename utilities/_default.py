'''
################################################################
# Utilities - Default tools.
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Basic tools for this module.
# The default tools would be imported directly into the current
# sub-module. It could be viewed as an extension of basic APIs
# in this category.
# Version: 0.35 # 2019/11/27
# Comments:
#   1. Fix a bug for checking the existence of the file when
#      saving a new model.
#   2. Fix a bug when saving repeated symbolic weights.
#   3. Enable save_model and load_model to support storing/
#      recovering a customized loss/metric function.
# Version: 0.31 # 2019/10/27
# Comments:
#   Let save_model support compression.
# Version: 0.30 # 2019/10/15
# Comments:
#   Let save_model and load_model support storing/recovering
#   variable loss weights.
# Version: 0.26 # 2019/6/16
# Comments:
#   Fix a small bug for load_model.
# Version: 0.25 # 2019/6/19
# Comments:
#   Revise the save_model and load_model and split the optimizer
#   configurations into another file.
# Version: 0.20 # 2019/6/18
# Comments:
#   Finsh this submodule containing save_model and load_model.
# Version: 0.10 # 2019/6/16
# Comments:
#   Create this submodule.
################################################################
'''

import io
import os
import json
import numpy as np
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.python.ops import variables
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.engine.saving import model_from_config, preprocess_weights_for_loading
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import serialization

from ..layers import customObjects as customObjects_layers
from ..functions import customObjects as customObjects_functions
from ..optimizers import customObjects as customObjects_optimizers

customObjects = dict()
customObjects.update(customObjects_layers)
customObjects.update(customObjects_functions)
customObjects.update(customObjects_optimizers)

# pylint: disable=g-import-not-at-top
try:
    import h5py
except ImportError:
    h5py = None

def save_model(model, filepath, headpath=None, optmpath=None, overwrite=True, include_optimizer=True, compress=True):
    """Saves a model to two JSON files and a HDF5 file
    The saved model would be divided into three parts. In the first part,
    i.e. the JSON file, there are:
        - the model's configuration (topology)
        - the model's parameter list
        - the model's optimizer's parameter list (if any)
    The second file is also a JSON file which contains
        - the configurations for current optimizer state
    It exists only when `overwrite` is `True`.
    The third file, i.e. HDF5 file, contains:
        - the model's weights
    Thus the saved model can be reinstantiated in the exact same state, 
    without any of the code used for model definition or training.
    This scheme is used to replace the original strategy in tf-Keras,
    because in tf.keras.Model.save, the model's (including the 
    optimizer's) configurations are saved in the heading of HDF5 file.
    However, the HDF5 file requires the heading to be smaller than
    64KB. Hence, a too complicated model may generates a too large
    configuration file. The refined version in MDNT avoid this situation
    by splitting the configurations into the JSON file, but it requires
    users to use mdnt.save_model and mdnt.load_model together.
    Arguments:
        model: Keras model instance to be saved.
        filepath: One of the following:
            - String, path where to save the model weights
            - `h5py.File` object where to save the model weights
        headpath: One of the following:
            - String, path where to save the model configurations
            - `File` object where to save the model configurations
            - If set None, would get deduced from `filepath`
        optmpath: One of the following:
            - String, path where to save the model configurations
            - `File` object where to save the model configurations
            - If set None, would get deduced from `filepath`
            - In most cases, this variable could be left `None`
        overwrite: Whether we should overwrite any existing
            model at the target location, or instead
            ask the user with a manual prompt.
        include_optimizer: If True, save optimizer's state together.
    Raises:
        ImportError: if h5py is not available.
    """

    if h5py is None:
        raise ImportError('`save_model` requires h5py.')

    from tensorflow.python.keras import __version__ as keras_version    # pylint: disable=g-import-not-at-top

    # TODO(psv) Add warning when we save models that contain non-serializable
    # entities like metrics added using `add_metric` and losses added using
    # `add_loss.`

    # Examine the file existence and open the HDF5 file.
    if not isinstance(filepath, h5py.File):
        # If file exists and should not be overwritten.
        filepath_nopsfx = os.path.splitext(filepath)[0]
        psfx = os.path.splitext(filepath)[1].casefold()
        if (psfx != '.h5') and (psfx != '.hdf5'):
            filepath = filepath_nopsfx + '.h5'
        if (not overwrite) and (os.path.isfile(filepath)):
            proceed = ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return
                
        f = h5py.File(filepath, mode='w')
        opened_new_file = True
    else:
        f = filepath
        opened_new_file = False

    # Infer the headpath if set None.
    if headpath is None:
        headpath = f.filename
        if hasattr(headpath, 'decode'):
            headpath = headpath.decode('utf-8')
        headpath = os.path.splitext(headpath)[0]
        ind = headpath.rfind('-')
        if ind != -1:
            headpath = headpath[:ind]
        ind = headpath.find('_loss')
        if ind != -1:
            headpath = headpath[:ind]
        ind = headpath.find('_acc')
        if ind != -1:
            headpath = headpath[:ind]
        headpath = headpath + '.json'

    # Examine the file existence and open the JSON file.
    if not isinstance(headpath, io.IOBase):
        # If file exists and should not be overwritten.
        if (not overwrite) and os.path.isfile(headpath):
            proceed = ask_to_proceed_with_overwrite(headpath)
            if not proceed:
                return
        psfx = os.path.splitext(headpath)[1].casefold()
        if (psfx != '.json'):
            headpath = headpath + '.json'
        fh = open(headpath, 'w')
        opened_new_head = True
    else:
        fh = headpath
        opened_new_head = False

    if include_optimizer and model.optimizer:
        if optmpath is None:
            optmpath = f.filename
            if hasattr(optmpath, 'decode'):
                optmpath = optmpath.decode('utf-8')
            optmpath = os.path.splitext(optmpath)[0] + '.json'
        
        if isinstance(optmpath, str) and optmpath == headpath:
            optmpath = os.path.splitext(optmpath)[0] + '_opt.json'

        # Examine the file existence and open the JSON file.
        if not isinstance(optmpath, io.IOBase):
            # If file exists and should not be overwritten.
            if (not overwrite) and os.path.isfile(optmpath):
                proceed = ask_to_proceed_with_overwrite(optmpath)
                if not proceed:
                    return
            psfx = os.path.splitext(optmpath)[1].casefold()
            if (psfx != '.json'):
                optmpath = optmpath + '.json'
            fo = open(optmpath, 'w')
            opened_new_optm = True
        else:
            fo = optmpath
            opened_new_optm = False
    else:
        fo = None

    try:
        json_dict = {
            'keras_version': str(keras_version),
            'backend': K.backend(),
            'model_config': {
                'class_name': model.__class__.__name__,
                'config': model.get_config()
            }
        }

        model_weights_group = f.create_group('model_weights')
        model_layers = model.layers
        save_weights_to_hdf5_group(model_weights_group, json_dict, model_layers, compress=compress)

        if include_optimizer and model.optimizer:
            if isinstance(model.optimizer, optimizers.TFOptimizer):
                logging.warning(
                    'TensorFlow optimizers do not '
                    'make it possible to access '
                    'optimizer attributes or optimizer state '
                    'after instantiation. '
                    'As a result, we cannot save the optimizer '
                    'as part of the model save file.'
                    'You will have to compile your model again after loading it. '
                    'Prefer using a Keras optimizer instead '
                    '(see keras.io/optimizers).')
            else:
                json_optm_dict = {
                    'training_config': {
                        'optimizer_config': {
                            'class_name': model.optimizer.__class__.__name__,
                            'config': model.optimizer.get_config()
                        },
                        'loss': model.loss,
                        'metrics': model._compile_metrics,
                        'weighted_metrics': model._compile_weighted_metrics,
                        'sample_weight_mode': model.sample_weight_mode
                        #'loss_weights': loss_weights
                    }
                }

                # Save loss weights
                loss_weights_group = f.create_group('loss_weights')
                save_loss_weights_to_hdf5_group(loss_weights_group, json_optm_dict, model.loss_weights, compress=compress)

                # Save optimizer weights.
                symbolic_weights = getattr(model.optimizer, 'weights')
                if symbolic_weights:
                    optimizer_weights_group = f.create_group('optimizer_weights')
                    weight_values = K.batch_get_value(symbolic_weights)
                    weight_names = []
                    for w, val in zip(symbolic_weights, weight_values):
                        name = str(w.name)
                        weight_names.append(name)
                    save_attributes_to_hdf5_group(json_dict, optimizer_weights_group.name, 'weight_names', weight_names)
                    for name, val in zip(weight_names, weight_values):
                        param_dset = optimizer_weights_group.create_dataset(
                                name, val.shape, dtype=val.dtype)
                        if not val.shape:
                            # scalar
                            param_dset[()] = val
                        else:
                            param_dset[:] = val
                
                fo.write(json.dumps(json_optm_dict, default=serialization.get_json_type, indent=4))
        
        # Save the JSON file.
        fh.write(json.dumps(json_dict, default=serialization.get_json_type, indent=4))

        f.flush()
        fh.flush()
        if fo is not None:
            fo.flush()
    finally:
        if opened_new_file:
            f.close()
        if opened_new_head:
            fh.close()
        if fo is not None:
            if opened_new_optm:
                fo.close()


def load_model(filepath, headpath=None, optmpath=None, custom_objects=None, compile=True):    # pylint: disable=redefined-builtin
    """Loads a model saved via `save_model`.
    This revised version split the configurations and weights of a model
    into two JSON files and an HDF5 file respectively. To learn why we should
    apply this technique. Check mdnt.save_model. Actually, this implement-
    ation requires users to use mdnt.save_model and mdnt.load_model together.
    Arguments:
        filepath: One of the following:
            - String, path to the saved model
            - `h5py.File` object from which to load the model
        headpath: One of the following:
            - String, path where to save the model configurations
            - `File` object where to save the model configurations
            - If set None, would get deduced from `filepath`
        optmpath: One of the following:
            - String, path where to save the model configurations
            - `File` object where to save the model configurations
            - If set None, would get deduced from `filepath`
            - In most cases, this variable could be left `None`
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.
        compile: Boolean, whether to compile the model
            after loading.
    Returns:
        A Keras model instance. If an optimizer was found
        as part of the saved model, the model is already
        compiled. Otherwise, the model is uncompiled and
        a warning will be displayed. When `compile` is set
        to False, the compilation is omitted without any
        warning.
    Raises:
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    if h5py is None:
        raise ImportError('`load_model` requires h5py.')

    if not custom_objects:
        custom_objects = {}
    custom_objects.update(customObjects)

    def convert_custom_objects(obj):
        """Handles custom object lookup.
        Arguments:
            obj: object, dict, or list.
        Returns:
            The same structure, where occurrences
            of a custom object name have been replaced
            with the custom object.
        """
        if isinstance(obj, list):
            deserialized = []
            for value in obj:
                deserialized.append(convert_custom_objects(value))
            return deserialized
        if isinstance(obj, dict):
            if ('class_name' in obj) and (obj['class_name'] in custom_objects):
                return deserialize_keras_object(
                    obj,
                    module_objects=globals(),
                    custom_objects=custom_objects,
                    printable_module_name='loss function')
            deserialized = {}
            for key, value in obj.items():
                deserialized[key] = convert_custom_objects(value)
            return deserialized
        if obj in custom_objects:
            return custom_objects[obj]
        return obj

    # Examine the input type and open the HDF5 file.
    opened_new_file = not isinstance(filepath, h5py.File)
    if opened_new_file:
        psfx = os.path.splitext(filepath)[1].casefold()
        if (psfx != '.h5') and (psfx != '.hdf5'):
            filepath = filepath + '.h5'
        f = h5py.File(filepath, mode='r')
    else:
        f = filepath
    
    # Infer the headpath if set None.
    if headpath is None:
        headpath = f.filename
        if hasattr(headpath, 'decode'):
            headpath = headpath.decode('utf-8')
        headpath = os.path.splitext(headpath)[0]
        ind = headpath.rfind('-')
        if ind != -1:
            headpath = headpath[:ind]
        ind = headpath.find('_loss')
        if ind != -1:
            headpath = headpath[:ind]
        ind = headpath.find('_acc')
        if ind != -1:
            headpath = headpath[:ind]
        headpath = headpath + '.json'

    # Examine the input type and open the JSON file.
    opened_new_head = not isinstance(headpath, io.IOBase)
    if opened_new_head:
        psfx = os.path.splitext(headpath)[1].casefold()
        if (psfx != '.json'):
            headpath = headpath + '.json'
        fh = open(headpath, 'r')
    else:
        fh = headpath

    # Check optimizer configuration file when setting `compile`.
    if compile:
        if optmpath is None:
            optmpath = f.filename
            if hasattr(optmpath, 'decode'):
                optmpath = optmpath.decode('utf-8')
            optmpath = os.path.splitext(optmpath)[0] + '.json'
        
        if isinstance(optmpath, str) and optmpath == headpath:
            optmpath = os.path.splitext(optmpath)[0] + '_opt.json'

        # Examine the file existence and open the JSON file.
        opened_new_optm = not isinstance(optmpath, io.IOBase)
        if opened_new_optm:
            psfx = os.path.splitext(optmpath)[1].casefold()
            if (psfx != '.json'):
                optmpath = optmpath + '.json'
            if os.path.isfile(optmpath):
                fo = open(optmpath, 'r')
            else:
                fo = None
        else:
            fo = optmpath
    else:
        fo = None

    model = None
    try:
        # Load all configurations from JSON file.
        json_dict = json.loads(fh.read())
        # instantiate model
        model_config = json_dict.get('model_config')
        if model_config is None:
            raise ValueError('No model found in config file.')
        model = model_from_config(model_config, custom_objects=custom_objects)

        # set weights
        load_weights_from_hdf5_group(f['model_weights'], json_dict, model.layers)

        if compile:
            # instantiate optimizer
            if fo is not None:
                json_optm_dict = json.loads(fo.read())
            else:
                json_optm_dict = dict()
            training_config = json_optm_dict.get('training_config')
            if training_config is None:
                logging.warning('No training configuration found in save file: '
                                'the model was *not* compiled. Compile it manually.')
                return model
            optimizer_config = training_config['optimizer_config']
            optimizer = optimizers.deserialize(
                optimizer_config, custom_objects=custom_objects)

            # Recover loss functions and metrics.
            loss = convert_custom_objects(training_config['loss'])
            metrics = convert_custom_objects(training_config['metrics'])
            weighted_metrics = convert_custom_objects(
                training_config.get('weighted_metrics', None))
            sample_weight_mode = training_config['sample_weight_mode']

            if 'loss_weights' in f:
                loss_weights = load_loss_weights_from_hdf5_group(f['loss_weights'], json_optm_dict)
            else: # Compatibility for old versions.
                loss_weights = training_config['loss_weights']

            # Compile model.
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                weighted_metrics=weighted_metrics,
                loss_weights=loss_weights,
                sample_weight_mode=sample_weight_mode)

            # Set optimizer weights.
            if 'optimizer_weights' in f:
                # Build train function (to get weight updates).
                # Models that aren't graph networks must wait until they are called
                # with data to _make_train_function() and so can't load optimizer
                # weights.
                if model._is_graph_network:    # pylint: disable=protected-access
                    model._make_train_function()
                    optimizer_weights_group = f['optimizer_weights']
                    optimizer_weight_names = load_attributes_from_hdf5_group(json_dict, optimizer_weights_group.name, 'weight_names')
                    optimizer_weight_values = [
                        optimizer_weights_group[n] for n in optimizer_weight_names
                    ]
                    try:
                        model.optimizer.set_weights(optimizer_weight_values)
                    except ValueError:
                        logging.warning('Error in loading the saved optimizer '
                                        'state. As a result, your model is '
                                        'starting with a freshly initialized '
                                        'optimizer.')
                else:
                    logging.warning('Sequential models without an `input_shape` '
                                    'passed to the first layer cannot reload their '
                                    'optimizer state. As a result, your model is'
                                    'starting with a freshly initialized optimizer.')

    finally:
        if opened_new_file:
            f.close()
        if opened_new_head:
            fh.close()
        if fo is not None:
            if opened_new_optm:
                fo.close()
    return model

def save_weights_to_hdf5_group(f, fh_dict, layers, compress=False):
    """Saves the weights of a list of layers to a HDF5 group.
    This is revised version. We split the attributes of HDF5 group into another
    JSON file to avoid the heading memory excessing problem. Compared to original
    Keras API, we need to load an extra file IO handle, fh_dict.
    Arguments:
        f:        HDF5 group.
        fh_dict:  JSON config dictionary.
        layers:   a list of layer instances.
        compress: whether to compress the weights.
    """
    compression = 'gzip' if compress else None
    save_attributes_to_hdf5_group(
            fh_dict, f.name, 'layer_names', [layer.name for layer in layers])

    for layer in layers:
        g = f.create_group(layer.name)
        symbolic_weights = layer.weights
        weight_values = K.batch_get_value(symbolic_weights)
        weight_names = []
        for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
            if hasattr(w, 'name') and w.name:
                name = str(w.name)
            else:
                name = 'param_' + str(i)
            weight_names.append(name)
        save_attributes_to_hdf5_group(fh_dict, g.name, 'weight_names', weight_names)
        for name, val in zip(weight_names, weight_values):
            param_dset = g.create_dataset(name, val.shape, dtype=val.dtype, compression=compression)
            if not val.shape:
                # scalar
                param_dset[()] = val
            else:
                param_dset[:] = val


def load_weights_from_hdf5_group(f, fh_dict, layers):
    """Implements topological (order-based) weight loading.
    This is revised version. We split the attributes of HDF5 group into another
    JSON file to avoid the heading memory excessing problem. Compared to original
    Keras API, we need to load an extra file IO handle, fh_dict.
    In the same time, the keras_version and backend infomation should be provided
    by fh_dict directly.
    Arguments:
        f:       a pointer to a HDF5 group.
        fh_dict: JSON config dictionary.
        layers:  a list of target layers.
    Raises:
        ValueError: in case of mismatch between provided layers
            and weights file.
    """
    original_keras_version = fh_dict.get('keras_version', '1')
    original_backend = fh_dict.get('backend', None)

    filtered_layers = []
    for layer in layers:
        weights = layer.weights
        if weights:
            filtered_layers.append(layer)

    layer_names = load_attributes_from_hdf5_group(fh_dict, f.name, 'layer_names')
    filtered_layer_names = []
    for name in layer_names:
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(fh_dict, g.name, 'weight_names')
        if weight_names:
            filtered_layer_names.append(name)
    layer_names = filtered_layer_names
    if len(layer_names) != len(filtered_layers):
        raise ValueError('You are trying to load a weight file '
                         'containing ' + str(len(layer_names)) +
                         ' layers into a model with ' + str(len(filtered_layers)) +
                         ' layers.')

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(fh_dict, g.name, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
        layer = filtered_layers[k]
        symbolic_weights = layer.weights
        weight_values = preprocess_weights_for_loading(
            layer, weight_values, original_keras_version, original_backend)
        if len(weight_values) != len(symbolic_weights):
            raise ValueError('Layer #' + str(k) + ' (named "' + layer.name +
                             '" in the current model) was found to '
                             'correspond to layer ' + name + ' in the save file. '
                             'However the new layer ' + layer.name + ' expects ' +
                             str(len(symbolic_weights)) +
                             ' weights, but the saved weights have ' +
                             str(len(weight_values)) + ' elements.')
        weight_value_tuples += zip(symbolic_weights, weight_values)
    K.batch_set_value(weight_value_tuples)

def save_attributes_to_hdf5_group(fh_dict, group_name, name, data):
    """Saves attributes (data) of the specified name into the HDF5 group.
    Actually, this revised version would not save the attribute to HDF5
    group but split the information into a JSON file. So it does not need
    to take into the header limit into consideration.
    Arguments:
        fh_dict:    JSON config dictionary.
        group_name: the name of band HDF5 group.
        name:       a name of the attributes to save.
        data:       attributes data to store.
    Raises:
        RuntimeError: if any single attribute is too large to be saved.
    """
    # Do not need to check heading limit anymore.
    gp_entry = fh_dict.get('group_attributes', None)
    if gp_entry is None:
        gp_entry = dict()
        fh_dict['group_attributes'] = gp_entry
    gp = gp_entry.get(group_name, None)
    if gp is None:
        gp = dict()
        gp_entry[group_name] = gp

    gp[name] = data

def load_attributes_from_hdf5_group(fh_dict, group_name, name):
    """Loads attributes of the specified name from the HDF5 group.
    This function would extracts all attributes from the JSON config
    dictionary instead of reading the head of HDF5 group.
    Arguments:
        fh_dict:    JSON config dictionary.
        group_name: the name of band HDF5 group.
        name:       a name of the attributes to load.
    Returns:
        data: attributes data.
    """
    gp_entry = fh_dict['group_attributes']
    gp = gp_entry[group_name]
    return gp[name]

def save_loss_weights_to_hdf5_group(f, fh_dict, loss_weights, compress=False):
    """Implements loss weight saving.
    This is the extension for implementing the saving session for loss
    weights. It will enable the save_model to save loss weights if they
    are compiled as variables. The parameter value would be dumped into
    hdf5 file and in the configuration, the variables are tagged by their
    names.
    Arguments:
        f:            a pointer to a HDF5 loss weights group.
        fh_dict:      JSON config dictionary.
        loss_weights: a list, or dictionary of loss weights.
        compress:     whether to compress the weights.
    Raises:
        ValueError: if the loss_weights is not list, tuple or dict.
    """
    compression = 'gzip' if compress else None
    cfg_entry = fh_dict['training_config']
    if loss_weights is None: # In None case, the 
        cfg_entry['loss_weights'] = None
        return

    # Serialize the loss weights, filter the constants and get variables.
    symbolic_weights = []
    weight_names = []
    i = 0
    if isinstance(loss_weights, (list, tuple)):
        serialized = []
        for value in loss_weights:
            if isinstance(value, variables.Variable):
                if hasattr(value, 'name') and value.name:
                    name = str(value.name)
                else:
                    name = 'loss_param_' + str(i)
                    i += 1
                serialized.append(name)
                if value not in symbolic_weights:
                    weight_names.append(name)
                    symbolic_weights.append(value)
            else:
                serialized.append(value)
    elif isinstance(loss_weights, dict):
        serialized = {}
        for key, value in loss_weights.items():
            if isinstance(value, variables.Variable):
                if hasattr(value, 'name') and value.name:
                    name = str(value.name)
                else:
                    name = 'loss_param_' + str(i)
                    i += 1
                serialized[key] = name
                if value not in symbolic_weights:
                    weight_names.append(name)
                    symbolic_weights.append(value)
            else:
                serialized[key] = value
    else:
        raise ValueError('The parameter loss_weights needs to be a list or a dictionary, maybe you need to recompile your model.')
    # Dump configurations
    cfg_entry['loss_weights'] = serialized
    # Dump variables into hdf5 set.
    weight_values = K.batch_get_value(symbolic_weights)
    for name, val in zip(weight_names, weight_values):
        param_dset = f.create_dataset(name, val.shape, dtype=val.dtype, compression=compression)
        if not val.shape:
            # scalar
            param_dset[()] = val
        else:
            param_dset[:] = val

def load_loss_weights_from_hdf5_group(f, fh_dict):
    """Implements loss weight loading.
    This is the extension for implementing the loading session for loss
    weights. It will enable the load_model to load loss weights if they
    are compiled as variables before saving. The variable names would
    be extracted from the config dictionary and the values would be
    extracted from the hdf5 dataset by indexing the variable names.
    Arguments:
        f:       a pointer to a HDF5 loss weights group.
        fh_dict: JSON config dictionary.
    Returns:
        loss_weights: a list, or dictionary of loss weights.
    Raises:
        ValueError: if the loss_weights is not list, tuple or dict.
        ValueError: if a saved variable has a wrong tag.
    """
    cfg_entry = fh_dict['training_config']
    loss_serialized = cfg_entry['loss_weights']
    if loss_serialized is None: # In None case, the 
        return None
    
    # Retain constant weights, and retrieve variables according to name tag.
    if isinstance(loss_serialized, (list, tuple)):
        loss_weights = []
        for value in loss_serialized:
            if isinstance(value, str):
                var = list(filter(lambda x:x.name==value, [op for op in variables.global_variables(scope=None)]))
                if var:
                    var = var[0]
                    K.set_value(var, value=np.asarray(f[value]))
                else:
                    if value[-2:] == ':0':
                        var = K.variable(value=np.asarray(f[value]), name=value[:-2])
                    else:
                        raise ValueError('The name of a variable in loss_weights should end with :0, because it is produced by K.variable.')
            else:
                var = value
            loss_weights.append(var)
    elif isinstance(loss_serialized, dict):
        loss_weights = {}
        for key, value in loss_serialized.items():
            if isinstance(value, str):
                var = list(filter(lambda x:x.name==value, [op for op in variables.global_variables(scope=None)]))
                if var:
                    var = var[0]
                    K.set_value(var, value=np.asarray(f[value]))
                else:
                    if value[-2:] == ':0':
                        var = K.variable(value=np.asarray(f[value]), name=value[:-2])
                    else:
                        raise ValueError('The name of a variable in loss_weights should end with :0, because it is produced by K.variable.')
            else:
                var = value
            loss_weights[key] = var
    else:
        raise ValueError('The parameter loss_weights needs to be a list or a dictionary, maybe you need to recompile your model.')

    return loss_weights