'''
################################################################
# Utilities - Callbacks
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# Extend loss functions. These functions could serve as both
# losses and metrics.
# Version: 0.10 # 2019/6/13
# Comments:
#   Create this submodule, and finish linear_jaccard_loss
#   and lovasz_jaccard_loss.
################################################################
'''

from datetime import datetime
import os
import numpy as np
from tensorflow.python.keras import callbacks
from tensorflow.python.platform import tf_logging as logging

from . import _default

class ModelCheckpoint(callbacks.Callback):
    """Save the model after every epoch. (Revised)
    Revised Model checkpointer. Compared to original version, it supports
    such new features:
        1. When `save_weights_only` is set `False`, it uses the MDNT version
           of model saver and avoid the heading excessing problem of saving
           HDF5 file.
        2. The model configurations and the network weights are splitted.
           It will be easier for user to see the configuration through the
           saved JSON file.
        3. When setting `keep_max`, only recent weights would be retained.
    Now `filepath` should not contain named formatting options, because
    the format options are moved into `record_format`. The final output
    configuration file name should be:
        `filapath + '.json'`
    while the weights file name should be:
        `filepath + '-' + record_format.format(...) + '.h5'`
    For example, if `filepath` is `'model'` while `record_format` is
    `'e{epoch:02d}_v{val_loss:.2f}'`, the latter part will be filled the 
    value of `epoch` and keys in `logs` (passed in `on_epoch_end`). The 
    output may be like:
        `'model.json'` and `'model-e05_v0.33.h5'`.
    Then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    Arguments:
        filepath: string, path to save the model file.
        record_format: the format of the using records. If set None, it
            would be set as a time stamp.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        keep_max: the maximum of kept weight file during the training
            phase. If set None, all files would be kept. This option
            requires users to have the authority to delete files in the 
            saved path.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self,
                 filepath,
                 record_format=None,
                 monitor='val_loss',
                 verbose=0,
                 keep_max=None,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.record_format = record_format
        if set('{}%').issubset(set(self.filepath)):
            raise TypeError('filepath should not contains formats anymore. Use `record_format` to define that part.')
        self.keep_max = keep_max
        if keep_max is not None:
            self.__keep_list = []
            self.__current_num = 0
        else:
            self.__keep_list = None
            self.__current_num = None
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            logging.warning('ModelCheckpoint mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
    
    def __keep_max_function(self, new_file_names):
        if self.keep_max is None:
            return
        if self.__current_num < self.keep_max:
            self.__current_num += 1
            self.__keep_list.append(new_file_names)
        else:
            old_file_names = self.__keep_list.pop(0)
            for old_file_name in old_file_names:
                if os.path.exists(old_file_name):
                    os.remove(old_file_name)
            self.__keep_list.append(new_file_names)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            configpath = self.filepath + '.json'
            if self.record_format:
                weightpath = self.filepath + '-' + self.record_format
                weightpath = weightpath.format(epoch=epoch + 1, **logs)
            else:
                weightpath = self.filepath + datetime.timestamp(datetime.now())
            optmpath = weightpath + '.json'
            weightpath = weightpath + '.h5'
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('Can save best model only with %s available, '
                                                    'skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                  current, weightpath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(weightpath, overwrite=True)
                        else:
                            self.__keep_max_function((weightpath, optmpath))
                            _default.save_model(self.model, weightpath, configpath, optmpath, overwrite=True)
                            #self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, weightpath))
                if self.save_weights_only:
                    self.model.save_weights(weightpath, overwrite=True)
                else:
                    self.__keep_max_function((weightpath, optmpath))
                    _default.save_model(self.model, weightpath, configpath, optmpath, overwrite=True)
                    #self.model.save(filepath, overwrite=True)