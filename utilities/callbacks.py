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
# Version: 0.23 # 2019/10/27
# Comments:
#   Enable ModelCheckpoint to use compression to save models.
# Version: 0.22 # 2019/10/23
# Comments:
#   Enable ModelCheckpoint to not save optimizer.
# Version: 0.20 # 2019/10/15
# Comments:
#   Finish LossWeightsScheduler.
# Version: 0.18 # 2019/6/24
# Comments:
# 1. Finish ModelWeightsReducer.
# 2. Fix bugs for ModelWeightsReducer.
# 3. Find a better way for implementing the soft thresholding
#    for ModelWeightsReducer.
# Version: 0.16 # 2019/6/23
# Comments:
#   Add OptimizerSwitcher and fix a bug.
# Version: 0.10 # 2019/6/13
# Comments:
#   Create this submodule, and finish ModelCheckpoint.
################################################################
'''

from datetime import datetime
import os
import numpy as np
from tensorflow.python.ops import variables
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import backend as K
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import gen_math_ops

from . import _default

class LossWeightsScheduler(callbacks.Callback):
    """Learning rate scheduler.
    Arguments:
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            loss weights as output.
        verbose: int. 0: quiet, 1: update messages.
    Here we show two examples:
    ```python
    # This function is designed for a two-phase training. In the
    # first phase, the learning rate is (0.8, 0.2);
    # In the second phase, the learning rate is 
    # (0.2, 0.8);
    def scheduler(epoch):
        if epoch < 10:
            return [0.8, 0.2]
        else:
            return [0.2, 0.8]
    model.compile(..., loss_weights=[K.variable(0.5),
                  K.variable(0.5)])
    callback = mdnt.utilities.callbacks.LossWeightsScheduler(scheduler)
    model.fit(data, labels, epochs=100, callbacks=[callback],
              validation_data=(val_data, val_labels))
    ```
    ```python
    # This function is designed for a two-phase training. In the
    # first phase, the learning rate is (alpha=0.8, beta=0.2);
    # In the second phase, the learning rate is 
    # (alpha=0.2, beta=0.8);
    def scheduler(epoch):
        if epoch < 10:
            return {'alpha':0.8, 'beta':0.2}
        else:
            return {'alpha':0.2, 'beta':0.8}
    model.compile(..., loss_weights={'alpha':K.variable(0.5), 
                  'beta':K.variable(0.5)})
    callback = mdnt.utilities.callbacks.LossWeightsScheduler(scheduler)
    model.fit(data, labels, epochs=100, callbacks=[callback],
              validation_data=(val_data, val_labels))
    ```
    """

    def __init__(self, schedule, verbose=0):
        super(LossWeightsScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model, 'loss_weights'):
            raise ValueError('Model must have a "loss_weights" attribute.')
        lw = self.model.loss_weights
        if lw is None:
            raise ValueError('model.loss_weights needs to be set.')
        lw_val = self.schedule(epoch) # Get losses
        if isinstance(lw, dict):
            if not isinstance(lw_val, dict):
                raise ValueError('model.loss_weights is a dict, you need to '
                                 'provides a corresponding dict for updating it.')
            for k, v in lw:
                if isinstance(v, variables.Variable):
                    K.set_value(v, lw_val[k])
        elif isinstance(lw, (list, tuple)):
            if not isinstance(lw_val, (list, tuple, np.ndarray)):
                raise ValueError('model.loss_weights is a sequence, you need to '
                                 'provides a corresponding sequence for updating it.')
            s = 0
            for v in lw:
                if isinstance(v, variables.Variable):
                    K.set_value(v, lw_val[s])
                    s += 1
        else:
            raise ValueError('model.loss_weights could not be updated, please check'
                             'your definition.')
        if self.verbose > 0:
            print('\nEpoch %05d: LossWeightsScheduler set var.lw to %s.' % (epoch + 1, lw_val))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lw = self.model.loss_weights
        lw_var = None
        if isinstance(lw, dict):
            lw_var = {}
            for k, v in lw:
                if isinstance(v, variables.Variable):
                    lw_var[k] = K.get_value(v)
                else:
                    lw_var[k] = v
        elif isinstance(lw, (list, tuple)):
            lw_var = []
            for v in lw:
                if isinstance(v, variables.Variable):
                    lw_var.append(K.get_value(v))
                else:
                    lw_var.append(v)
        logs['loss_weights'] = lw_var

class ModelWeightsReducer(callbacks.Callback):
    """Model weights reducer
    Insert a weight decay operation before each iteration during the training.
    When it is applied to pure SGD, this callback is equivalent to adding
    L1/L2 regularization to each kernel.
    However, the optimizer with momentum or adaptive learning rate would make
    the regularization terms not equivalent to weight decay. As an alternative,
    Tensorflow provides AdamW (weight decayed Adam) in contribution module.
    This callback serves as an alternative for using weight decayed optimizers.
    For example, using ModelWeightsReducer(mu=0.1) + Adam is equivalent to
    using AdamW(weight_decay=0.1).
    This callback provides both soft threshold method and weight decay method,
    which are used for maintained the sparsity and small module length respec-
    tively. Compared to adding regularization terms, this callback does not
    get influenced by a specific optimizing algorithm.
    Arguments:
        lam: proximal coefficient. It is used to apply soft thresholding and
            maintain the sparsity of all kernels.
            It only take effects when > 0.0.
        mu: Tikhonov coefficient. It is used to apply the weight decay method
            and maintain the reduced length of the weight module.
            It only take effects when > 0.0.
    """
    def __init__(self, lam=0.0, mu=0.0, epsilon=1e-5):
        with K.name_scope(self.__class__.__name__):
            self.get_lambda = K.variable(lam, name='lambda')
            self.get_mu = K.variable(mu, name='mu')
        self.bool_l1 = lam > 0.0
        self.bool_l2 = mu > 0.0
        self.session = None
        if not (self.bool_l1 or self.bool_l2):
            raise ValueError('Need to specify either one of "lam" and "mu".')
    
    def on_train_begin(self, logs=None):
        # First collect all trainable weights
        self.model._check_trainable_weights_consistency()
        get_w_list = self.model.trainable_weights
        get_w_dec_list = []
        # Filter all weights and select those named 'kernel'
        for w in get_w_list:
            getname = w.name
            pos = getname.rfind('/')
            if pos != -1:
                checked = 'kernel' in getname[pos+1:]
            else:
                checked = 'kernel' in getname
            if checked:
                get_w_dec_list.append(w)
        if not get_w_dec_list:
            raise ValueError('The trainable weights of the model do not include any kernel.')
        # Define the update ops
        getlr = self.model.optimizer.lr
        with K.name_scope(self.__class__.__name__):
            self.w_updates = []
            self.w_updates_aft = []
            for w in get_w_dec_list:
                w_l = w
                if self.bool_l2:
                    w_l = (1 - getlr * self.get_mu) * w_l
                if self.bool_l1:
                    w_abs = math_ops.abs(w_l) + self.get_lambda
                    w_l = ( gen_math_ops.sign(w_l) + gen_math_ops.sign(random_ops.random_uniform(w_l.get_shape(), minval=-1.0, maxval=1.0)) * math_ops.cast(gen_math_ops.equal(w_l, 0), dtype=w_l.dtype) ) * w_abs
                    w_abs_x = math_ops.abs(w) - self.get_lambda
                    w_x = gen_math_ops.sign(w) * math_ops.cast(gen_math_ops.greater(w_abs_x, 0), dtype=w.dtype) * w_abs_x
                    self.w_updates_aft.append(state_ops.assign(w, w_x))
                self.w_updates.append(state_ops.assign(w, w_l))
        # Get and store the session
        self.session = K.get_session()

    def on_train_end(self, logs=None):
        self.session = None

    def on_train_batch_begin(self, batch, logs=None):
        # Define the updating function
        self.session.run(fetches=self.w_updates)

    def on_train_batch_end(self, batch, logs=None):
        if self.bool_l1:
            self.session.run(fetches=self.w_updates_aft)

class OptimizerSwitcher(callbacks.Callback):
    """Optimizer switcher
    Need to use with MDNT optimizers that support mannual phase-switching
    method `optimizer.switch()`. 
    Now such optimizers include:
        mdnt.optimizers.Adam2SGD
        mdnt.optimizers.NAdam2NSGD
    Arguments:
        switch_epochs: an int or an int list which determines when to switch
            the optimizer phase. The switch would happens on the end of 
            assigned epochs. Should start with 1 (the first epoch).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, switch_epochs, verbose=0):
        super(OptimizerSwitcher, self).__init__()
        if isinstance(switch_epochs, (list, tuple)):
            if all(type(i)==int for i in switch_epochs):
                self.switch_epochs = list(switch_epochs)
            else:
                raise ValueError('The input list switch_epochs should only contains int elements.')
        else:
            if type(switch_epochs) != int:
                raise ValueError('The input scalar switch_epochs should be an int element.')
            self.switch_epochs = [switch_epochs]
        self.switch_epochs.sort(reverse=True)
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        if not callable(getattr(self.model.optimizer, 'switch')):
            raise ValueError('Optimizer must have a "switch" method to support manually switching the training phase.')
        popflag = False
        while self.switch_epochs and self.switch_epochs[-1] < 1:
            self.switch_epochs.pop()
            popflag = True
        if popflag and self.verbose > 0:
            print('The input switch_epochs is revised as {0}.'.format(self.switch_epochs))

    def on_epoch_end(self, epoch, logs=None, mode='train'):
        if mode == 'train' and self.switch_epochs:
            if self.switch_epochs[-1] == (epoch + 1):
                self.model.optimizer.switch(None)
                if self.verbose > 0:
                    print('\nEpoch {0:05d}: Optimizer switcher switches the optimizer phase'.format(epoch + 1))
                self.switch_epochs.pop()

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
        save_optimizer: If `save_optimizer=True`, the optimizer configu-
            rations would be dumped as a json file.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        compress: whether to apply the compression for saving models.
            this option is only avaliable when save_weights_only=False.
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
                 save_optimizer=True,
                 save_best_only=False,
                 save_weights_only=False,
                 compress=True,
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
        self.save_optimizer = save_optimizer
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.compress = compress
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
                            _default.save_model(self.model, weightpath, configpath, optmpath, overwrite=True, include_optimizer=self.save_optimizer, compress=self.compress)
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
                    _default.save_model(self.model, weightpath, configpath, optmpath, overwrite=True, include_optimizer=self.save_optimizer, compress=self.compress)
                    #self.model.save(filepath, overwrite=True)