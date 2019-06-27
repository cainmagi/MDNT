'''
################################################################
# Optimizers - Extended adaptive learning rate optimizers.
# @ Modern Deep Network Toolkits for Tensorflow-Keras
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6+
#   tensorflow r1.13+
# This module contains extended optimizers that are based on
# adaptive learning rate theory. Generally these optimizers
# could converge more quickly while the solution is easier to
# be overfitting.
# Version: 0.10 # 2019/6/27
# Comments:
#   Create this submodule, finish MNadam, Adabound and
#   Nadabound.
################################################################
'''

from tensorflow.python.framework import ops
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import gen_math_ops

class Nadabound(optimizers.Optimizer):
    """Nesterov Adabound optimizer
    The Nesterov version of the Adabound optimizer. This implementation is
    modified from mdnt.optimizers.Nadam and mdnt.optimizers.Adabound. Compared
    to Adabound optimizer, it uses estimated Nesterov gradient to update the
    momentum.
    Arguments:
        lr: float >= 0. Learning rate.
        lr_boost: float >=0. Suggest to > 1, because generally SGD optimizer
            requires a larger learning rate than Adam.
        gamma: float > 0. learning rate converging speed control factor.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
        sgdcorr: boolean. Because adam and SGD update momentum by different ways,
            when setting this flag True, the momentum updating rate would be
            approaching from 1. - beta_1 to 1. This correction is not applied in
            the original paper. Users should determine whether to use it carefully.
    """

    def __init__(self,
                 lr=0.002,
                 lr_boost=10.0,
                 gamma=1e-3,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=None,
                 decay=0.,
                 schedule_decay=0.004,
                 amsgrad=False,
                 sgdcorr=True,
                 **kwargs):
        super(Nadabound, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.m_schedule = K.variable(1., name='m_schedule')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.lr_boost = K.variable(lr_boost, name='lr_boost')
            self.gamma = K.variable(gamma, name='gamma')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.schedule_decay = schedule_decay
        self.amsgrad = amsgrad
        self.sgdcorr = sgdcorr

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [state_ops.assign_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * ( 1. / (1. + self.decay * math_ops.cast(self.iterations,K.dtype(self.decay))) )

        t = math_ops.cast(self.iterations, K.floatx()) + 1

        lower_bound = self.lr_boost * (1. - 1. / (self.gamma * t + 1.))
        upper_bound = self.lr_boost * (1. + 1. / (self.gamma * t))
        if self.sgdcorr:
            m_rate = 1. - self.beta_1 / (self.gamma * t + 1.)
        else:
            m_rate = 1. - self.beta_1

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (
            1. - 0.5 *
            (math_ops.pow(K.cast_to_floatx(0.96), t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (
            1. - 0.5 *
            (math_ops.pow(K.cast_to_floatx(0.96), (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        self.updates.append((self.m_schedule, m_schedule_new))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + m_rate * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * math_ops.square(g)
            if self.amsgrad:
                vhat_t = math_ops.maximum(vhat, v_t)
                self.updates.append(state_ops.assign(vhat, vhat_t))
                v_t_prime = vhat_t / (1. - math_ops.pow(self.beta_2, t))
            else:
                v_t_prime = v_t / (1. - math_ops.pow(self.beta_2, t))
            m_t_bar = (m_rate / (1.-self.beta_1)) * (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime
            beta_1_reduce = 1. - math_ops.pow(self.beta_1, t)
            lr_v = gen_math_ops.reciprocal((gen_math_ops.sqrt(v_t_prime) + self.epsilon) * beta_1_reduce)

            self.updates.append(state_ops.assign(m, m_t))
            self.updates.append(state_ops.assign(v, v_t))

            lr_bound = gen_math_ops.minimum(gen_math_ops.maximum(lr_v, lower_bound), upper_bound)
            p_t = p - lr * lr_bound * beta_1_reduce * m_t_bar
            
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'lr_boost': float(K.get_value(self.lr_boost)),
            'gamma': float(K.get_value(self.gamma)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'epsilon': self.epsilon,
            'decay': float(K.get_value(self.decay)),
            'schedule_decay': self.schedule_decay,
            'amsgrad': self.amsgrad,
            'sgdcorr': self.sgdcorr
        }
        base_config = super(Nadabound, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Adabound(optimizers.Optimizer):
    """Adabound optimizer.
    This optimizer would get initialized by an initial learning rate, a final
    learning rate and a converging speed control parameter, i.e. gamma which
    would define the upper bound and the lower bound of the adaptive learning
    rate. At the beginning, the optimizer behaves like Adam, but when its
    learning rate get converged, it would behaves like SGD+Momentum.
    The converging speed control function is defined as:
        lower_bound = final_lr * (1. - 1. / (self.gamma * t + 1.))
        upper_bound = final_lr * (1. + 1. / (self.gamma * t))
    The lower_bound would be (0.95 * final_lr) after (gamma * t = 19).
    The upper_bound would be (0.95 * final_lr) after (gamma * t = 20).
    Users need to specify proper gamma to make sure that the algorithm would not
    get converged too quickly.
    Arguments:
        lr: float >= 0. Learning rate.
        lr_boost: float >=0. Final learning rate (for SGD) is defined as:
            final_lr = lr * lr_boost.
        gamma: float > 0. learning rate converging speed control factor.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
        sgdcorr: boolean. Because adam and SGD update momentum by different ways,
            when setting this flag True, the momentum updating rate would be
            approaching from 1. - beta_1 to 1. This correction is not applied in
            the original paper. Users should determine whether to use it carefully.
    """

    def __init__(self,
                 lr=0.001,
                 lr_boost=10.0,
                 gamma=1e-3,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=None,
                 decay=0.,
                 amsgrad=False,
                 sgdcorr=True,
                 **kwargs):
        super(Adabound, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.lr_boost = K.variable(lr_boost, name='lr_boost')
            self.gamma = K.variable(gamma, name='gamma')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.sgdcorr = sgdcorr

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * ( 1. / (1. + self.decay * math_ops.cast(self.iterations,K.dtype(self.decay))) )

        with ops.control_dependencies([state_ops.assign_add(self.iterations, 1)]):
            t = math_ops.cast(self.iterations, K.floatx())
        lr_t = gen_math_ops.sqrt(1. - math_ops.pow(self.beta_2, t)) / (1. - math_ops.pow(self.beta_1, t))

        lower_bound = self.lr_boost * (1. - 1. / (self.gamma * t + 1.))
        upper_bound = self.lr_boost * (1. + 1. / (self.gamma * t))
        if self.sgdcorr:
            m_rate = 1. - self.beta_1 / (self.gamma * t + 1.)
        else:
            m_rate = 1. - self.beta_1

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + m_rate * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g)
            if self.amsgrad:
                vhat_t = math_ops.maximum(vhat, v_t)
                lr_v = gen_math_ops.reciprocal(gen_math_ops.sqrt(vhat_t) + self.epsilon)
                self.updates.append(state_ops.assign(vhat, vhat_t))
            else:
                lr_v = gen_math_ops.reciprocal(gen_math_ops.sqrt(v_t) + self.epsilon)

            lr_bound = gen_math_ops.minimum(gen_math_ops.maximum(lr_t * lr_v, lower_bound), upper_bound)
            p_t = p - lr * lr_bound * m_t

            self.updates.append(state_ops.assign(m, m_t))
            self.updates.append(state_ops.assign(v, v_t))
            
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'lr_boost': float(K.get_value(self.lr_boost)),
            'gamma': float(K.get_value(self.gamma)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'sgdcorr': self.sgdcorr
        }
        base_config = super(Adabound, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MNadam(optimizers.Optimizer):
    """Nesterov Adam optimizer (MDNT version)
    We use MNadam here to avoid the name conflict on tf.keras.optimizers.Nadam.
    Much like Adam is essentially RMSprop with momentum, Nadam is Adam RMSprop with
    Nesterov momentum. Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer at their default 
    values.
    This optimizer is modifed based on tf.keras.optimizers.Nadam. Compared to
    original implementation, this version supports two more things:
        1. Decay rate for the initial learning rate.
        2. Amsgrad option.
    Arguments:
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond".
    """

    def __init__(self,
                 lr=0.002,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=None,
                 decay=0.,
                 schedule_decay=0.004,
                 amsgrad=False,
                 **kwargs):
        super(MNadam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.m_schedule = K.variable(1., name='m_schedule')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.schedule_decay = schedule_decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [state_ops.assign_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * ( 1. / (1. + self.decay * math_ops.cast(self.iterations,K.dtype(self.decay))) )

        t = math_ops.cast(self.iterations, K.floatx()) + 1

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (
            1. - 0.5 *
            (math_ops.pow(K.cast_to_floatx(0.96), t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (
            1. - 0.5 *
            (math_ops.pow(K.cast_to_floatx(0.96), (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        self.updates.append((self.m_schedule, m_schedule_new))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * math_ops.square(g)
            if self.amsgrad:
                vhat_t = math_ops.maximum(vhat, v_t)
                self.updates.append(state_ops.assign(vhat, vhat_t))
                v_t_prime = vhat_t / (1. - math_ops.pow(self.beta_2, t))
            else:
                v_t_prime = v_t / (1. - math_ops.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

            self.updates.append(state_ops.assign(m, m_t))
            self.updates.append(state_ops.assign(v, v_t))

            p_t = p - lr * m_t_bar / (gen_math_ops.sqrt(v_t_prime) + self.epsilon)
            
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'epsilon': self.epsilon,
            'decay': float(K.get_value(self.decay)),
            'schedule_decay': self.schedule_decay,
            'amsgrad': self.amsgrad
        }
        base_config = super(MNadam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))