import tensorflow as tf
import numpy as np
from layers import RBFLayer, ARCLayer, GPLayer

class BNN_from_list(tf.Module):
    def __init__(self, layer_list, name=None):
        super(BNN_from_list, self).__init__(name=name)
        self.layers = layer_list

    def __call__(self, X, allow_gradient_from_W=True):
        for layer in self.layers:
            if isinstance(layer, GPLayer):
                X = layer(X, allow_gradient_from_W=allow_gradient_from_W)
            else:
                X = layer(X)
        return X

    def set_random_fixed(self, state):
        for layer in self.layers:
            if isinstance(layer, RBFLayer) or isinstance(layer, ARCLayer):
                assert hasattr(layer, 'random_fixed'), "Layers cannot set random_fixed!"
                layer.set_random_fixed(state)
    @property
    def gp_layers(self):
        return (self.layers[2*l+1] for l in range(len(self.layers)//2))


class BNN_from_list_input_cat(BNN_from_list):
    def __init__(self, layer_list, name=None):
        super(BNN_from_list_input_cat, self).__init__(layer_list, name=name)

    def __call__(self, X, allow_gradient_from_W=True):
        total_layers = len(self.layers)
        F = X
        for l, layer in enumerate(self.layers):
            if l == 0 or l == total_layers - 1:
                F = layer(F)
            else:
                if isinstance(layer, GPLayer):
                    F = layer(F, allow_gradient_from_W=allow_gradient_from_W)
                else: # input concatenate
                    F = tf.concat([F, X], axis=-1)
                    F = layer(F)
        return F

def log_gaussian(x, mean=0., var=1.):
    return - 0.5 * (tf.math.log(2. * np.pi) + tf.math.log(var) + tf.square(x - mean) / var)

def cyclical_step_rate(step_index, cycle_length, schedule='cosine', min_value=0.001):
    """
    adjust step rate in MCMC
    :param step_index: current step (i.e. batch index), from 1;
    :param cycle_length: Number of iterations in each cycle;
    :param schedule: "cosine", "glide", or "flat"
    :param min_value: float, the minimum rate returned by this method.
    """
    if step_index <= 0:
        raise ValueError('Step index should be larger than zero!')

    frac = tf.cast(tf.math.mod(step_index - 1, cycle_length),
                   dtype=tf.float32) / tf.cast(cycle_length, tf.float32)
    if schedule == 'cosine':
        step_rate = min_value + (1.0 - min_value) * 0.5 * (tf.math.cos(np.pi * frac) + 1.0)
    elif schedule == 'glide':
        step_rate = min_value + (1.0 - min_value) * (tf.math.exp(-frac / (1.0 - frac)))
    elif schedule == 'flat':
        step_rate = 1.0
    else:
        raise NotImplementedError

    is_end_of_period_iteration = tf.equal(tf.math.mod(step_index, cycle_length), 0)

    return step_rate, is_end_of_period_iteration

