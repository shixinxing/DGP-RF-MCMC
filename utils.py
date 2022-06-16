import tensorflow as tf
import numpy as np
from layers import RBFLayer, GPLayer

class BNN_from_list(tf.Module):
    def __init__(self, layer_list, name=None):
        super(BNN_from_list, self).__init__(name=name)
        self.layers = layer_list

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def set_random_fixed(self, state):
        for layer in self.layers:
            if isinstance(layer, RBFLayer):
                assert hasattr(layer, 'random_fixed'), "Layers cannot set random_fixed!"
                layer.set_random_fixed(state)

class BNN_from_list_input_cat(BNN_from_list):
    def __init__(self, layer_list, name=None):
        super(BNN_from_list_input_cat, self).__init__(layer_list, name=name)

    def __call__(self, X):
        total_layers = len(self.layers)
        F = X
        for l, layer in enumerate(self.layers):
            if l == 0 or l == total_layers - 1:
                F = layer(F)
            else:
                if isinstance(layer, GPLayer):
                    F = layer(F)
                else: # input concatenate
                    F = tf.concat([F, X], axis=-1)
                    F = layer(F)
        return F

def log_gaussian(x, mean=0., var=1.):
    return - 0.5 * (tf.math.log(2. * np.pi) + tf.math.log(var) + tf.square(x - mean) / var)


def cyclical_lr_schedule(lr_0, k, cycle_length):
    """
    adjust learning rate according to cyclical schedule
    :param lr_0: initial learning rate;
    :param k: current epoch, from 0;
    :param cycle_length: Number of epoches in each cycle;
    """
    cos_inner = k % cycle_length
    cos_inner = tf.constant(np.pi) / cycle_length * cos_inner
    lr = lr_0 / 2. * (tf.math.cos(cos_inner) + 1.)
    return lr