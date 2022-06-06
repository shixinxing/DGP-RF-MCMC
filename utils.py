import tensorflow as tf
import numpy as np

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
            if hasattr(layer, 'random_fixed'):
                layer.set_random_fixed(state)


def log_gaussian(x, mean=0., var=1.):
    return - 0.5 * (tf.math.log(2. * np.pi) + tf.math.log(var) + tf.square(x - mean) / var)