import numpy as np
import tensorflow as tf
# import tensorflow_probability as tfp

class RBFKernel(tf.Module):
    def __init__(self, n_feature=1, amplitude=1., length_scale=None, trainable=True, is_ard=False, name=None):
        """
        k(x, y) = amplitude**2 * exp(-||x - y||**2 / (2 * length_scale**2))
        :param n_feature: the last dim of input
        :param length_scale: can be a scalar or vector
        """
        super(RBFKernel,self).__init__(name=name)
        self.kernel_type = "RBF"
        self.n_feature = n_feature
        # initialize length scale to \sqrt(d_in)
        if length_scale is None:
            length_scale = tf.cast(n_feature, tf.float32) ** 0.5

        if tf.rank(length_scale) >= 2: # not a vector
            raise ValueError("The length scale of RBF dim error!")
        inv_length_scale = 1. / tf.constant(length_scale)
        if tf.rank(inv_length_scale) == 0 and is_ard: # scalar
            inv_length_scale = inv_length_scale * np.ones(n_feature)
            self.is_ard = is_ard
        elif tf.rank(inv_length_scale) == 1: #vector
            if n_feature != tf.size(inv_length_scale):
                raise ValueError("The size of length scale and features do not match!")
            else:
                self.is_ard = True
        else:
            self.is_ard = False

        self.log_amplitude = tf.Variable(tf.math.log(amplitude), trainable=trainable, name="log_amplitude")
        self.log_inv_length_scale = tf.Variable(tf.math.log(inv_length_scale), trainable=trainable,
                                                dtype=tf.float32, name="log_inv_length_scale")

    @property
    def amplitude(self):
        return tf.math.exp(self.log_amplitude)

    @property
    def length_scale(self):
        return 1. / self.inv_length_scale

    @property
    def inv_length_scale(self):
        return tf.math.exp(self.log_inv_length_scale)
