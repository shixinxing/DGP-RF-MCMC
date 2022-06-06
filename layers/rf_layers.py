import tensorflow as tf
from kernels import RBFKernel


class RBFLayer(tf.Module):
    def __init__(self, kernel, out_feature, random_fixed=True, name=None):
        """
        :param kernel: RBFKernel class
        :param out_feature: Number of sampled \Omegas
        """
        super(RBFLayer, self).__init__(name=name)
        assert isinstance(kernel, RBFKernel), "Input kernel is not RBF!"
        self.kernel = kernel

        self.in_feature = kernel.n_feature
        self.out_feature = out_feature
        self.n_rf = 2 * out_feature
        self.amplitude = kernel.amplitude
        self.inv_length_scale = kernel.inv_length_scale

        self.random_fixed = random_fixed
        if random_fixed: # when training
            self.z = tf.random.normal([self.in_feature, self.out_feature])

    def __call__(self, X):
        """
        :param X: 【Batch-size, in_feature】
        :return: [Batch_size, n_rf ]
        """
        if self.random_fixed:
            if not self.kernel.is_ard:
                Omega  = self.inv_length_scale * self.z
            else: # is ard
                Omega = tf.expand_dims(self.inv_length_scale, axis=-1) * self.z
        else:  # not self.random_fixed:
            z_resampled = tf.random.normal([self.in_feature, self.out_feature])
            Omega = tf.expand_dims(self.inv_length_scale, axis=-1) * z_resampled
        inner_prod = tf.linalg.matmul(X, Omega)
        random_feature = tf.concat([tf.math.cos(inner_prod), tf.math.sin(inner_prod)], axis=-1)
        random_feature = self.amplitude / tf.sqrt(tf.cast(self.out_feature, tf.float32)) * random_feature
        return  random_feature

    def set_random_fixed(self, state):
        self.random_fixed = state


# class ARCCosLayer(tf.Module):
#     def __init__(self, kernel, ):