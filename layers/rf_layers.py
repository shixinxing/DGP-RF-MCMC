import tensorflow as tf
from kernels import RBFKernel, ARCKernel


class RBFLayer(tf.Module):
    def __init__(self, kernel, out_feature, random_fixed=True, set_nonzero_mean=False, name=None):
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
        # Do not use the following
        # self.amplitude = self.kernel.amplitude
        # self.inv_length_scale = self.kernel.inv_length_scale
        self.random_fixed = random_fixed
        if random_fixed: # when training
            self.z = tf.random.normal([self.in_feature, self.out_feature])
        self.set_nonzero_mean = set_nonzero_mean
        if set_nonzero_mean:
            self.mean = tf.Variable(tf.zeros([self.in_feature, 1]), dtype=tf.float32)
        else:
            self.mean = tf.zeros([self.in_feature, 1], dtype=tf.float32)

    def __call__(self, X):
        """
        :param X: [Batch-size, in_feature]
        :return: [Batch_size, n_rf ]
        """
        if self.random_fixed:
            if not self.kernel.is_ard:
                Omega  = self.kernel.inv_length_scale * self.z + self.mean
            else: # is ard
                Omega = tf.expand_dims(self.kernel.inv_length_scale, axis=-1) * self.z + self.mean
        else:  # not self.random_fixed:
            z_resampled = tf.random.normal([self.in_feature, self.out_feature])
            Omega = tf.expand_dims(self.kernel.inv_length_scale, axis=-1) * z_resampled + self.mean
        inner_prod = tf.linalg.matmul(X, Omega)
        random_feature = tf.concat([tf.math.cos(inner_prod), tf.math.sin(inner_prod)], axis=-1)
        random_feature = self.kernel.amplitude / tf.sqrt(tf.cast(self.out_feature, tf.float32)) * random_feature
        return  random_feature

    def set_random_fixed(self, state):
        self.random_fixed = state


class ARCLayer(tf.Module):
    def __init__(self, kernel, out_feature, random_fixed=True, set_nonzero_mean=False, name=None):
        """
        :param kernel: ARC-cosine Kernel class
        :param out_feature: Number of sampled \Omegas
        :param input_cat_dims: None: no input concatenate, otherwise it is the additional dimensions
        """
        super(ARCLayer, self).__init__(name=name)
        assert isinstance(kernel, ARCKernel), "Input kernel is not ARC!"
        self.kernel = kernel
        self.in_feature = kernel.n_feature
        self.out_feature = out_feature
        self.n_rf = out_feature
        # Do not use the following
        # self.amplitude = self.amplitude
        # self.inv_length_scale = self.inv_length_scale
        self.random_fixed = random_fixed
        if random_fixed: # when training
            self.z = tf.random.normal([self.in_feature, self.out_feature])
        if set_nonzero_mean:
            self.mean = tf.Variable(tf.zeros([self.in_feature, 1]), dtype=tf.float32, name='mean')
        else:
            self.mean = tf.zeros([self.in_feature, 1], dtype=tf.float32)

    def __call__(self, X):
        """
        :param X: 【Batch-size, in_feature】
        :return: [Batch_size, n_rf ]
        """
        if self.random_fixed:
            if not self.kernel.is_ard:
                Omega  = self.kernel.inv_length_scale * self.z + self.mean
            else: # is ard
                Omega = tf.expand_dims(self.kernel.inv_length_scale, axis=-1) * self.z + self.mean
        else:  # not self.random_fixed:
            z_resampled = tf.random.normal([self.in_feature, self.out_feature])
            Omega = tf.expand_dims(self.kernel.inv_length_scale, axis=-1) * z_resampled + self.mean
        inner_prod = tf.linalg.matmul(X, Omega)
        random_feature = tf.nn.relu(inner_prod)
        random_feature = tf.sqrt(2.) * self.kernel.amplitude / tf.sqrt(tf.cast(self.out_feature, tf.float32)) * random_feature
        return  random_feature

    def set_random_fixed(self, state):
        self.random_fixed = state
