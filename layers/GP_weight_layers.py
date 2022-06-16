import tensorflow as tf


class GPLayer(tf.Module):
    def __init__(self, in_feature, out_feature, name=None):
        super(GPLayer, self).__init__(name=name)
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.W = tf.Variable(tf.random.normal([in_feature, out_feature]), name='GP_layer_W')

    def __call__(self, X):
        return tf.matmul(X, self.W)
