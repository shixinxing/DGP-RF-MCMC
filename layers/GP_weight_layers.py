import tensorflow as tf


class GPLayer(tf.Module):
    def __init__(self, in_feature, out_feature, name=None):
        super(GPLayer, self).__init__(name=name)
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.W = tf.Variable(tf.random.normal([in_feature, out_feature], mean=0., stddev=1.), name='GP_layer_W')

    def __call__(self, X, allow_gradient_from_W=True):
        if allow_gradient_from_W:
            return tf.matmul(X, self.W)
        else: # when W is from assigned value, regard it as constant to prevent gradient tape breaking
            return tf.matmul(X, tf.stop_gradient(self.W))

    def assign_W(self, W_value):
        if isinstance(W_value, tf.Variable):
            W_value = W_value.numpy()
        self.W.assign(W_value)
