import tensorflow as tf
from utils import log_gaussian


class BNNTestBase(tf.Module):
    def __init__(self):
        super(BNNTestBase, self).__init__()

    def __call__(self, X):
        raise NotImplementedError

    def prior_W(self):
        """
        :return: \log p(W) ~ N(0, I)
        """
        log_p_W = 0.
        for param in self.trainable_variables:
            log_p_W += tf.reduce_sum(log_gaussian(param, mean=0., var=1.))
        return log_p_W

    def log_likelihood(self, X_batch, Y_batch):
        F = self.__call__(X_batch)
        Y = tf.cast(Y_batch, tf.int32) #[N, 1]
        return - tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y[:, 0], logits=F)

    def U(self, X_batch, Y_batch, data_size):
        batch_size = tf.shape(X_batch)[0]
        batch_size = tf.cast(batch_size, tf.float32)
        data_size = tf.cast(data_size, tf.float32)
        log_prior_sum = self.prior_W() / data_size
        log_likelihood = tf.reduce_sum(self.log_likelihood(X_batch, Y_batch)) / batch_size
        return - (log_prior_sum + log_likelihood)

    def sgmcmc_update(self, X_batch, Y_batch, data_size, lr=0.01, beta=0.95, temperature=1.):
        """
        :param lr: here the learning rate is the instantaneous effect of the average-minibatch-gradient
        on the parameter. Must be positive and typically is in a range from 1.0e-6 to 0.1.
        :param beta: momentum_decay, in the range [0,1),
        typically close to 1, e.g. values such as 0.75, 0.8, 0.9, 0.95.
        """
        with tf.GradientTape() as tape:
            U = self.U(X_batch, Y_batch, data_size)
        grads = tape.gradient(U, self.trainable_variables)

        h = tf.sqrt(lr / data_size)
        for param, grad in zip(self.trainable_variables, grads):
            assert hasattr(param, "auxiliary_m"), "Trainable Params do not have attr auxiliary_m!"
            m_new = beta * param.auxiliary_m - h * data_size * grad
            eps = tf.random.normal(tf.shape(param))
            m_new = m_new + (2.0 * (1. - beta) * temperature) ** 0.5 * tf.sqrt(param.precond_M) * eps
            param.auxiliary_m = m_new
            param.assign_add(h * tf.math.reciprocal(param.precond_M) * m_new)

    def precond_update(self, ds, data_size, K_batches=32, precond_type='adagrad', rho_rms=0.99):
        """
        :param ds: tf.data.Dataset as Iterable
        :param data_size: N
        :param K_batches: K number of mini-batches
        :return: preconditioner M w.r.t each param
        """
        if precond_type == 'identity':
            for param in self.trainable_variables:
                if not hasattr(param, "precond_M"):
                    param.precond_M = tf.constant(1., dtype=tf.float32)
                if not  hasattr(param, "auxiliary_m"):
                    # param.auxiliary_m = tf.zeros_like(param)
                    param.auxiliary_m = tf.random.normal(tf.shape(param))
        else:
            for param in self.trainable_variables:
                if not hasattr(param, "m_c"):
                    param.m_c = None
                if not hasattr(param, "precond_M"):
                    param.precond_M = tf.ones_like(param, dtype=tf.float32)
                if not hasattr(param, "auxiliary_m"):
                    # param.auxiliary_m = tf.zeros_like(param)
                    param.auxiliary_m = tf.random.normal(tf.shape(param))
                param.m_c = tf.math.rsqrt(param.precond_M) * param.auxiliary_m

            DEFAULT_REGULARIZATION = 1.0e-7
            k = 0
            for X_batch, Y_batch in ds:
                with tf.GradientTape() as tape:
                    U = self.U(X_batch, Y_batch, data_size)
                grads = tape.gradient(U, self.trainable_variables)

                for param, grad in zip(self.trainable_variables, grads):
                    if not hasattr(param, 'v_sum'):
                        param.v_sum = tf.zeros_like(param, dtype=tf.float32)
                    if precond_type == 'adagrad':
                        param.v_sum = param.v_sum + tf.square(grad)
                    elif precond_type == 'rmsprop':
                        param.v_sum = rho_rms * param.v_sum + (1.- rho_rms) * tf.square(grad)
                k = k + 1
                if k >= K_batches:
                    break

            sigma_min = None
            for param in self.trainable_variables:
                sigma = param.v_sum / K_batches + DEFAULT_REGULARIZATION
                sigma = tf.sqrt(sigma)
                tmp = tf.reduce_min(sigma)
                if sigma_min is None:
                    sigma_min = tmp
                elif tmp < sigma_min:
                    sigma_min = tmp
                if not hasattr(param, "sigma"):
                    param.sigma = None
                param.sigma = sigma

            for param in self.trainable_variables:
                param.precond_M = param.sigma / sigma_min
                param.auxiliary_m = tf.math.sqrt(param.precond_M) * param.m_c


class BNNTest(BNNTestBase):
    def __init__(self):
        super(BNNTest, self).__init__()
        self.w1 = tf.Variable(tf.random.normal([28*28, 1000]))
        self.b1 = tf.Variable(tf.random.normal([1000]))
        self.w2 = tf.Variable(tf.random.normal([1000, 100]))
        self.b2 = tf.Variable(tf.random.normal([100]))
        self.w3 = tf.Variable(tf.random.normal([100,10]))
        self.b3 = tf.Variable(tf.random.normal([10]))


    def __call__(self, X):
        y = tf.nn.relu(tf.matmul(X, self.w1) + self.b1)
        y = tf.nn.relu(tf.matmul(y, self.w2) + self.b2)
        y = tf.matmul(y, self.w3) + self.b3
        return y


class BNNTestCos(BNNTestBase):
    def __init__(self):
        super(BNNTestCos, self).__init__()
        n_rf = 1000
        self.w1 = tf.Variable(tf.random.normal([28 * 28, n_rf]))
        self.w2 = tf.Variable(tf.random.normal([2*n_rf, n_rf]))
        self.w3 = tf.Variable(tf.random.normal([2*n_rf, n_rf]))
        self.w4 = tf.Variable(tf.random.normal([2*n_rf, 10]))

    def __call__(self, X):
        y = tf.matmul(X, self.w1)
        y = tf.concat([tf.math.cos(y), tf.math.sin(y)], axis=-1)
        y = tf.matmul(y, self.w2)
        y = tf.concat([tf.math.cos(y), tf.math.sin(y)], axis=-1)
        y = tf.matmul(y, self.w3)
        y = tf.concat([tf.math.cos(y), tf.math.sin(y)], axis=-1)
        y = tf.matmul(y, self.w4)
        return y






