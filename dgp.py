import tensorflow as tf
from likelihoods import Softmax, Gaussian
from kernels import RBFKernel
from layers import RBFLayer, GPLayer
from utils import BNN_from_list, log_gaussian


class DGP_RF(tf.Module):
    def __init__(self, d_in, d_out, n_hidden_layers=1, n_rf=20, n_gp=2, likelihood=Softmax(),
                 kernel_list=None, random_fixed=True, name=None):
        """
        :param d_in: Input dim
        :param d_out: Output dim
        :param n_hidden_layers: Number of hidden layers
        :param n_rf: Number of random features
        :param n_gp: Number of latent GPs in each layer
        :param likelihood: Likelihood class in the last layer
        :param kernel: determine is_ard via the length scale params
        :param random_fixed: z fixed or not when feeding forward
        """
        super(DGP_RF, self).__init__(name=name)
        self.d_in = d_in
        self.d_out = d_out
        self.n_hidden_layers = n_hidden_layers
        if tf.rank(n_rf) == 0:
            self.n_rf = n_rf * tf.ones([n_hidden_layers], dtype=tf.int32) #[20, 20]
        else:
            self.n_rf = tf.constant(n_rf)
        if tf.rank(n_gp) == 0:
            self.n_gp = n_gp * tf.ones([n_hidden_layers], dtype=tf.int32) #[2, 2]
        else:
            self.n_gp = tf.constant(n_gp)
        self.likelihood = likelihood
        if kernel_list is None:
            before_n_rf = tf.concat([[d_in], self.n_gp[:-1]], axis=0)
            self.kernel_list = [ RBFKernel(n_feature=before_n_rf[i], trainable=True, is_ard=True) for i in range(n_hidden_layers) ]
        else:
            self.kernel_list = kernel_list
        self.random_fixed = random_fixed
        self.BNN = self.transformed_BNN()

    def transformed_BNN(self):
        """
        :return: corresponding transformed BNN as an attribute
        """
        bnn = []
        for l in range(self.n_hidden_layers):
            kernel_tmp = self.kernel_list[l]
            if kernel_tmp.kernel_type == "RBF":
                layer_Omega = RBFLayer(kernel_tmp, self.n_rf[l], random_fixed=self.random_fixed)
                layer_GP = GPLayer(2 * self.n_rf[l], self.n_gp[l])
                bnn.extend([layer_Omega, layer_GP])
            else:
                raise  NotImplementedError
        return BNN_from_list(bnn)

    def log_likelihood(self, X, Y):
        """
        Compute log likelihood given all params \log p(D|all params) by feeding forward
        :param X: [N, D]
        :param Y: [N, D_out]
        :return: [N, ]
        """
        F = self.BNN(X)
        log_likelihood = self.likelihood.log_prob(F, Y)
        return log_likelihood

    def prior_W(self):
        """
        :return: \log p(W) ~ N(0, I)
        """
        log_p_W = 0.
        for l in range(self.n_hidden_layers):
            gp_layer = self.BNN.layers[2 * l + 1]
            w_l = gp_layer.trainable_variables #return a tuple
            for var in w_l:
                log_p_W += tf.reduce_sum(log_gaussian(var, mean=0., var=1.))
        return log_p_W

    def prior_kernel_params(self):
        log_p_params = 0.
        for l in range(self.n_hidden_layers):
            omega_layer = self.BNN.layers[2 * l]

            log_amp = omega_layer.kernel.log_amplitude
            log_p_params += tf.reduce_sum(log_gaussian(log_amp, mean=0., var=1.))
            log_inv_length = omega_layer.kernel.log_inv_length_scale
            log_p_params += tf.reduce_sum(log_gaussian(log_inv_length, mean=0., var=1.))
        return log_p_params

    def prior_likelihood_params(self):
        if isinstance(self.likelihood, Softmax):
            return 0.
        elif isinstance(self.likelihood, Gaussian):
            likelihood_params = self.likelihood.trainable_variables
            log_p_params = 0.
            for var in likelihood_params:
                log_p_params += tf.reduce_sum(log_gaussian(var, mean=0., var=1.))
            return log_p_params

    def U(self, X_batch, Y_batch, data_size):
        batch_size = tf.shape(X_batch)[0]
        batch_size = tf.cast(batch_size, tf.float32)
        data_size = tf.cast(data_size, tf.float32)
        # log_prior_sum = (self.prior_W() + self.prior_kernel_params() + self.prior_likelihood_params()) / data_size
        log_prior_sum = 0.
        for param in self.trainable_variables:
            log_prior_sum += tf.reduce_sum(log_gaussian(param, mean=0., var=1.)) / data_size
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





