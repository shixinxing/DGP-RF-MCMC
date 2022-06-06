import tensorflow as tf
import numpy as np
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
            self.kernel_list = [ RBFKernel(n_feature=before_n_rf[i]) for i in range(n_hidden_layers) ]
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
        X = tf.constant(X, dtype=tf.float32)
        Y = tf.constant(Y, dtype=tf.float32)
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
        log_prior_sum = (self.prior_W() + self.prior_kernel_params() + self.prior_likelihood_params()) / data_size
        log_likelihood = tf.reduce_sum(self.log_likelihood(X_batch, Y_batch)) / batch_size
        return - (log_prior_sum + log_likelihood)


    def sghmc_update(self, X_batch, Y_batch, lr, data_size, eta=0.9, temperature=1.):
        """
        Standard SGLD and SGHMC:
        :param eta: eta = 1. SGLD; eta < 1, SGHMC
        """
        with tf.GradientTape() as tape:
            U = self.U(X_batch, Y_batch, data_size)
        grads = tape.gradient(U, self.trainable_variables)

        for param, grad in zip(self.trainable_variables, grads):
            if not hasattr(param, "auxiliary_v"):
                # param.auxiliary_v = tf.zeros_like(param)
                param.auxiliary_v = tf.random.normal(tf.shape(param))
            v_new = (1. - eta) * param.auxiliary_v - lr * grad
            eps = tf.random.normal(tf.shape(param))
            v_new += (2.0 * lr * eta * temperature / data_size) ** 0.5 * eps
            param.auxiliary_v = v_new
            param.assign_add(v_new)

    def csgmcmc_update(self, X_batch, Y_batch, lr, data_size, eta=0.9,
                     epoch=None, epoch_per_cycle=50, num_samples_per_cycle=5, temperature=1.):
        with tf.GradientTape() as tape:
            U = self.U(X_batch, Y_batch, data_size)
        grads = tape.gradient(U, self.trainable_variables)

        for param, grad in zip(self.trainable_variables, grads):
            if not hasattr(param, "auxiliary_v"):
                param.auxiliary_v = tf.zeros_like(param)
                # param.auxiliary_v = tf.random.normal(tf.shape(param))
            v_new = (1. - eta) * param.auxiliary_v - lr * grad
            if (epoch is None) or (epoch % epoch_per_cycle) + 1 > epoch_per_cycle - num_samples_per_cycle:
                eps = tf.random.normal(tf.shape(param))
                v_new += (2.0 * lr * eta * temperature / data_size) ** 0.5 * eps
            param.auxiliary_v = v_new
            param.assign_add(v_new)

    # def sgmcmc_update(self, X_batch, Y_batch, h, data_size, gamma, temperature=1.):
    #     with tf.GradientTape() as tape:
    #         U = self.U(X_batch, Y_batch, data_size)
    #     grads = tape.gradient(U, self.trainable_variables)
    #
    #     for param, gard in zip(self.trainable_variables, grads):
