import tensorflow as tf
from likelihoods import Softmax, Gaussian
from kernels import RBFKernel, ARCKernel
from layers import RBFLayer, ARCLayer, GPLayer
from utils import BNN_from_list, BNN_from_list_input_cat, log_gaussian


class DGP_RF(tf.Module):
    def __init__(self, d_in, d_out, n_hidden_layers=1, n_rf=20, n_gp=2, likelihood=Softmax(),
                 kernel_type_list=None, kernel_trainable=True, random_fixed=True, input_cat=False,
                 set_nonzero_mean=False, name=None):
        """
        :param d_in: Input dim
        :param d_out: Output dim
        :param n_hidden_layers: Number of hidden layers
        :param n_rf: Number of random features
        :param n_gp: Number of latent GPs in each layer
        :param likelihood: Likelihood class in the last layer
        :param kernel_type_list: kernel type list, set is_ard default True
        :param kernel_trainable: fix the trainable variables or not
        :param random_fixed: z fixed or not when feeding forward
        :param input_cat: concatenate input to each hidden layer except the final layer
        """
        super(DGP_RF, self).__init__(name=name)
        self.d_in = d_in
        self.d_out = d_out
        self.n_hidden_layers = n_hidden_layers
        self.random_fixed = random_fixed
        self.input_cat = input_cat
        self.set_nonzero_mean = set_nonzero_mean
        self.kernel_trainable = kernel_trainable

        if tf.rank(n_rf) == 0:
            self.n_rf = n_rf * tf.ones([n_hidden_layers], dtype=tf.int32) #[20, 20]
        else:
            self.n_rf = tf.constant(n_rf, dtype=tf.int32)
        if tf.rank(n_gp) == 0:
            self.n_gp = n_gp * tf.ones([n_hidden_layers], dtype=tf.int32) #[2, 2]
        else:
            self.n_gp = tf.constant(n_gp, dtype=tf.int32)

        self.likelihood = likelihood
        if kernel_type_list is None:
            self.kernel_type_list = ['RBF' for _ in range(n_hidden_layers)]
            # self.kernel_type_list = ['ARC' for _ in range(n_hidden_layers)]
        else:
            assert len(kernel_type_list) == n_hidden_layers, "Kernel type list's length does not match!"
            self.kernel_type_list = kernel_type_list
        self.kernel_list = self.transform_kernel_list()
        self.BNN = self.transformed_BNN()

    def transform_kernel_list(self):
        kernel_list = []
        if not self.input_cat:
            before_n_rf = tf.concat([[self.d_in], self.n_gp[:-1]], axis=0)
        else:
            before_n_rf = tf.concat([[self.d_in], [n_gp + self.d_in for n_gp in self.n_gp[:-1]]], axis=0)
        for i, kernel_type in enumerate(self.kernel_type_list):
            if kernel_type == 'RBF':
                kernel = RBFKernel(n_feature=before_n_rf[i], trainable=self.kernel_trainable,
                                   is_ard=True, length_scale=None)
                kernel_list.append(kernel)
            elif kernel_type == 'ARC':
                kernel = ARCKernel(n_feature=before_n_rf[i], trainable=self.kernel_trainable,
                                   is_ard=True, length_scale=None)
                kernel_list.append(kernel)
            else:
                raise NotImplementedError
        return kernel_list

    def transformed_BNN(self):
        """
        :return: corresponding transformed BNN as an attribute
        """
        bnn = []
        for l in range(self.n_hidden_layers):
            kernel_tmp = self.kernel_list[l]
            if kernel_tmp.kernel_type == "RBF":
                layer_Omega = RBFLayer(kernel_tmp, self.n_rf[l], random_fixed=self.random_fixed,
                                       set_nonzero_mean=self.set_nonzero_mean)
                layer_GP = GPLayer(2 * self.n_rf[l], self.n_gp[l])
            elif kernel_tmp.kernel_type == "ARC":
                layer_Omega = ARCLayer(kernel_tmp, self.n_rf[l], random_fixed=self.random_fixed,
                                       set_nonzero_mean=self.set_nonzero_mean)
                layer_GP = GPLayer(self.n_rf[l], self.n_gp[l])
            else:
                raise  NotImplementedError
            bnn.extend([layer_Omega, layer_GP])

        if not self.input_cat:
            return BNN_from_list(bnn)
        else:
            return BNN_from_list_input_cat(bnn)
        # else: # add input concatenation

    def set_random_fixed(self, state):
        for l in range(self.n_hidden_layers):
            rf_layer = self.BNN.layers[2 * l]
            rf_layer.set_random_fixed(state)

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
            # change the variance of the prior? since it serves as regulation term
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
        else:
            raise NotImplementedError

    def U(self, X_batch, Y_batch, data_size):
        """
        minibatch average potential energy per sample: -(1/M) \sum_{i=1}^M log p(y_i|x_i,w) - (1/N) log p(w)
        :param data_size: N
        """
        batch_size = tf.shape(X_batch)[0]
        batch_size = tf.cast(batch_size, tf.float32)
        data_size = tf.cast(data_size, tf.float32)
        # log_prior_sum = (self.prior_W() + self.prior_kernel_params() + self.prior_likelihood_params()) / data_size
        log_prior_sum = 0.
        for param in self.trainable_variables:
            log_prior_sum += tf.reduce_sum(log_gaussian(param, mean=0., var=1.)) / data_size
        log_likelihood = tf.reduce_sum(self.log_likelihood(X_batch, Y_batch)) / batch_size
        return - (log_prior_sum + log_likelihood)

    def sgmcmc_update(self, X_batch, Y_batch, data_size, lr=0.01, momentum_decay=0.95,
                      resample_moments=False, temperature=1.):
        """
        implement one step of SGHMC and SGLD fore each group of params;
        :param lr: here the learning rate is the instantaneous effect of the average-minibatch-gradient
        on the parameter. Must be positive and typically is in a range from 1.0e-6 to 0.1.
        :param momentum_decay: i.e. beta, in the range [0,1), when beta=0, equal to SGLD
        typically close to 1, e.g. values such as 0.75, 0.8, 0.9, 0.95.
        """
        with tf.GradientTape() as tape:
            U = self.U(X_batch, Y_batch, data_size)
        grads = tape.gradient(U, self.trainable_variables)

        h = tf.sqrt(lr / data_size)
        for param, grad in zip(self.trainable_variables, grads):
            assert hasattr(param, "moments"), "Trainable Params do not have attr moments!"
            if resample_moments:
                param.moments = tf.random.normal(tf.shape(param))
            m_new = momentum_decay * param.moments - h * data_size * grad
            eps = tf.random.normal(tf.shape(param))
            assert hasattr(param, "M"), "Trainable Params do not have attr preconditioner M!"
            m_new = m_new + tf.math.sqrt(2.0 * (1. - momentum_decay) * temperature * param.M) * eps
            param.moments = m_new
            param.assign_add(h * tf.math.reciprocal(param.M) * param.moments)

    def precond_update(self, ds, data_size, K_batches=32, precond_type='rmsprop', second_moment_centered=False):
        """
        update the preconditioner M based on computing the gradients of mini-batches.
        :param ds: tf.data.Dataset as Iterable
        :param data_size: N
        :param K_batches: use K number of mini-batches to estimate preconditioner M, greater than 1;
        :param precond_type: "identity" - M = I; "rmsprop" - using gradient noise estimator;
        :param second_moment_centered: if True, use gradient noise variance, if False, use the second moment
        :return: preconditioner M w.r.t each param
        """
        # add attributes "moments" and preconditioner "M" to each group of variables
        for param in self.trainable_variables:
            if not hasattr(param, "M"):
                param.M = tf.constant(1., dtype=tf.float32)
            if not hasattr(param, "moments"):
                # param.moments = tf.zeros_like(param)
                param.moments = tf.random.normal(tf.shape(param))
        if precond_type == 'identity':
            return None
        elif precond_type == 'rmsprop':
            for param in self.trainable_variables:
                if not hasattr(param, "m_c"):
                    param.m_c = None
                param.m_c = tf.math.rsqrt(param.M) * param.moments
            # using gradient noise estimator with in first K batches to decide preconditioner M
            # Statistic variables are obtained by Welford's online algorithm
            DEFAULT_REGULARIZATION = 1.0e-7
            k = 0
            for X_batch, Y_batch in ds:
                with tf.GradientTape() as tape:
                    U_pre = self.U(X_batch, Y_batch, data_size)
                grads = tape.gradient(U_pre, self.trainable_variables)

                k = k + 1
                for param, grad in zip(self.trainable_variables, grads):
                    if not hasattr(param, 'm2_pre'): # auxiliary moments in Welford
                        param.m2_pre = tf.zeros_like(param, dtype=tf.float32)
                    if not hasattr(param, 'mean_pre'): #
                        param.mean_pre = tf.zeros_like(param, dtype=tf.float32)

                    delta = grad - param.mean_pre
                    param.mean_pre = param.mean_pre + delta / k #update gradient mean over batched
                    delta2 = grad - param.mean_pre
                    param.m2_pre = param.m2_pre + delta * delta2 #update auxiliary moments
                if k == K_batches:
                    break
            assert k == K_batches, f"Estimating M ends before we use {K_batches} batches, we actually use {k} batches!"
            # estimate the mass w.r.t each trainable variable
            mass_min = None
            for param in self.trainable_variables:
                if not hasattr(param, 'mass_estimate'):
                    param.mass_estimate = tf.zeros([], dtype=tf.float32)
                if second_moment_centered: #estimate gradient variance
                    mass_estimate_sq = param.m2_pre / tf.cast(K_batches - 1, tf.float32)
                    mass_estimate_sq = tf.reduce_mean(mass_estimate_sq)
                    param.mass_estimate = tf.math.sqrt(mass_estimate_sq + DEFAULT_REGULARIZATION)
                else: #estimate E[G^2] = (E[G])^2 + Var[G]
                    variance_esitmate = param.m2_pre / tf.cast(K_batches, tf.float32)
                    mass_estimate_sq = tf.math.square(param.mean_pre) + variance_esitmate
                    mass_estimate_sq = tf.reduce_mean(mass_estimate_sq)
                    param.mass_estimate = tf.math.sqrt(mass_estimate_sq + DEFAULT_REGULARIZATION)

                if mass_min is None:
                    mass_min = param.mass_estimate
                elif param.mass_estimate < mass_min:
                    mass_min = param.mass_estimate
            # scale the minimum estimated mass to one
            for param in self.trainable_variables:
                param.M = param.mass_estimate / mass_min
                param.moments = tf.math.sqrt(param.M) * param.m_c
            return None
        else:
            raise NotImplementedError
