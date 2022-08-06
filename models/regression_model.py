import tensorflow as tf

from models.dgp import DGP_RF
from likelihoods import Gaussian

class RegressionDGP(DGP_RF):
    def __init__(self, d_in, d_out, n_hidden_layers=1, n_rf=20, n_gp=2, likelihood=Gaussian(),
                 kernel_type_list=None, kernel_trainable=True,
                 random_fixed=True, input_cat=False, set_nonzero_mean=False, name=None):
        super(RegressionDGP, self).__init__(d_in, d_out, n_hidden_layers=n_hidden_layers,
                                            n_rf=n_rf, n_gp=n_gp, likelihood=likelihood,
                                            kernel_type_list=kernel_type_list, kernel_trainable=kernel_trainable,
                                            random_fixed=random_fixed,
                                            input_cat=input_cat,set_nonzero_mean=set_nonzero_mean, name=name)

    def feed_forward(self, ds):
        y_batch_before_likelihood = None
        for x_batch, y_batch in ds:
            x_batch = tf.constant(x_batch, tf.float32)
            # output mean because of the Gaussian likelihood in the final layer
            y_batch_before_likelihood = self.BNN(x_batch)
        return y_batch_before_likelihood

    def feed_forward_all_layers(self, X):
        X = tf.constant(X, dtype=tf.float32)
        output_list = []
        for l, layer in enumerate(self.BNN.layers):
            X = layer(X)
            if l % 2 == 1:
                output_list.append(X)
        return output_list

    def eval_log_likelihood_and_se(self, ds):
        """
        :param ds:iterable X: [N, D_in]; Y: [N, D_out];
        :return: output matrix of log likelihood log p(Y|F), shape [N, ] and square errors [N, ]
        """
        log_p_all_data = []
        se_all_data = []
        for x_batch, y_batch in ds:
            out_before_likelihood = self.BNN(x_batch)
            log_p_batch = self.likelihood.log_prob(out_before_likelihood, y_batch)
            log_p_all_data.append(log_p_batch)
            assert isinstance(self.likelihood, Gaussian), "The likelihood of the model is not Gaussian!"
            ### reduce_sum???
            se_batch = tf.reduce_mean(tf.square(y_batch - out_before_likelihood), axis=-1)
            se_all_data.append(se_batch)
        log_p_all_data = tf.concat(log_p_all_data, axis=0)
        se_all_data = tf.concat(se_all_data, axis=0)
        return log_p_all_data, se_all_data