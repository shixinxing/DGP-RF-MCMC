import numpy as np
import tensorflow as tf

import sys
sys.path.append("..")
from dgp import DGP_RF
from likelihoods import Gaussian
from utils_dataset import load_UCI_dataset
from utils import cyclical_lr_schedule


class RegressionDGP(DGP_RF):
    def __init__(self, d_in, d_out, n_hidden_layers=1, n_rf=20, n_gp=2, likelihood=Gaussian(),
                 kernel_type_list=None, random_fixed=True, input_cat=False, set_nonzero_mean=False, name=None):
        super(RegressionDGP, self).__init__(d_in, d_out, n_hidden_layers=n_hidden_layers,
                                            n_rf=n_rf, n_gp=n_gp, likelihood=likelihood,
                                            kernel_type_list=kernel_type_list, random_fixed=random_fixed,
                                            input_cat=input_cat,set_nonzero_mean=set_nonzero_mean, name=name)

    def feed_forward(self, ds):
        for x_batch, y_batch in ds:
            x_batch = tf.constant(x_batch, tf.float32)
            # output mean because of the Gaussian likelihood in the final layer
            out_before_likelihood = self.BNN(x_batch)
        return out_before_likelihood

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
            ### redce_sum???
            se_batch = tf.reduce_mean(tf.square(y_batch - out_before_likelihood), axis=-1)
            se_all_data.append(se_batch)
        log_p_all_data = tf.concat(log_p_all_data, axis=0)
        se_all_data = tf.concat(se_all_data, axis=0)
        return log_p_all_data, se_all_data


batch_size = 128
ds_train, ds_test, train_shape, test_shape = load_UCI_dataset('boston', batch_size=batch_size, transform_fn=None,
                                                                      data_dir='./data/')
ds_M = ds_train
model = RegressionDGP(train_shape[1], 1, n_hidden_layers=2, n_rf=500, n_gp=[13, 10], likelihood=Gaussian(),
                          kernel_type_list=['RBF','RBF'], random_fixed=True, set_nonzero_mean=False, input_cat=True)

total_epoches = 2000
start_sampling = 1000
lr_0 = 0.01
beta = 0.98
cycle_length = 50

log_p = []
mse = []
for epoch in range(total_epoches):
    model.precond_update(ds_M, train_shape[0], K_batches=32, precond_type='rmsprop', rho_rms=0.99)
    for img_batch, label_batch in ds_train:
        if epoch < start_sampling: # fixed learning rate, zero temperature
            model.sgmcmc_update(img_batch, label_batch, train_shape[0],
                                lr=lr_0, beta=beta, temperature=0.)
        else: # cyclical learning rate, non-zero temperature
            lr = cyclical_lr_schedule(lr_0, epoch - start_sampling, cycle_length)
            model.sgmcmc_update(img_batch, label_batch, train_shape[0],
                                lr=lr, beta=beta, temperature=1.)

    if epoch < start_sampling:
        lr_current = lr_0
    else:
        lr_current = lr
    if (epoch + 1) % 50 == 0:
        train_log_p, train_se = model.eval_log_likelihood_and_se(ds_train)
        print(f"On training data, Epoch: {epoch},  lr: {lr_current}, ",
              f"Total log likelihood: {tf.reduce_mean(train_log_p)},  ",
              f"Total MSE: {tf.reduce_mean(train_se)}; ")
        test_log_p, test_se = model.eval_log_likelihood_and_se(ds_test)
        print(f"On test data, Epoch: {epoch},  lr: {lr_current}, ",
              f"Total log likelihood: {tf.reduce_mean(test_log_p)},  ",
              f"Total MSE: {tf.reduce_mean(test_se)}; ")
        print(" ")

    if epoch >= start_sampling:
        if (epoch - start_sampling) % cycle_length == cycle_length - 1:
            test_log_p, test_se = model.eval_log_likelihood_and_se(ds_test)
            print("#"*20, " Sampling the model ", "#"*20)
            print(f"On test data, Epoch: {epoch},  lr: {lr_current}, ",
                  f"Total log likelihood: {tf.reduce_mean(test_log_p)},  ",
                  f"Total MSE: {tf.reduce_mean(test_se)}; ")
            print(" ")
            log_p.append(test_log_p)
            mse.append(test_se)

log_p = tf.stack(log_p, axis=0) # [S, N]
mse = tf.stack(mse, axis=0) #[S, N]

n_models = tf.cast(tf.shape(mse)[0], tf.float32)
predict_log_p = tf.reduce_logsumexp(log_p, axis=0) - tf.math.log(n_models)
predict_log_p = tf.reduce_mean(predict_log_p)
predict_rmse = tf.math.sqrt(tf.reduce_mean(mse))

print(f"Number of sampling models: {tf.shape(mse)[0]} ")
print(f"Test log likelihood of sampling models: {predict_log_p}")
print(f"Test Root MSE: {predict_rmse}")




