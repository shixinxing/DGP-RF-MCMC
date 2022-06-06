import numpy as np
import tensorflow as tf

import sys
sys.path.append('..')
from dgp import DGP_RF
from likelihoods import Gaussian


class ToyRegressionDGP(DGP_RF):
    def __init__(self, d_in, d_out, n_hidden_layers=1, n_rf=20, n_gp=2, likelihood=Gaussian(),
                 kernel_list=None, random_fixed=True, name=None):
        super(ToyRegressionDGP, self).__init__(d_in, d_out, n_hidden_layers=n_hidden_layers,
                                            n_rf=n_rf, n_gp=n_gp,
                                            likelihood=likelihood, kernel_list=kernel_list,
                                            random_fixed=random_fixed, name=name)

    def feed_forward(self, X):
        """
        :param X: [N, D]
        :return: [N, D_out]
        """
        X = tf.constant(X, dtype=tf.float32)
        # output mean because of the Gaussian likelihood in the final layer
        out = self.BNN(X)
        return out

    def feed_forward_all_layers(self, X):
        X = tf.constant(X, dtype=tf.float32)
        output_list = []
        for l, layer in enumerate(self.BNN.layers):
            X = layer(X)
            if l % 2 == 1:
                output_list.append(X)
        return output_list


    def adjust_traditional_learning_rate(self, epoch, lr_0=0.5, dtype=tf.float32):
        epoch = tf.cast(epoch, dtype)
        lr = 0.05 * (1. + epoch) ** (-0.55) * lr_0
        return lr

    def adjust_cyclical_learning_rate(self, epoch, K, M, lr_0=0.5):
        """
        using cyclical lr schedule
        :param epoch: current epoch index
        :param K: total iterations
        :param M: Number of cycles
        :param lr_0: initial learning rate
        """
        cos_inner = tf.constant(np.pi) / (K // M) * (epoch % (K // M))
        cos_out = tf.math.cos(cos_inner) + 1
        lr = 0.5 * lr_0 * cos_out
        return lr


def get_sin_data(num_training, num_testing, std_noise=0.1):
    """
    :return: [N, 1]
    """
    X_train = np.random.uniform(-1.5, 1.5, num_training)
    Y_train = np.sin(np.pi * X_train) + np.random.randn(num_training) * std_noise
    X_train = np.reshape(X_train, [num_training, 1])
    Y_train = np.reshape(Y_train, [num_training, 1])

    X_test = np.linspace(-2., 2., num_testing)[:, None]

    return X_train, Y_train, X_test

num_training = 20
num_testing = 100
std_noise = 0.1

X_train, Y_train, X_test = get_sin_data(num_training, num_testing, std_noise=std_noise)

d_in = 1
d_out =1
# kernel_list =

model = ToyRegressionDGP(d_in, d_out, n_hidden_layers=2, n_rf=10, n_gp=1, likelihood=Gaussian(),
                         kernel_list=None, random_fixed=True)

num_mixing_epoches = 10
num_samples = 2
eta = 0.9

total_epoches = 1000 #K
M = 5
K_M = total_epoches // M

# out_samples = []
# for epoch in range(total_epoches):
#     lr = model.adjust_cyclical_learning_rate(epoch, total_epoches, M, lr_0=0.1)
#     model.sghmc_update(X_train, Y_train, lr, num_training, eta=eta,
#                        epoch=epoch, epoch_per_cycle=K_M, num_samples_per_cycle=5,
#                        temperature=1./num_training)
#     if epoch % K_M + 1 > K_M - 3: # 3 samples per cycle
#         out = model.feed_forward(X_test)
#         out_samples.append(out)
#
# out_samples = tf.concat(out_samples, axis=-1).numpy()  # [N, num_samples]
#
# print(out_samples)

num_training = 20
num_testing = 100
std_noise = 0.1

X_train, Y_train, X_test = get_sin_data(num_training, num_testing, std_noise=std_noise)

d_in = 1
d_out =1
# kernel_list =

model = ToyRegressionDGP(d_in, d_out, n_hidden_layers=2, n_rf=50, n_gp=1, likelihood=Gaussian(),
                         kernel_list=None, random_fixed=False)

num_mixing_epoches = 2000
num_samples = 50
eta = 0.9

out_samples = []
for epoch in range(num_mixing_epoches + num_samples):
    lr = model.adjust_traditional_learning_rate(epoch, lr_0=0.5)
    if epoch < num_mixing_epoches:
        model.sghmc_update(X_train, Y_train, lr, num_training, eta=eta, temperature=1.)
    elif epoch >= num_mixing_epoches:
        model.sghmc_update(X_train, Y_train, lr, num_training, eta=eta, temperature=1.)
        out = model.feed_forward_all_layers(X_test)
        out_samples.append(out)

out_hid_list = []
for sample in out_samples:
    out_hid = sample[0]
    out_hid_list.append(out_hid)

out_hid_list = tf.concat(out_hid_list, axis=-1).numpy() #[N, num_samples]
# print(out_samples)
