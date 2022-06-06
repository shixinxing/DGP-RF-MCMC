import numpy as np
import tensorflow as tf

from dgp import DGP_RF
from likelihoods import Softmax
from utils_dataset import load_UCI_data


class RegressionDGP(DGP_RF):
    def __init__(self, d_in, d_out, n_hidden_layers=1, n_rf=20, n_gp=2, likelihood=Softmax(),
                 kernel_list=None, randon_fixed=True, name=None):
        super(RegressionDGP, self).__init__(d_in, d_out, n_hidden_layers=n_hidden_layers,
                                            n_rf=n_rf, n_gp=n_gp,
                                            likelihood=likelihood, kernel_list=kernel_list,
                                            randon_fixed=randon_fixed, name=name)


X, Y, Xs, Ys, X_mean, X_std, Y_mean = load_UCI_data('boston')




