import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from dgp import DGP_RF
from likelihoods import Softmax
from utils_dataset import load_dataset, normalize_MNIST


class ClassificationDGP(DGP_RF):
    def __init__(self, d_in, d_out, n_hidden_layers=1, n_rf=20, n_gp=2, likelihood=Softmax(),
                 kernel_list=None, randon_fixed=True, name=None):
        super(ClassificationDGP, self).__init__(d_in, d_out, n_hidden_layers=n_hidden_layers,
                                            n_rf=n_rf, n_gp=n_gp,
                                            likelihood=likelihood, kernel_list=kernel_list,
                                            randon_fixed=randon_fixed, name=name)

    def eval_accuracy(self, X_batch, Y_batch):
        """
        :param X_batch: [N, D]
        :param Y_batch: [N, 1]
        :return: acc give one sample set of params in BNN
        """
        out = self.BNN(X_batch)
        out = self.likelihood.predict_full(out) #[N, num_class]
        predicts = tf.math.argmax(out, axis=-1) #[N]
        labels = tf.squeeze(Y_batch) #[N]
        right = tf.cast(tf.reduce_sum(predicts == labels), tf.float32)
        acc = right/tf.cast(tf.shape(X_batch)[0], tf.float32)
        return acc


