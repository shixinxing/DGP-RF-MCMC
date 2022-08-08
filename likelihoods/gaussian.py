import numpy as np
import tensorflow as tf
from utils import log_gaussian


class Gaussian(tf.Module):
    def __init__(self, variance=0.1, trainable=True):
        """
        :param variance: \sigma^2, noise variance should be greater than 0.
        """
        super().__init__()
        self.lik_log_var = tf.Variable(tf.math.log(variance), trainable=trainable, name="lik_log_var")

    @property
    def variance(self):
        return tf.math.exp(self.lik_log_var)

    def log_prob(self, F, Y):
        """
        :param F: [B, N, D]
        :param Y: [B, N, D]
        :return:  [B, N]
        """
        log_each_D = log_gaussian(Y, mean=F, var=self.variance)
        return tf.reduce_sum(log_each_D, axis=-1)
