import tensorflow as tf


class Softmax(tf.Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def log_prob(self, F, Y):
        """
        :param F: [*, B, N, latent-dim]
        :param Y: [*, B, N, 1]
        :return: log likelihood \log p(Y|F) whose shape is [*, B, N]
        """
        return - tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y[:, 0], logits=F)

    def predict_full(self, F):
        """
        :param F: [*, B, N, latent-dim]
        :return: [*, B, N, num_class]
        """
        return tf.nn.softmax(F)