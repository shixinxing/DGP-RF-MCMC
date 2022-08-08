import tensorflow as tf

from models.dgp import DGP_RF
from likelihoods import Softmax


class ClassificationDGP(DGP_RF):
    def __init__(self, d_in, d_out, n_hidden_layers=1, n_rf=30, n_gp=10, likelihood=Softmax(),
                 kernel_type_list=None, random_fixed=True, input_cat=False,
                 kernel_trainable=True, set_nonzero_mean=False, name=None):
        super(ClassificationDGP, self).__init__(d_in, d_out, n_hidden_layers=n_hidden_layers,
                                                n_rf=n_rf, n_gp=n_gp, likelihood=likelihood,
                                                kernel_type_list=kernel_type_list, input_cat=input_cat,
                                                random_fixed=random_fixed, kernel_trainable=kernel_trainable,
                                                set_nonzero_mean=set_nonzero_mean, name=name)

    def eval_accuracy(self, X_batch, Y_batch):
        """
        :param X_batch: [N, D]
        :param Y_batch: [N, 1]
        :return: acc give one sample set of params in BNN
        """
        out = self.BNN(X_batch)
        out = self.likelihood.predict_full(out) #[N, num_class], float32
        predicts = tf.cast(tf.math.argmax(out, axis=-1), tf.float32) #[N], int
        labels = tf.squeeze(Y_batch) #[N], float
        right_index = tf.cast(predicts == labels, tf.float32)
        right = tf.reduce_sum(right_index)
        acc = right / tf.cast(tf.shape(X_batch)[0], tf.float32)
        return acc

    def eval_test_all(self, ds_test):
        right = 0.
        test_size = 0.
        for img_batch, label_batch in ds_test:
            batch_size = tf.cast(tf.shape(img_batch)[0], tf.float32)
            right += self.eval_accuracy(img_batch, label_batch) * batch_size
            test_size += batch_size
        # print(f"Total rights: {right}, test size: {test_size} ")
        acc_test_all = right / test_size
        return  acc_test_all

    def eval_test_free_random(self,ds_test):
        self.BNN.set_random_fixed(False)
        acc =  self.eval_test_all(ds_test)
        self.BNN.set_random_fixed(True)
        return acc