import sys
sys.path.append("..")
import tensorflow as tf

from dgp import DGP_RF
from likelihoods import Softmax
from utils_dataset import load_dataset, normalize_MNIST
from utils import cyclical_lr_schedule


class ClassificationDGP(DGP_RF):
    def __init__(self, d_in, d_out, n_hidden_layers=1, n_rf=30, n_gp=10, likelihood=Softmax(),
                 kernel_type_list=None, random_fixed=True, input_cat=False, set_nonzero_mean=False, name=None):
        super(ClassificationDGP, self).__init__(d_in, d_out, n_hidden_layers=n_hidden_layers,
                                                n_rf=n_rf, n_gp=n_gp, likelihood=likelihood,
                                                kernel_type_list=kernel_type_list, input_cat=input_cat,
                                                random_fixed=random_fixed, set_nonzero_mean=set_nonzero_mean, name=name)

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


batch_size = 256
ds_train, ds_test, train_full_size, test_full_size = load_dataset('mnist', batch_size=batch_size,
                                                                  transform_fn=normalize_MNIST)
ds_M = ds_train
model = ClassificationDGP(28*28, 10, n_hidden_layers=3, n_rf=500, n_gp=[1000, 100, 10], likelihood=Softmax(),
                          kernel_type_list=None, random_fixed=True, set_nonzero_mean=False, input_cat=True)

total_epoches = 5
start_sampling = 5
lr_0 = 0.001  # Training acc reaches over 95% very fast.
beta = 0.98
cycle_length = 50

for epoch in range(total_epoches):
    model.precond_update(ds_M, train_full_size, K_batches=32, precond_type='rmsprop', rho_rms=0.99)
    acc = 0.
    iteration = 0
    for img_batch, label_batch in ds_train:
        if epoch < start_sampling: # fixed learning rate, zero temperature
            model.sgmcmc_update(img_batch, label_batch, train_full_size,
                                lr=lr_0, beta=beta, temperature=0.)
            # test_acc_iter = model.eval_test_all(ds_test)
            # train_acc_iter = model.eval_test_all(ds_train)
            # print(f"Epoch: {epoch}, Iter: {iteration}, lr: {lr_0}, Acc on training data: {train_acc_iter}  ")
            # print(f"Epoch: {epoch}, Iter: {iteration}, lr: {lr_0}, Acc on test data: {test_acc_iter}  ")
        else: # cyclical learning rate, non-zero temperature
            lr = cyclical_lr_schedule(lr_0, epoch - start_sampling, cycle_length)
            model.sgmcmc_update(img_batch, label_batch, train_full_size,
                                lr=lr, beta=beta, temperature=1.)
            # acc = model.eval_accuracy(img_batch, label_batch)s
            # print(f"Epoch: {epoch}, Iter: {iteration}, lr: {lr}, Acc: {acc}  ")
        iteration += 1

    if epoch < start_sampling:
        lr_current = lr_0
    else:
        lr_current = lr
        # acc = model.eval_accuracy(img_batch, label_batch)
        # print(f"Epoch: {epoch}, Iter: {iteration}, lr: {lr_0}, Batch Acc: {acc}  ")
    train_acc = model.eval_test_all(ds_train)
    print(f"On training data, Epoch: {epoch},  lr: {lr_current}, Total Acc: {train_acc}  ")
    test_acc = model.eval_test_all(ds_test)
    print(f"On test data, Epoch: {epoch},  lr: {lr_current}, Total Acc: {test_acc}  ")
    print(" ")







