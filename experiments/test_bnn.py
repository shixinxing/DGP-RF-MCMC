import sys
sys.path.append("..")
import tensorflow as tf

from BNN_test import BNNTest, BNNTestCos
from utils_dataset import load_dataset, normalize_MNIST
from utils import cyclical_lr_schedule


class ClassificationBNN(BNNTestCos): # could change BNNTest or BNNTestCos
    def __init__(self):
        super(ClassificationBNN, self).__init__()

    def eval_accuracy(self, X_batch, Y_batch):
        """
        :param X_batch: [N, D]
        :param Y_batch: [N, 1]
        :return: acc give one sample set of params in BNN
        """
        out = self.__call__(X_batch)
        out = tf.nn.softmax(out) #[N, num_class], float32
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
        print(f"Total rights: {right}, test size: {test_size} ")
        acc_test_all = right / test_size
        return acc_test_all


batch_size = 256
ds_train, ds_test, train_full_size, test_full_size = load_dataset('mnist', batch_size=batch_size,
                                                                  transform_fn=normalize_MNIST)
ds_M = ds_train
model = ClassificationBNN()

total_epoches = 150
start_sampling = 150  # Acc is over 90% at the third epoch.
lr_0 = 0.1
beta = 0.98
cycle_length = 50

for epoch in range(total_epoches):
    model.precond_update(ds_M, train_full_size, K_batches=32, precond_type='rmsprop', rho_rms=0.99)
    acc = 0.
    for img_batch, label_batch in ds_train:
        if epoch < start_sampling: # fixed learning rate, zero temperature
            model.sgmcmc_update(img_batch, label_batch, train_full_size,
                                lr=lr_0, beta=beta, temperature=0.)
            # acc = model.eval_accuracy(img_batch, label_batch)
            # print(f"Epoch: {epoch}, Iter: {iteration}, lr: {lr_0}, Acc: {acc}  ")
        else: # cyclical learning rate, non-zero temperature
            lr = cyclical_lr_schedule(lr_0, epoch - start_sampling, cycle_length)
            model.sgmcmc_update(img_batch, label_batch, train_full_size,
                                lr=lr, beta=beta, temperature=1.)
            # acc = model.eval_accuracy(img_batch, label_batch)
            # print(f"Epoch: {epoch}, Iter: {iteration}, lr: {lr}, Acc: {acc}  ")

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





