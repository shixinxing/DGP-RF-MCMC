import sys
sys.path.append("..")
import tensorflow as tf
import tensorflow.keras.optimizers as optimizer

from BNN_test import NNCos
from utils_dataset import load_dataset, normalize_MNIST


class ClassificationNN(NNCos):
    def __init__(self):
        super(ClassificationNN, self).__init__()

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
        # print(f"Total rights: {right}, test size: {test_size} ")
        acc_test_all = right / test_size
        return acc_test_all


batch_size = 256
ds_train, ds_test, train_full_size, test_full_size = load_dataset('mnist', batch_size=batch_size,
                                                                  transform_fn=normalize_MNIST)
ds_M = ds_train
model = ClassificationNN()

total_epoches = 150
lr = 0.01
opt = optimizer.SGD(learning_rate=lr)


for epoch in range(total_epoches):
    for img_batch, label_batch in ds_train:
        with tf.GradientTape() as tape:
            loss = model.cross_entropy_loss(img_batch, label_batch)
            # loss = model.MSE_loss(img_batch, label_batch)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

    train_acc = model.eval_test_all(ds_train)
    print(f"On training data, Epoch: {epoch},  lr: {lr}, Total Acc: {train_acc}  ")
    test_acc = model.eval_test_all(ds_test)
    print(f"On test data, Epoch: {epoch},  lr: {lr}, Total Acc: {test_acc}  ")
    print(" ")





