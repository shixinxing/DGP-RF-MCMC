import numpy as np
import tensorflow as tf

import sys
sys.path.append("..")

from models.classification_model import ClassificationDGP
from likelihoods import Softmax
from utils_dataset import load_tf_dataset, normalize_MNIST
from utils import cyclical_step_rate


batch_size = 200
ds_train, ds_test, train_full_size, test_full_size = load_tf_dataset('mnist', batch_size=batch_size,
                                                                  transform_fn=normalize_MNIST)
ds_M = ds_train
d_in = 28 * 28
d_out = 10
model = ClassificationDGP(d_in, d_out, n_hidden_layers=2, n_rf=200, n_gp=[30, 10], likelihood=Softmax(),
                          kernel_type_list=['RBF','RBF'], random_fixed=True,
                          kernel_trainable=True ,set_nonzero_mean=False, input_cat=True)
print(f"Kernel type in the model is {model.kernel_type_list}")

total_epochs = 10000
start_sampling_epoch = 2000 #maybe use this as mixing or warming up ???
lr_0 = 0.01
beta = 0.98
epochs_per_cycle = 50
iterations_per_epoch = int(np.ceil(train_full_size / batch_size))
cycle_length = epochs_per_cycle * iterations_per_epoch #number of iterations in one period

print_epoch_cycle = 20
log_p = []
acc = []
for epoch in range(total_epochs):
    model.precond_update(ds_M, train_full_size, K_batches=32, precond_type='rmsprop', second_moment_centered=False)
    batch_index = 0
    for img_batch, label_batch in ds_train:
        batch_index = batch_index + 1
        if epoch < start_sampling_epoch: # fixed learning rate, zero temperature
            model.sgmcmc_update(img_batch, label_batch, train_full_size,
                                lr=lr_0, momentum_decay=beta,
                                resample_moments=False, temperature=0.)
        else: # cyclical learning rate, non-zero temperature
            step_index = (epoch - start_sampling_epoch) * iterations_per_epoch + batch_index
            step_rate, is_end = cyclical_step_rate(step_index, cycle_length, schedule='cosine', min_value=0.)
            lr = lr_0 * (step_rate**2)
            is_new_cycle = tf.equal(tf.math.mod(step_index, cycle_length), 1)
            model.sgmcmc_update(img_batch, label_batch, train_full_size,
                                lr=lr, momentum_decay=beta,
                                resample_moments=is_new_cycle, temperature=1.)
            if is_end: # sampling the model
                test_log_p = model.eval_log_likelihood(ds_test)
                test_acc = model.eval_all_accuracy(ds_test)
                log_p.append(test_log_p)
                acc.append(test_acc)
                print('#' * 20, f'Sampling at Epoch {epoch} ', f"lr = {lr}", '#' * 20)
    # print sampling process
    if (epoch + 1) % print_epoch_cycle == 0:
        train_log_p = model.eval_log_likelihood(ds_train)
        test_log_p = model.eval_log_likelihood(ds_test)
        train_acc = model.eval_all_accuracy(ds_train)
        test_acc = model.eval_all_accuracy(ds_test)
        print(f"Epoch: {epoch}")
        print(f"Mean Log Likelihood -- train: {tf.reduce_mean(train_log_p)}, "
              f"-- test: {tf.reduce_mean(test_log_p)} ")
        print(f"Accuracy -- train: {train_acc}, "
              f"-- test: {test_acc} ")
        print(" ")

log_p = tf.stack(log_p, axis=0) # [S, N]
acc = tf.stack(acc, axis=0) #[S,]

n_models = tf.shape(acc)[0]
predict_log_p = tf.reduce_logsumexp(log_p, axis=0) - tf.math.log(tf.cast(n_models, tf.float32))
predict_log_p = tf.reduce_mean(predict_log_p)
predict_acc = tf.reduce_mean(acc)

print(f"Number of sampled models: {n_models} ")
print(f"Test Log Likelihood of all sampled models: {predict_log_p}")
print(f"Test Mean acc of all sampled models: {predict_acc}")


