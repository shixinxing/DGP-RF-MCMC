import numpy as np
import tensorflow as tf

import sys
sys.path.append("..")

from models.regression_model import RegressionDGP
from likelihoods import Gaussian
from utils_dataset import load_UCI_dataset
from utils import cyclical_step_rate


batch_size = 200
ds_train, ds_test, train_shape, test_shape = load_UCI_dataset('boston', batch_size=batch_size, transform_fn=None,
                                                                      data_dir='./data/')
train_size = train_shape[0]
ds_M = ds_train

# set the structure of dgp
hid_n_gp = min(int(train_shape[1]), 30)
d_out = 1
model = RegressionDGP(train_shape[1], d_out, n_hidden_layers=2, n_rf=500,
                      n_gp=[hid_n_gp, d_out], likelihood=Gaussian(),
                      kernel_type_list=['RBF','RBF'], kernel_trainable=True,
                      random_fixed=True, set_nonzero_mean=False, input_cat=True)

total_epochs = 5000
start_sampling_epoch = 2000 #maybe use this as mixing or warming up ???
lr_0 = 0.01
beta = 0.9
epochs_per_cycle = 50
iterations_per_epoch = int(np.ceil(train_size / batch_size))
cycle_length = epochs_per_cycle * iterations_per_epoch #number of iterations in one period

print_epoch_cycle = 100
log_p = []
mse = []
for epoch in range(total_epochs):
    model.precond_update(ds_M, train_size, K_batches=32, precond_type='rmsprop', second_moment_centered=False)
    batch_index = 0
    for img_batch, label_batch in ds_train:
        batch_index = batch_index + 1
        if epoch < start_sampling_epoch: # fixed learning rate, zero temperature
            model.sgmcmc_update(img_batch, label_batch, train_size,
                                lr=lr_0, momentum_decay=beta,
                                resample_moments=False, temperature=0.)
        else: # cyclical learning rate, non-zero temperature
            step_index = (epoch - start_sampling_epoch) * iterations_per_epoch + batch_index
            step_rate, is_end = cyclical_step_rate(step_index, cycle_length, schedule='cosine', min_value=0.)
            lr = lr_0 * (step_rate**2)
            is_new_cycle = tf.equal(tf.math.mod(step_index, cycle_length), 1)
            model.sgmcmc_update(img_batch, label_batch, train_size,
                                lr=lr, momentum_decay=beta,
                                resample_moments=is_new_cycle, temperature=1.)
            if is_end: # sampling the model
                test_log_p, test_se = model.eval_log_likelihood_and_se(ds_test)
                log_p.append(test_log_p)
                mse.append(test_se)
                print('#' * 20, f'Sampling at Epoch {epoch} ', f"lr = {lr}", '#' * 20)
    # print sampling process
    if (epoch + 1) % print_epoch_cycle == 0:
        train_log_p, train_se = model.eval_log_likelihood_and_se(ds_train)
        test_log_p, test_se = model.eval_log_likelihood_and_se(ds_test)
        print(f"Epoch: {epoch}")
        print(f"Mean Log Likelihood -- train: {tf.reduce_mean(train_log_p)}, "
              f"-- test: {tf.reduce_mean(test_log_p)} ")
        print(f"Root Mean Squared Error -- train: {tf.math.sqrt(tf.reduce_mean(train_se))}, "
              f"-- test: {tf.math.sqrt(tf.reduce_mean(test_se))} ")
        print(" ")

log_p = tf.stack(log_p, axis=0) # [S, N]
mse = tf.stack(mse, axis=0) #[S, N]

n_models = tf.shape(mse)[0]
predict_log_p = tf.reduce_logsumexp(log_p, axis=0) - tf.math.log(tf.cast(n_models, tf.float32))
predict_log_p = tf.reduce_mean(predict_log_p)
predict_rmse = tf.math.sqrt(tf.reduce_mean(mse))

print(f"Number of sampled models: {n_models} ")
print(f"Test Log Likelihood of all sampled models: {predict_log_p}")
print(f"Test Root MSE of all sampled models: {predict_rmse}")


