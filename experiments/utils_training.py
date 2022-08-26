import numpy as np
import tensorflow as tf

import sys
sys.path.append("..")

from utils_dataset import load_UCI_dataset, download_UCI_data_info
from utils import cyclical_step_rate


def regression_train(model, dataset_name='boston', batch_size=200, data_dir='./data/',
                     lr_0=0.01, momentum_decay=0.9,
                     precond_type='identity', K_batches=None, second_moment_centered=None,
                     resample_in_cycle_head = True,
                     total_epochs=5000, start_sampling_epoch=2000, epochs_per_cycle=50,
                     print_epoch_cycle=100):
    if precond_type != 'identity' and K_batches is None and second_moment_centered is None:
        raise ValueError("Args K_batches or second_moment_centered shouldn't be None!")
    # Note that y_batch is normalized, record Y's std
    _, _, _, _, _, _, Y_std = download_UCI_data_info(dataset_name, data_path=data_dir)
    ds_train, ds_test, train_shape, test_shape = load_UCI_dataset(dataset_name, batch_size=batch_size,
                                                                  data_dir=data_dir) # ds_train has dropped remainder
    train_size = train_shape[0]
    train_size_drop_remainder = train_size - train_size % batch_size
    print(f"Training size is {train_size_drop_remainder} after remainder dropping. ")
    if train_size_drop_remainder < batch_size:
        print("Batch size is larger than all training data! ")
        batch_size = train_size_drop_remainder
        print(f"reset the batch size to {train_size_drop_remainder}! ")

    ds_M = ds_train
    iterations_per_epoch = int(train_size_drop_remainder / batch_size) # force float to int
    cycle_length = epochs_per_cycle * iterations_per_epoch #number of iterations in one period

    log_p = []
    mse = []
    for epoch in range(total_epochs):
        model.precond_update(ds_M, train_size, K_batches=K_batches,
                             precond_type=precond_type, second_moment_centered=second_moment_centered)
        batch_index = 0
        for img_batch, label_batch in ds_train:
            batch_index = batch_index + 1
            if epoch < start_sampling_epoch: # fixed learning rate, zero temperature
                model.sgmcmc_update(img_batch, label_batch, train_size,
                                    lr=lr_0, momentum_decay=momentum_decay,
                                    resample_moments=False, temperature=0.)
            else: # cyclical learning rate, non-zero temperature
                step_index = (epoch - start_sampling_epoch) * iterations_per_epoch + batch_index
                step_rate, is_end = cyclical_step_rate(step_index, cycle_length, schedule='cosine', min_value=0.)
                lr = lr_0 * (step_rate**2)
                if resample_in_cycle_head == True:
                    is_new_cycle = tf.equal(tf.math.mod(step_index, cycle_length), 1)
                else:
                    is_new_cycle = False
                model.sgmcmc_update(img_batch, label_batch, train_size,
                                    lr=lr, momentum_decay=momentum_decay,
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
            print(f"Root Mean Squared Error -- train: {tf.math.sqrt(tf.reduce_mean(train_se))*Y_std}, "
                  f"-- test: {tf.math.sqrt(tf.reduce_mean(test_se))*Y_std} ")
            print(" ")

    log_p = tf.stack(log_p, axis=0) # [S, N]
    mse = tf.stack(mse, axis=0) #[S, N]

    n_models = tf.shape(mse)[0]
    predict_log_p = tf.reduce_logsumexp(log_p, axis=0) - tf.math.log(tf.cast(n_models, tf.float32))
    predict_log_p = tf.reduce_mean(predict_log_p)
    predict_rmse = tf.math.sqrt(tf.reduce_mean(mse)) * Y_std

    print(f"Dataset: {dataset_name}, Number of sampled models: {n_models} ")
    print(f"Test Log Likelihood of all sampled models: {predict_log_p}")
    print(f"Test Root MSE of all sampled models: {predict_rmse}")

    return log_p, mse*(Y_std**2)

def MCEM_sampler(model, dataset_name='boston', batch_size=200, data_dir='./data/',
                 lr_0=0.01, momentum_decay=0.9,
                 precond_type='identity', K_batches=None, second_moment_centered=None,
                 resample_in_cycle_head=True, start_sampling_epoch=2000, epochs_per_cycle=50):
    if precond_type != 'identity' and K_batches is None and second_moment_centered is None:
        raise ValueError("Args K_batches or second_moment_centered shouldn't be None!")
    # Note that y_batch is normalized, record Y's std
    _, _, _, _, _, _, Y_std = download_UCI_data_info(dataset_name, data_path=data_dir)
    Y_std = Y_std[0]
    ds_train, ds_test, train_shape, test_shape = load_UCI_dataset(dataset_name, batch_size=batch_size,
                                                                  data_dir=data_dir)
    train_size = train_shape[0]
    train_size_drop_remainder = train_size - train_size % batch_size
    print(f"Training size is {train_size_drop_remainder} after remainder dropping. ")
    if train_size_drop_remainder < batch_size:
        print("Batch size is larger than all training data! ")
        batch_size = train_size_drop_remainder
        print(f"reset the batch size to {train_size_drop_remainder}! ")
    ds_M = ds_train
    iterations_per_epoch = int(train_size_drop_remainder / batch_size)  # force float to int
    cycle_length = epochs_per_cycle * iterations_per_epoch  # number of iterations in one period

    def sampler(num_samples=100, print_epoch_cycle=100):
        total_epochs = start_sampling_epoch + num_samples * epochs_per_cycle
        W_samples = []
        log_p = []
        mse = []
        sampled_model_index = 0
        for epoch in range(total_epochs):
            model.precond_update(ds_M, train_size, precond_type=precond_type, full_bayesian=False,
                                 K_batches=K_batches, second_moment_centered=second_moment_centered)
            batch_index = 0
            for img_batch, label_batch in ds_train:
                batch_index = batch_index + 1
                if epoch < start_sampling_epoch: # fixed learning rate, zero temperature
                    model.sgmcmc_update(img_batch, label_batch, train_size, lr=lr_0, momentum_decay=momentum_decay,
                                             resample_moments=False, temperature=0., full_bayesian=False)
                else: # cyclical learning rate, non-zero temperature
                    step_index = (epoch - start_sampling_epoch) * iterations_per_epoch + batch_index
                    step_rate, is_end = cyclical_step_rate(step_index, cycle_length, schedule='cosine', min_value=0.)
                    lr = lr_0 * (step_rate**2)
                    if resample_in_cycle_head == True:
                        is_new_cycle = tf.equal(tf.math.mod(step_index, cycle_length), 1)
                    else:
                        is_new_cycle = False
                    model.sgmcmc_update(img_batch, label_batch, train_size, lr=lr, momentum_decay=momentum_decay,
                                             resample_moments=is_new_cycle, temperature=1., full_bayesian=False)
                    if is_end: # sampling the model
                        sampled_model_index += 1
                        W_samples.append(model.W_mcmc) # W_mcmc is a list
                        print('#' * 20, f'Sample No.{sampled_model_index} at Epoch {epoch} ', f"lr = {lr}", '#' * 20)

                        test_log_p, test_se = model.eval_log_likelihood_and_se(ds_test)
                        log_p.append(test_log_p)
                        mse.append(test_se)
            # print sampling process
            if (epoch + 1) % print_epoch_cycle == 0:
                train_log_p, train_se = model.eval_log_likelihood_and_se(ds_train)
                test_log_p, test_se = model.eval_log_likelihood_and_se(ds_test)
                print(f"Sampling Epoch: {epoch}")
                print(f"Mean Log Likelihood -- train: {tf.reduce_mean(train_log_p)}, "
                      f"-- test: {tf.reduce_mean(test_log_p)} ")
                print(f"Root Mean Squared Error -- train: {tf.math.sqrt(tf.reduce_mean(train_se)) * Y_std}, "
                      f"-- test: {tf.math.sqrt(tf.reduce_mean(test_se)) * Y_std} \n")

        log_p = tf.stack(log_p, axis=0)  # [S, N]
        mse = tf.stack(mse, axis=0)  # [S, N]
        n_models = tf.shape(mse)[0]
        predict_log_p = tf.reduce_logsumexp(log_p, axis=0) - tf.math.log(tf.cast(n_models, tf.float32))
        predict_log_p = tf.reduce_mean(predict_log_p)
        predict_rmse = tf.math.sqrt(tf.reduce_mean(mse)) * Y_std

        print("*"*20, f" Dataset: {dataset_name} -- End of Sampling ", "*"*20)
        print(f"Number of sampled models: {n_models} ")
        print(f"Test Log Likelihood of all sampled models: {predict_log_p}")
        print(f"Test Root MSE of all sampled models: {predict_rmse}")
        print("*"*70,"\n")
        return W_samples, log_p, mse * (Y_std **2)

    return sampler

def MCEM_Q_maximizer(model, data_size, optimizer):
    def maximizer(W_samples, X_batch, Y_batch):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch([model.Omega_hyperparams, model.Likelihood_hyperparams])
            Q = 0.
            num_samples = 0
            for W_model_list in W_samples:
                model.assign_W(W_model_list)
                # regard W as constant and retain the computing graph
                log_p_sample = - model.U(X_batch, Y_batch, data_size,
                                         full_bayesian=False, allow_gradient_from_W=False)
                Q = Q + log_p_sample
                num_samples += 1
            Q = Q / tf.cast(num_samples, tf.float32)
            neg_Q = - Q
        grads = tape.gradient(neg_Q, tape.watched_variables()) #maximize Q, should use negetive Q to minimize
        print("*"*70)
        print(f"Q function is {Q} averaged by {num_samples} samples.")
        print("*"*70, "\n")
        optimizer.apply_gradients(zip(grads, tape.watched_variables()))
    return maximizer

def MCEM(sampler_EM, maximizer, sampler_fixing_hyper, total_EM_steps, ds_train,
         num_samples_EM=100, num_samples_fixing_hyper=200,
         print_epoch_cycle_EM=100, print_epoch_cycle_fixing=100):
    ds_train_repeat = ds_train.repeat()
    em_step = 0
    for x_batch, y_batch in ds_train_repeat:
        em_step += 1
        # E step
        print("#"*15, f"EM step {em_step} of total {total_EM_steps} steps. E Step: ", "#"*15)
        W_samples, _, _ = sampler_EM(num_samples=num_samples_EM, print_epoch_cycle=print_epoch_cycle_EM)
        # M step
        print("#"*15, f"EM step {em_step} of total {total_EM_steps} steps, M Step: ", "#"*15)
        maximizer(W_samples, x_batch, y_batch)
        if em_step == total_EM_steps:
            break
    print("#"*15,f"After {total_EM_steps} EM steps, fixing hyperparams and sample from posterior.","#"*15)
    _, log_p, mse = sampler_fixing_hyper(num_samples=num_samples_fixing_hyper,
                                          print_epoch_cycle=print_epoch_cycle_fixing)
    return log_p, mse

def MCEM_windows(sampler_EM, maximizer, sampler_fixing_hyper, total_EM_steps, ds_train,
                 num_samples_fixing_hyper=200, window_size=300,
                 print_epoch_cycle_EM=100, print_epoch_cycle_fixing=100):
    W_samples_window = []
    log_p_window = None
    mse_window = None

    ds_train_repeat = ds_train.repeat()
    em_step = 0
    for x_batch, y_batch in ds_train_repeat:
        em_step += 1
        # E step
        print("#"*15, f"EM step {em_step} of total {total_EM_steps} steps. E Step: ", "#"*15)
        W_samples, log_p, mse = sampler_EM(num_samples=1, print_epoch_cycle=print_epoch_cycle_EM)
        W_samples_window.extend(W_samples)
        if len(W_samples_window) == 1:
            log_p_window = log_p
            mse_window = mse
        elif len(W_samples_window) <= window_size:
            log_p_window = tf.concat([log_p_window, log_p],axis=0)
            mse_window = tf.concat([mse_window, mse], axis=0)
        else:
            W_samples_window = W_samples_window[-window_size:]
            log_p_window = tf.concat([log_p_window, log_p], axis=0)[1:,:]
            mse_window = tf.concat([mse_window, mse], axis=0)[1:,:]
        n_models = tf.shape(mse_window)[0]
        predict_log_p = tf.reduce_logsumexp(log_p_window, axis=0) - tf.math.log(tf.cast(n_models, tf.float32))
        predict_log_p = tf.reduce_mean(predict_log_p)
        predict_rmse = tf.math.sqrt(tf.reduce_mean(mse_window))
        print("*"*20," End of E step ", "*"*20)
        print(f"Number of all sampled models in window: {n_models} ")
        print(f"Test Log Likelihood of all models in window: {predict_log_p}")
        print(f"Test Root MSE of all models in window: {predict_rmse}\n")
        # M step
        print("#"*15, f"EM step {em_step} of total {total_EM_steps} steps, M Step: ", "#"*15)
        i = np.random.randint(len(W_samples_window))
        maximizer([W_samples_window[i]], x_batch, y_batch)
        if em_step == total_EM_steps:
            break
    print("#"*15,f"After {total_EM_steps} EM steps, fixing hyperparams and sample from posterior.","#"*15)
    _, log_p, mse = sampler_fixing_hyper(num_samples=num_samples_fixing_hyper,
                                         print_epoch_cycle=print_epoch_cycle_fixing)
    return log_p, mse

def MCEM_increasing_windows(sampler_EM, maximizer, sampler_fixing_hyper, total_EM_steps, ds_train,
                 num_samples_fixing_hyper=200, window_size=300,
                 print_epoch_cycle_EM=100, print_epoch_cycle_fixing=100):
    W_samples_window = []
    log_p_window = None
    mse_window = None

    ds_train_repeat = ds_train.repeat()
    em_step = 0
    for x_batch, y_batch in ds_train_repeat:
        em_step += 1
        # E step
        print("#"*15, f"EM step {em_step} of total {total_EM_steps} steps. E Step: ", "#"*15)
        W_samples, log_p, mse = sampler_EM(num_samples=1, print_epoch_cycle=print_epoch_cycle_EM)
        W_samples_window.extend(W_samples)
        if len(W_samples_window) == 1:
            log_p_window = log_p
            mse_window = mse
        elif len(W_samples_window) <= window_size:
            log_p_window = tf.concat([log_p_window, log_p],axis=0)
            mse_window = tf.concat([mse_window, mse], axis=0)
        else:
            W_samples_window = W_samples_window[-window_size:]
            log_p_window = tf.concat([log_p_window, log_p], axis=0)[1:,:]
            mse_window = tf.concat([mse_window, mse], axis=0)[1:,:]
        n_models = tf.shape(mse_window)[0]
        predict_log_p = tf.reduce_logsumexp(log_p_window, axis=0) - tf.math.log(tf.cast(n_models, tf.float32))
        predict_log_p = tf.reduce_mean(predict_log_p)
        predict_rmse = tf.math.sqrt(tf.reduce_mean(mse_window))
        print("*"*20," End of E step ", "*"*20)
        print(f"Number of all sampled models in window: {n_models} ")
        print(f"Test Log Likelihood of all models in window: {predict_log_p}")
        print(f"Test Root MSE of all models in window: {predict_rmse}\n")
        # M step
        print("#"*15, f"EM step {em_step} of total {total_EM_steps} steps, M Step: ", "#"*15)
        i = np.random.randint(len(W_samples_window))
        maximizer([W_samples_window[i]], x_batch, y_batch)
        if em_step == total_EM_steps:
            break
    print("#"*15,f"After {total_EM_steps} EM steps, fixing hyperparams and sample from posterior.","#"*15)
    _, log_p, mse = sampler_fixing_hyper(num_samples=num_samples_fixing_hyper,
                                         print_epoch_cycle=print_epoch_cycle_fixing)
    return log_p, mse
