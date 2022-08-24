import numpy as np
import tensorflow as tf

import sys
sys.path.append("..")

from utils import cyclical_step_rate


def regression_train_demo(model_demo, ds_train, ds_test, train_size, batch_size, X_test,
                          lr_0=0.01, momentum_decay=0.9,
                          resample_in_cycle_head = True,
                          total_epochs=5000, start_sampling_epoch=2000, epochs_per_cycle=50,
                          print_epoch_cycle=100):
    if train_size % batch_size != 0:
        raise ValueError(f"In the demo, train size {train_size} should be exactly divided by batch size {batch_size}.")
    # no preconditioner
    ds_M = None
    iterations_per_epoch = int(train_size/batch_size)  # force float to int

    cycle_length = epochs_per_cycle * iterations_per_epoch #number of iterations in one period

    log_p = []
    mse = []
    lines = []
    W = {}
    for i in range(model_demo.n_hidden_layers):
        W.update({'W_'+str(i): []})
    for epoch in range(total_epochs):
        model_demo.precond_update(ds_M, train_size, precond_type='identity')
        batch_index = 0
        for img_batch, label_batch in ds_train:
            batch_index = batch_index + 1
            if epoch < start_sampling_epoch: # fixed learning rate, zero temperature
                model_demo.sgmcmc_update(img_batch, label_batch, train_size,
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
                model_demo.sgmcmc_update(img_batch, label_batch, train_size,
                                    lr=lr, momentum_decay=momentum_decay,
                                    resample_moments=is_new_cycle, temperature=1.)
                if is_end: # sampling the model
                    test_log_p, test_se = model_demo.eval_log_likelihood_and_se(ds_test)
                    log_p.append(test_log_p)
                    mse.append(test_se)
                    # collect lines
                    line_sampled = model_demo.feed_forward_all_layers(X_test)
                    lines.append(line_sampled)

                    W_sampled = model_demo.collect_W()
                    for i in range(model_demo.n_hidden_layers):
                        W['W_'+str(i)].append(W_sampled['W_'+str(i)])

                    print('#' * 20, f'Sampling at Epoch {epoch} ', f"lr = {lr}", '#' * 20)
        # print sampling process
        if (epoch + 1) % print_epoch_cycle == 0:
            train_log_p, train_se = model_demo.eval_log_likelihood_and_se(ds_train)
            test_log_p, test_se = model_demo.eval_log_likelihood_and_se(ds_test)
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

    return log_p, mse, lines, W

def MCEM_sampler_demo(model_demo, ds_train, ds_test, train_size, batch_size, X_test,
                      lr_0=0.01, momentum_decay=0.9, resample_in_cycle_head=False,
                      start_sampling_epoch=0, epochs_per_cycle=50):
    def sampler(num_samples=100, return_lines_Wdict=False, print_epoch_cycle=100):
        if train_size % batch_size != 0:
            raise ValueError(f"In the demo, train size {train_size} should be exactly divided by batch size {batch_size}.")
        # no preconditioner
        ds_M = None
        iterations_per_epoch = int(train_size/batch_size)  # force float to int
        cycle_length = epochs_per_cycle * iterations_per_epoch #number of iterations in one period
        total_epochs = start_sampling_epoch + num_samples * epochs_per_cycle

        W_samples = []
        log_p = []
        mse = []
        if return_lines_Wdict:
            lines = []
            W = {}
            for i in range(model_demo.n_hidden_layers):
                W.update({'W_'+str(i): []})

        sampled_model_index = 0
        for epoch in range(total_epochs):
            model_demo.precond_update(ds_M, train_size, precond_type='identity', full_bayesian=False)
            batch_index = 0
            for img_batch, label_batch in ds_train:
                batch_index = batch_index + 1
                if epoch < start_sampling_epoch: # fixed learning rate, zero temperature
                    model_demo.sgmcmc_update(img_batch, label_batch, train_size, lr=lr_0, momentum_decay=momentum_decay,
                                             resample_moments=False, temperature=0., full_bayesian=False)
                else: # cyclical learning rate, non-zero temperature
                    step_index = (epoch - start_sampling_epoch) * iterations_per_epoch + batch_index
                    step_rate, is_end = cyclical_step_rate(step_index, cycle_length, schedule='cosine', min_value=0.)
                    lr = lr_0 * (step_rate**2)
                    if resample_in_cycle_head == True:
                        is_new_cycle = tf.equal(tf.math.mod(step_index, cycle_length), 1)
                    else:
                        is_new_cycle = False
                    model_demo.sgmcmc_update(img_batch, label_batch, train_size, lr=lr, momentum_decay=momentum_decay,
                                             resample_moments=is_new_cycle, temperature=1., full_bayesian=False)
                    if is_end: # sampling the model
                        sampled_model_index += 1
                        W_samples.append(model_demo.W_mcmc) # W_mcmc is a list
                        print('#' * 20, f'Sample No.{sampled_model_index} at Epoch {epoch} ', f"lr = {lr}", '#' * 20)

                        test_log_p, test_se = model_demo.eval_log_likelihood_and_se(ds_test)
                        log_p.append(test_log_p)
                        mse.append(test_se)
                        if return_lines_Wdict: # collect lines
                            line_sampled = model_demo.feed_forward_all_layers(X_test)
                            lines.append(line_sampled)
                            W_sampled_dict = model_demo.collect_W()
                            for i in range(model_demo.n_hidden_layers):
                                W['W_'+str(i)].append(W_sampled_dict['W_'+str(i)])
            # print sampling process
            if (epoch + 1) % print_epoch_cycle == 0:
                train_log_p, train_se = model_demo.eval_log_likelihood_and_se(ds_train)
                test_log_p, test_se = model_demo.eval_log_likelihood_and_se(ds_test)
                print(f"Sampling Epoch: {epoch}")
                print(f"Mean Log Likelihood -- train: {tf.reduce_mean(train_log_p)}, "
                      f"-- test: {tf.reduce_mean(test_log_p)} ")
                print(f"Root Mean Squared Error -- train: {tf.math.sqrt(tf.reduce_mean(train_se))}, "
                      f"-- test: {tf.math.sqrt(tf.reduce_mean(test_se))} \n")

        log_p = tf.stack(log_p, axis=0) # [S, N]
        mse = tf.stack(mse, axis=0) #[S, N]

        n_models = tf.shape(mse)[0]
        predict_log_p = tf.reduce_logsumexp(log_p, axis=0) - tf.math.log(tf.cast(n_models, tf.float32))
        predict_log_p = tf.reduce_mean(predict_log_p)
        predict_rmse = tf.math.sqrt(tf.reduce_mean(mse))

        print("*"*20, " End of Sampling ", "*"*20)
        print(f"Number of sampled models: {n_models} ")
        print(f"Test Log Likelihood of all sampled models: {predict_log_p}")
        print(f"Test Root MSE of all sampled models: {predict_rmse}")
        print("*"*70,"\n")

        if return_lines_Wdict:
            return W_samples, log_p, mse, lines, W
        else:
            return W_samples, log_p, mse
    return sampler

def MCEM_Q_maximizer_demo(model_demo, data_size, optimizer):
    def maximizer(W_samples, X_batch, Y_batch):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch([model_demo.Omega_hyperparams, model_demo.Likelihood_hyperparams])
            Q = 0.
            num_samples = 0
            for W_model_list in W_samples:
                model_demo.assign_W(W_model_list)
                # regard W as constant and retain the computing graph
                log_p_sample = - model_demo.U(X_batch, Y_batch, data_size,
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

def MCEM_demo(sampler_EM, maximizer, sampler_fixing_hyper, total_EM_steps, ds_train,
         num_samples_EM=100, num_samples_fixing_hyper=200,
         print_epoch_cycle_EM=100, print_epoch_cycle_fixing=100):
    ds_train_repeat = ds_train.repeat()
    em_step = 0
    for x_batch, y_batch in ds_train_repeat:
        em_step += 1
        # E step
        print("#"*15, f"EM step {em_step} of total {total_EM_steps} steps. E Step: ", "#"*15)
        W_samples, _, _ = sampler_EM(num_samples=num_samples_EM, return_lines_Wdict=False,
                                     print_epoch_cycle=print_epoch_cycle_EM)
        # M step
        print("#"*15, f"EM step {em_step} of total {total_EM_steps} steps, M Step: ", "#"*15)
        maximizer(W_samples, x_batch, y_batch)
        if em_step == total_EM_steps:
            break
    print("#"*15,f"After {total_EM_steps} EM steps, fixing hyperparams and sample from posterior.","#"*15)
    _, log_p, mse, lines, W_dict = sampler_fixing_hyper(num_samples=num_samples_fixing_hyper,
                                                        return_lines_Wdict=True,
                                                        print_epoch_cycle=print_epoch_cycle_fixing)
    return log_p, mse, lines, W_dict

def MCEM_windows_demo(sampler_EM, maximizer, sampler_fixing_hyper, total_EM_steps, ds_train,
                      num_samples_fixing_hyper=200, window_size=50,
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
        if len(W_samples_window) < window_size:
            W_samples, log_p_window, mse_window = sampler_EM(num_samples=window_size,
                                                             return_lines_Wdict=False,
                                                             print_epoch_cycle=print_epoch_cycle_EM)
            W_samples_window.extend(W_samples)
        else:
            W_samples, log_p, mse = sampler_EM(num_samples=1, return_lines_Wdict=False,
                                               print_epoch_cycle=print_epoch_cycle_EM)
            W_samples_window.extend(W_samples)
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
    _, log_p, mse, lines, W_dict = sampler_fixing_hyper(num_samples=num_samples_fixing_hyper,
                                                        return_lines_Wdict=True,
                                                        print_epoch_cycle=print_epoch_cycle_fixing)
    return log_p, mse, lines, W_dict