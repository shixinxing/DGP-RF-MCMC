## Copyright 2016 Kurt Cutajar, Edwin V. Bonilla, Pietro Michiardi, Maurizio Filippone
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import sys
sys.path.append(".")


from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

from dataset import DataSet
import utils
import likelihoods
from dgp_rff import DgpRff
import tensorflow as tf
import numpy as np
import losses

from dataset_adapt import Datasets

# def import_dataset(dataset, fold):
#
#     train_X = np.loadtxt('FOLDS/' + dataset + '_ARD_Xtrain__FOLD_' + fold, delimiter=' ')
#     train_Y = np.loadtxt('FOLDS/' + dataset + '_ARD_ytrain__FOLD_' + fold, delimiter=' ')
#     train_Y = np.reshape(train_Y, (-1, 1))
#     test_X = np.loadtxt('FOLDS/' + dataset + '_ARD_Xtest__FOLD_' + fold, delimiter=' ')
#     test_Y = np.loadtxt('FOLDS/' + dataset + '_ARD_ytest__FOLD_' + fold, delimiter=' ')
#     test_Y = np.reshape(test_Y, (-1, 1))
#
#     data = DataSet(train_X, train_Y)
#     test = DataSet(test_X, test_Y)
#
#     return data, test

def download_UCI_data_info(name, data_path='./data/'):
    datasets = Datasets(data_path=data_path)
    dataset = datasets.all_datasets[name]
    data = dataset.get_data()
    X, Y, Xs, Ys, X_mean, Y_mean, Y_std = [np.float32(data[_]) for _ in ['X', 'Y', 'Xs', 'Ys', 'X_mean','Y_mean','Y_std']]
    Y = Y * Y_std
    Ys = Ys * Y_std

    assert dataset.N == X.shape[0] + Xs.shape[0], f"N + Ns does not match dataset.N (should be {X.shape[0] + Xs.shape[0]})! "
    assert dataset.D == X.shape[1], f"D does not match dataset.D(should be {X.shape[1]})!"
    data = DataSet(X, Y)
    test = DataSet(Xs, Ys)

    return data, test #Y_mean, Y_std shape [1,]

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

if __name__ == '__main__':
    record_file = 'UCI_results_3_layers.txt'
    dataset_name_list = ['energy']
    d_in_list = [8]
    batch_size_list = [200]
    nl = 3
    repeat_times = 3

    for i in range(len(dataset_name_list)):
        dataset_name = dataset_name_list[i]
        MLL_coll = []
        RMSE_coll = []
        for repeat in range(repeat_times):
            print('*'*25, f' Run {repeat} ', '*'*25)
            FLAGS = utils.get_flags()
            ## Set random seed for tensorflow and numpy operations
            FLAGS.seed = FLAGS.seed + repeat
            tf.set_random_seed(FLAGS.seed)
            np.random.seed(FLAGS.seed)
            FLAGS.dataset = dataset_name
            FLAGS.nl = nl
            FLAGS.df = d_in_list[i]
            FLAGS.batch_size = batch_size_list[i]
            # adapting and using other regression datasets
            # data, test = import_dataset(FLAGS.dataset, FLAGS.fold)
            data, test = download_UCI_data_info(FLAGS.dataset, data_path='./data/')
            print(70*"#")
            print(f"Training dataset shape:  X: {[data.num_examples, data.Din]}, Y: {[data.num_examples, data.Dout]}")
            print(f"Test dataset shape: Xs: {[test.num_examples, test.Din]}, Ys: {[test.num_examples, test.Dout]}")
            print(70*"#")

            ## Here we define a custom loss for dgp to show
            error_rate = losses.RootMeanSqError(data.Dout)

            ## Likelihood
            like = likelihoods.Gaussian()

            ## Optimizer
            optimizer = utils.get_optimizer(FLAGS.optimizer, FLAGS.learning_rate)

            ## Main dgp object
            dgp = DgpRff(like, data.num_examples, data.X.shape[1], data.Y.shape[1], FLAGS.nl, FLAGS.n_rff, FLAGS.df,
                         FLAGS.kernel_type, FLAGS.kernel_arccosine_degree, FLAGS.is_ard, FLAGS.local_reparam,
                         FLAGS.feed_forward, FLAGS.q_Omega_fixed, FLAGS.theta_fixed, FLAGS.learn_Omega)

            ## Learning
            MLL, RMSE = dgp.learn(data, FLAGS.learning_rate, FLAGS.mc_train, FLAGS.batch_size, FLAGS.n_iterations, optimizer,
                         FLAGS.display_step, test, FLAGS.mc_test, error_rate, FLAGS.duration, FLAGS.less_prints)
            MLL_coll.append(MLL)
            RMSE_coll.append(RMSE)

            total_epochs = FLAGS.n_iterations // (data.num_examples // FLAGS.batch_size)
            with open(record_file,'a') as f:
                f.write(f'Dataset: {FLAGS.dataset}, Run {repeat}, Total epochs {total_epochs} \n')
                f.write(f"MLL: {MLL}, RMSE: {RMSE}\n\n")

            del_all_flags(FLAGS) #delete all flags

        with open(record_file,'a') as f:
            f.write(f'Dataset: {dataset_name}\n')
            f.write(f'MLL: {MLL_coll}; RMSE: {RMSE_coll}\n')
            f.write(f'MLL mean: {np.mean(MLL_coll)}, std: {np.std(MLL_coll)}\n')
            f.write(f'RMSE mean: {np.mean(RMSE_coll)}, std: {np.std(RMSE_coll)}\n\n')