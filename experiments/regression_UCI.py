import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2

from gpflow.kernels import SquaredExponential, White
from gpflow.likelihoods import Gaussian
from gpflow.utilities import print_summary

from DGP import DeepGP
from datasets_valid import Datasets


def make_DGP(num_layers, X, Y, Z, num_samples=100):
    kernel_list = []
    D_in, D_out = X.shape[1], Y.shape[1]
    D_min = min(D_in, 30)
    layer_sizes = [D_in]
    for i in range(num_layers):
        kernel = SquaredExponential(variance=1.0, lengthscales=1.) + White(variance=1e-5)
        kernel_list.append(kernel)
        if i != num_layers -1:
            layer_sizes.append(D_min)
        else:
            layer_sizes.append(D_out)

    dgp = DeepGP(X, Y, Z, layer_sizes, kernel_list, likelihood=Gaussian(), num_samples=num_samples)

    for layer in dgp.layers[:-1]:
        layer.q_sqrt.assign(layer.q_sqrt.numpy()*1e-5)

    return dgp

def training_step(model, optimizer, X_train, Y_train, batch_size=1000):
    """
    one iteration of training
    """
    X_iter = tf.data.Dataset.from_tensor_slices(X_train).batch(batch_size).as_numpy_iterator()
    Y_iter = tf.data.Dataset.from_tensor_slices(Y_train).batch(batch_size).as_numpy_iterator()
    elbos = []

    for x_batch, y_batch in zip(X_iter, Y_iter):
        with tf.GradientTape(watch_accessed_variables=False) as tape_dgp:
            tape_dgp.watch(model.trainable_variables)
            loss_dgp = - model.ELBO((x_batch, y_batch))
        grad_dgp = tape_dgp.gradient(loss_dgp, model.trainable_variables)
        optimizer.apply_gradients(zip(grad_dgp, model.trainable_variables))
        elbos.append(-loss_dgp.numpy())
    return np.mean(elbos)

def evaluation_step(model, X_test, Y_test, batch_size=1000, num_samples=100):
    X_iter = tf.data.Dataset.from_tensor_slices(X_test).batch(batch_size).as_numpy_iterator()
    Y_iter = tf.data.Dataset.from_tensor_slices(Y_test).batch(batch_size).as_numpy_iterator()
    likelihoods, sq_diffs = [], []
    for x_batch, y_batch in zip(X_iter, Y_iter):
        mean, var = model.predict_y(x_batch, num_samples)  #(S, N, D)
        likelihood = model.predict_log_density(x_batch, y_batch, num_samples) #(N, )
        likelihoods.append(likelihood)

        mean = np.average(mean, 0) #（N, D）
        sq_diff = (mean - y_batch)**2 #（N, D）
        sq_diffs.append(sq_diff)
    likelihoods = np.concatenate(likelihoods, 0)
    sq_diffs = np.concatenate(sq_diffs, 0)
    return np.average(likelihoods), np.average(sq_diffs)**0.5


datasets = Datasets(data_path="./data/")
dataset_name = "power"
data = datasets.all_datasets[dataset_name].get_data()
X, Y, Xs, Ys, Xv, Yv, Y_std = [data[_] for _ in ["X", "Y", "Xs", "Ys", "Xv", "Yv", "Y_std"]]
# (Xs, Ys) is for testing
print("# " * 20)
print("dataset name:", dataset_name)
print(f"N: {X.shape[0]}, D: {X.shape[1]}, Ns: {Xs.shape[0]}, Nv: {Xv.shape[0]}")

num_inducing = 100
num_hidden_layer = 3
num_samples = 100
num_iter = 500

Z_init = kmeans2(X, num_inducing, minit="points")[0]

dgp = make_DGP(num_hidden_layer, X, Y, Z_init, num_samples=num_samples)
optimizer = tf.optimizers.Adam(learning_rate=0.01, epsilon=1e-8)

batch_size_train = min(X.shape[0], 1000)
batch_size_val = min(Xv.shape[0], 1000)
batch_size_test = min(Xs.shape[0], 1000)

# if dataset_name == "power":
#     elbo_list = np.zeros(num_iter)
#     like_train_list = np.zeros(num_iter)
#     like_val_list = np.zeros(num_iter)
#     like_test_list = np.zeros(num_iter)
#     rmse_train_list = np.zeros(num_iter)
#     rmse_val_list =np.zeros(num_iter)
#     rmse_test_list = np.zeros(num_iter)

max_likelihood_info = {"iteration": -1, "elbo": None,
                       "training_likelihood": None, "training_rmse": None,
                       "validation_likelihhood": None, "validation_rmse": None,
                       "test_likelihood": None, "test_rmse": None}
min_rmse_info = {"iteration": -1, "elbo": None,
                 "training_likelihood": None, "training_rmse": None,
                 "validation_likelihhood": None, "validation_rmse": None,
                 "test_likelihood": None, "test_rmse": None}

def fill_in_dict(info_dict, iteration, elbo, lik_train, rmse_train, lik_val, rmse_val, lik_test, rmse_test):
    info_dict["iteration"] = iteration
    info_dict["elbo"] = elbo
    info_dict["training_likelihood"] = lik_train
    info_dict["training_rmse"] = rmse_train
    info_dict["validation_likelihood"] = lik_val
    info_dict["validation_rmse"] = rmse_val
    info_dict["test_likelihood"] = lik_test
    info_dict["test_rmse"] = rmse_test
    return None


print_summary(dgp)
start = time.time()
breaking_condition = 50
for i in range(num_iter):
    elbo = training_step(dgp, optimizer, X, Y, batch_size=batch_size_train)
    lik_train, rmse_train = evaluation_step(dgp, X, Y, batch_size=batch_size_train, num_samples=num_samples)
    lik_val, rmse_val = evaluation_step(dgp, Xv, Yv, batch_size=batch_size_val, num_samples=num_samples)
    lik_test, rmse_test = evaluation_step(dgp, Xs, Ys, batch_size=batch_size_test, num_samples=num_samples)
    if i == 0:
        fill_in_dict(max_likelihood_info, i, elbo, lik_train, rmse_train, lik_val, rmse_val, lik_test, rmse_test)
        fill_in_dict(min_rmse_info, i, elbo, lik_train, rmse_train, lik_val, rmse_val, lik_test, rmse_test)
    else:
        if lik_val >= max_likelihood_info["validation_likelihood"]:
            fill_in_dict(max_likelihood_info, i, elbo, lik_train, rmse_train, lik_val, rmse_val, lik_test, rmse_test)
        if rmse_val <= min_rmse_info["validation_rmse"]:
            fill_in_dict(min_rmse_info, i, elbo, lik_train, rmse_train, lik_val, rmse_val, lik_test, rmse_test)

    print(f"Iteration {i}, elbo is {elbo},  training likelihood {lik_train}, training rmse {rmse_train}, ")
    print(f"                                val likelihood {lik_val}, val rmse {rmse_val}")
    print(f"                                test likelihood {lik_test}, test rmse {rmse_test}")
    print("     max validation likelihood iteration: ", max_likelihood_info["iteration"],
          "test likelihood:", max_likelihood_info["test_likelihood"],
          "test rmse:", max_likelihood_info["test_rmse"])
    print("     min validation rmse iteration: ", min_rmse_info["iteration"],
          "test likelihood:", min_rmse_info["test_likelihood"],
          "test rmse:", min_rmse_info["test_rmse"])

    max_iter_gap = i - max_likelihood_info["iteration"]
    min_iter_gap = i - min_rmse_info["iteration"]
    if max_iter_gap >= breaking_condition and min_iter_gap >= breaking_condition:
        break

    # if dataset_name == "power":
    #     elbo_list[i] = elbo
    #     like_train_list[i] = lik_train
    #     like_val_list[i] = lik_val
    #     like_test_list[i] = lik_test
    #     rmse_train_list[i] = rmse_train
    #     rmse_val_list[i] = rmse_val
    #     rmse_test_list[i] = rmse_test
end = time.time()
print_summary(dgp)
print(f"breaking at {i}: time is {end - start}; each epoch costs {(end - start) / (i+1)}")


# with open("elbo.npy0", "wb") as f :
#     np.save(f, elbo_list)
#
# with open("lik_train.npy0", "wb") as f :
#     np.save(f, like_train_list)
#
# with open("lik_val.npy0", "wb") as f :
#     np.save(f, like_val_list)
#
# with open("lik_test.npy0", "wb") as f :
#     np.save(f, like_test_list)
#
# with open("rmse_train.npy0", "wb") as f :
#     np.save(f, rmse_train_list)
#
# with open("rmse_val.npy0", "wb") as f :
#     np.save(f, rmse_val_list)
    
# with open("rmse_test.npy0", "wb") as f :
#     np.save(f, rmse_test_list)
