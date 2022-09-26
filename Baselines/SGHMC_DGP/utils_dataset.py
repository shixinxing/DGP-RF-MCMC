import numpy as np
import tensorflow as tf
# import tensorflow_datasets as tfds
from datasets import Datasets


def transform_UCI_tfds(X_train, Y_train, X_test, Y_test):
    ds_X_train = tf.data.Dataset.from_tensor_slices(X_train)
    ds_Y_train = tf.data.Dataset.from_tensor_slices(Y_train)
    ds_train = tf.data.Dataset.zip((ds_X_train, ds_Y_train))
    ds_X_test = tf.data.Dataset.from_tensor_slices(X_test)
    ds_Y_test = tf.data.Dataset.from_tensor_slices(Y_test)
    ds_test = tf.data.Dataset.zip((ds_X_test, ds_Y_test))
    return ds_train, ds_test

def download_UCI_data_info(name, data_path='./data/'):
    datasets = Datasets(data_path=data_path)
    dataset = datasets.all_datasets[name]
    data = dataset.get_data()
    X, Y, Xs, Ys, X_mean, Y_mean, Y_std = [np.float32(data[_]) for _ in ['X', 'Y', 'Xs', 'Ys', 'X_mean','Y_mean','Y_std']]

    assert dataset.N == X.shape[0] + Xs.shape[0], f"N + Ns does not match dataset.N (should be {X.shape[0] + Xs.shape[0]})! "
    assert dataset.D == X.shape[1], f"D does not match dataset.D(should be {X.shape[1]})!"
    print('#' * 30 + f" Getting data info:dataset name: {name} " + '#' * 30)
    print(f"D: {X.shape[1]}, N: {X.shape[0]}, Ns: {Xs.shape[0]}")
    print(f"X_mean: {X_mean}, Y_mean: {Y_mean}, Y_std: {Y_std}")
    print('#' * 70)
    return X, Y, Xs, Ys, X_mean, Y_mean, Y_std #Y_mean, Y_std shape [1,]

def load_UCI_dataset(dataset_name, batch_size=128, transform_fn=None, data_dir='./data/'):
    print('#'*30 + f" Getting data info:dataset name: {dataset_name} " + '#'*30)
    X, Y, Xs, Ys, X_mean, Y_mean, Y_std = download_UCI_data_info(dataset_name, data_path=data_dir)
    print(f"D: {X.shape[1]}, N: {X.shape[0]}, Ns: {Xs.shape[0]}")
    print(f"X_mean: {X_mean}, Y_mean: {Y_mean}, Y_std: {Y_std}")
    print('#' * 70)
    train_shape = np.shape(X)
    test_shape = np.shape(Xs)
    ds_train, ds_test = transform_UCI_tfds(X, Y, Xs, Ys)
    if transform_fn is not None: # transform data element-wise
        ds_train = ds_train.map(transform_fn)
        ds_test = ds_test.map(transform_fn)
    ds_train = ds_train.shuffle(train_shape[0])
    ds_test = ds_test.shuffle(test_shape[0])
    ds_train = ds_train.batch(batch_size, drop_remainder=True)  # drop remainder to prevent a small final batch
    ds_test = ds_test.batch(batch_size, drop_remainder=False)
    return ds_train, ds_test, train_shape, test_shape

# def load_tf_dataset(dataset_name, batch_size=128, transform_fn=None, data_dir='./tensorflow_datasets/'):
#     ds_train, ds_info = tfds.load(dataset_name, split='train', shuffle_files=True,
#                                   data_dir=data_dir, as_supervised=True, with_info=True)
#     ds_test = tfds.load(dataset_name, split='test', shuffle_files=True,
#                         data_dir=data_dir, as_supervised=True)
#     if transform_fn is not None: # transform data element-wise
#         ds_train = ds_train.map(transform_fn)
#         ds_test = ds_test.map(transform_fn)
#     train_full_size = ds_info.splits['train'].num_examples
#     test_full_size = ds_info.splits['test'].num_examples
#     ds_train = ds_train.shuffle(train_full_size)
#     ds_test = ds_test.shuffle(test_full_size)
#     ds_train = ds_train.batch(batch_size, drop_remainder=True)
#     ds_test = ds_test.batch(batch_size, drop_remainder=False)
#     return ds_train, ds_test, train_full_size, test_full_size # return tf.data.Dataset as iterable and full data size
#
# def normalize_MNIST(img, label):
#     img = tf.cast(tf.reshape(img, [28 * 28]), tf.float32) / 255. - 0.5
#     label = tf.cast(tf.reshape(label, [1]), tf.float32)
#     return img, label