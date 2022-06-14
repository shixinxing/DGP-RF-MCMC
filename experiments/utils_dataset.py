import tensorflow as tf
import tensorflow_datasets as tfds
from datasets import Datasets

def load_UCI_data(name, data_path='./data/'):
    datasets = Datasets(data_path=data_path)
    dataset = datasets.all_datasets[name]
    data = dataset.get_data()
    X, Y, Xs, Ys, X_mean, X_std, Y_mean = [data[_] for _ in ['X', 'Y', 'Xs', 'Ys', 'X_mean', 'X_std', 'Y_mean']]
    print('#'*30 + f"Getting data info:dataset name: {dataset.name}" + '#'*30)
    print(f"D: {X.shape[1]}, N: {X.shape[0]}, Ns: {Xs.shape[0]}")
    print(f"X_mean: {X_mean}, X_std: {X_std}, Y_mean: {Y_mean}")
    print('#'*70)
    return X, Y, Xs, Ys, X_mean, X_std, Y_mean


def load_dataset(dataset_name, batch_size=128, transform_fn=None, data_dir='./tensorflow_datasets/'):
    ds_train, ds_info = tfds.load(dataset_name, split='train', shuffle_files=True,
                                  data_dir=data_dir, as_supervised=True, with_info=True)
    ds_test = tfds.load(dataset_name, split='test', shuffle_files=True,
                        data_dir=data_dir, as_supervised=True)
    if transform_fn is not None: # transform data element-wise
        ds_train = ds_train.map(transform_fn)
        ds_test = ds_test.map(transform_fn)
    train_full_size = ds_info.splits['train'].num_examples
    test_full_size = ds_info.splits['test'].num_examples
    ds_train = ds_train.shuffle(train_full_size)
    ds_test = ds_test.shuffle(test_full_size)
    ds_train = ds_train.batch(batch_size)
    ds_test = ds_test.batch(batch_size)
    return ds_train, ds_test, train_full_size, test_full_size # return tf.data.Dataset as iterable and full data size

def normalize_MNIST(img, label):
    img = tf.cast(tf.reshape(img, [28 * 28]), tf.float32) / 255. - 0.5
    label = tf.cast(tf.reshape(label, [1]), tf.float32)
    return img, label


