import gin
import logging
import tensorflow as tf
from keras import datasets as tfds


def read_labeled_tfrecord(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string)}
    example = tf.io.parse_single_example(example, features)
    image = tf.io.decode_jpeg(example['image'], channels=3) / 255
    label = example['label']

    return image, label  # returns a dataset of (image, label) pairs


def get_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames)  # automatically interleaves reads from multiple files
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls=AUTOTUNE)

    return dataset


@gin.configurable
def load(name, save_path):  # (name, data_dir)

    logging.info(f"Preparing dataset {name}...")
    ds_test = get_dataset(save_path + 'test.tfrecords')
    ds_train = get_dataset(save_path + 'train.tfrecords')
    ds_val = get_dataset(save_path + 'validation.tfrecords')

    return prepare(ds_train, ds_val, ds_test)


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
    if caching:
        ds_train = ds_train.cache()

    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info
