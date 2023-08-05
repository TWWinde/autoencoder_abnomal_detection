import os

import gin
import tensorflow as tf


def get_images(data_dir):
    image = open(data_dir, 'rb').read()
    return image


def get_image_list(root_path):
    image_paths = []
    for item in os.listdir(root_path):
        image_paths.append(os.path.join(root_path, item))

    return image_paths


# Use this function to serialize images and labels into the TFRecord format
@gin.configurable
def write_Tfrecord(root, save_path):
    img_path = get_image_list(root)
    with tf.io.TFRecordWriter(save_path + 'all_images.tfrecords') as writer:
        for i in range((len(img_path))):
            image_raw = get_images(img_path[i])
            feature = {'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    with tf.io.TFRecordWriter(save_path + 'train.tfrecords') as writer:
        for i in range(int((len(img_path) * 0.8))):
            image_raw = get_images(img_path[i])
            feature = {'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    with tf.io.TFRecordWriter(save_path + 'val.tfrecords') as writer:
        for i in range(int(0.8 * len(img_path)), int(0.9 * len(img_path))):
            image_raw = get_images(img_path[i])
            feature = {'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    with tf.io.TFRecordWriter(save_path + 'test.tfrecords') as writer:
        for i in range(int(0.9 * len(img_path)), len(img_path)):
            image_raw = get_images(img_path[i])
            feature = {'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


root = '/home/wenwutang/PycharmProjects/autoencoder_abnomal_detection/gazebo_images/normal_train/images'
save_path = '/home/wenwutang/PycharmProjects/autoencoder_abnomal_detection/gazebo_images/'
write_Tfrecord(root, save_path)
