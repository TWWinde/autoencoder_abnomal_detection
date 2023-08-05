import os
import gin
import tensorflow as tf


def get_image_list(root_path):
    image_paths = []
    for item in os.listdir(root_path):
        image_paths.append(os.path.join(root_path, item))

    return image_paths


@gin.configurable
class ImageDataPipeline:
    def __init__(self, root_path, batch_size, image_size):

        self.batch_size = batch_size
        self.image_size = image_size
        self.root_path = root_path
        self.image_paths = get_image_list(self.root_path)

    def preprocess_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([256, 256, 3])
        image = tf.image.resize(image, [256, 256])
        image = image / 255.0
        return image

    def create_dataset(self, image_paths):

        images_dataset = tf.data.Dataset.from_tensor_slices(image_paths)

        dataset = images_dataset.map(self.preprocess_image)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def split_dataset(self, train_split=0.8, val_split=0.1, shuffle=True):

        total_samples = len(self.image_paths)
        train_samples = int(train_split * total_samples)
        val_samples = int(val_split * total_samples)

        if shuffle:
            indices = tf.range(total_samples)
            indices = tf.random.shuffle(indices)
            image_paths = tf.gather(self.image_paths, indices)
        else:
            image_paths = self.image_paths

        train_image_paths = image_paths[:train_samples]
        val_image_paths = image_paths[train_samples:train_samples + val_samples]
        test_image_paths = image_paths[train_samples + val_samples:]

        train_dataset = self.create_dataset(train_image_paths)
        val_dataset = self.create_dataset(val_image_paths)
        test_dataset = self.create_dataset(test_image_paths)

        return train_dataset, val_dataset, test_dataset


