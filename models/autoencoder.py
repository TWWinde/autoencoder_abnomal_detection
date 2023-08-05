#import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3,3), padding='same', activation=tf.nn.relu, input_shape=(96, 96, 3)),
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same'),
            tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same'),
            tf.keras.layers.Conv2D(3, (3,3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),

            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((4, 4)),

            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((4, 4)),
            tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


x = np.random.random((1, 96, 96, 3))
autoencoder= Autoencoder()




def encoder(inputs, kernel_size = (3, 3)):
    out = tf.keras.layers.Conv2D(16, kernel_size, padding='same', activation=tf.nn.relu, input_shape=(96, 96, 3))(
        inputs)
    out = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same')(out)
    out = tf.keras.layers.Conv2D(8, kernel_size, activation='relu', padding='same')(out)
    out = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same')(out)
    out = tf.keras.layers.Conv2D(3, kernel_size, activation='relu', padding='same')(out)
    out = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(out)

    return out


def decoder(inputs):
    out = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(96, 96, 3))(inputs)
    out = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same')(out)
    out = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(out)
    out = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same')(out)
    out = tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same')(out)
    out = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(out)

    return out

latent = encoder(x)
y = decoder(latent)
print(y.shape)
