import gin
import numpy as np
import tensorflow as tf


@gin.configurable
def autoencoder(input_shape):
    
    inputs = tf.keras.Input(input_shape)
    latent_variables = encoder(inputs)
    outputs = decoder(latent_variables)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='autoencoder')


@gin.configurable
def encoder(inputs, kernel_size, latent_dim):
    out = tf.keras.layers.Conv2D(16, kernel_size, padding='same', activation=tf.nn.relu, input_shape=(96, 96, 3))(
        inputs)
    out = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same')(out)
    out = tf.keras.layers.Conv2D(8, kernel_size, activation='relu', padding='same')(out)
    out = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same')(out)
    out = tf.keras.layers.Conv2D(3, kernel_size, activation='relu', padding='same')(out)
    out = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(out)  # (8,8,3)
    out = tf.keras.layers.Flatten()(out)  # latentsize (1, 192)
    out = tf.keras.layers.Dense(latent_dim, activation='relu')(out)

    return out


@gin.configurable
def decoder(inputs, kernel_size):
    out = tf.keras.layers.Dense(192, activation='relu')(inputs)
    out = tf.keras.layers.Reshape((8, 8, 3))(out)

    out = tf.keras.layers.Conv2D(3, kernel_size , activation='relu', padding='same')(out)
    out = tf.keras.layers.UpSampling2D((2, 2))(out)

    out = tf.keras.layers.Conv2D(8, kernel_size, activation='relu', padding='same')(out)
    out = tf.keras.layers.UpSampling2D((4, 4))(out)

    out = tf.keras.layers.Conv2D(16, kernel_size, activation='relu', padding='same')(out)
    out = tf.keras.layers.UpSampling2D((4, 4))(out)

    out = tf.keras.layers.Conv2D(3, kernel_size, activation='sigmoid', padding='same')(out)

    return out


if '__name__' == '__mian__':
    # test the output.shape
    x = np.random.random((1, 256, 256, 3))
    latent = encoder(x, (3, 3), 50)
    print('latentsize', latent.shape)
    y = autoencoder(x)
    print('outputsize', y[0].shape)
