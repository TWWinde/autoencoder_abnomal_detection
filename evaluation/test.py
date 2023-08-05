import gin
import numpy as np
import tensorflow as tf
import logging


def reconstruct_images(model, ds_test, num_images=5):
    fig, axes = plt.subplots(2, num_images, figsize=(10, 4))

    for i, image in enumerate(ds_test.take(num_images)):
        original_image = image[0].numpy()
        print(original_image)
        reconstructed_image = model(image)[0].numpy()

        axes[0, i].imshow(original_image)
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')

        axes[1, i].imshow(reconstructed_image)
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')

    plt.show()
# This function is to test normal pictures and to calculate the loss threshold
def evaluate(model, ds_test):
    loss_object = tf.keras.losses.MAE
    eval_loss = tf.keras.metrics.Mean(name='test_loss')
    losses = []
    for images in ds_test:
        print(images)
        print('###################################################################')
        predictions = model(images, training=False)
        plot_images(images, predictions, num_images=5)
        plt.tight_layout()
        plt.show()
        print('###################################################################')
        t_loss = loss_object(images, predictions)
        print(t_loss)
        losses.append(t_loss)
        eval_loss(t_loss)

    sorted_losses = sorted(losses)
    threshold_idx = int(0.95 * len(sorted_losses))
    print(len(sorted_losses))
    loss_threshold = sorted_losses[threshold_idx]
    template = ' Test Loss: {},  loss_threshold {} '
    logging.info(template.format(eval_loss.result(), loss_threshold))
    print(loss_threshold)
    return


import matplotlib.pyplot as plt
import numpy as np


def plot_images(images, predictions, num_images=5):
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 10))
    for i in range(num_images):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(predictions[i])
        axes[i, 1].set_title('Predicted Image')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


# 假设你已经运行了模型并得到了images和predictions，它们是numpy数组
# images是原始图像，predictions是模型预测的图像
# 假设每个数组包含5张图像，你可以根据需要调整num_images的值
num_images = 5
