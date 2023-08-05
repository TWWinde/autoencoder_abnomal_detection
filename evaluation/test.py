import gin
import numpy as np
import tensorflow as tf
import logging


# This function is to test normal pictures and to calculate the loss threshold
def evaluate(model, ds_test):
    loss_object = tf.keras.losses.MAE(from_logits=True)
    eval_accuracy = tf.keras.metrics.MSE(name='eval_accuracy')
    eval_loss = tf.keras.metrics.MAE(name='eval_loss')
    losses = []
    for images in ds_test:
        predictions = model(images, training=False)
        t_loss = loss_object(images, predictions)
        losses.append(t_loss)
        eval_loss(t_loss)

        eval_accuracy(images, predictions)

    sorted_losses = sorted(losses)
    threshold_idx = int(0.95 * len(sorted_losses))
    loss_threshold = sorted_losses[threshold_idx]
    template = ' Test Loss: {}, Test Accuracy: {} loss_threshold {} '
    logging.info(template.format(eval_loss.result(), eval_accuracy.result() * 100, loss_threshold))

    return loss_threshold
