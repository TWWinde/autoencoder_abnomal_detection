import datetime
import os

import gin
import tensorflow as tf
import logging


@gin.configurable
class Trainer(object):

    def __init__(self, model, ds_train, ds_val, run_paths, total_steps, log_interval, ckpt_interval, num_epochs):
        # Summary Writer
        self.step = 0
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        current_dir = os.path.dirname(__file__)
        tensorboard_log_dir = os.path.join(current_dir, 'logs')
        log_dir = os.path.join(tensorboard_log_dir, current_time)
        logging.info(f"Tensorboard output will be stored in: {log_dir}")
        self.train_log_dir = os.path.join(log_dir, 'train')
        self.val_log_dir = os.path.join(log_dir, 'validation')

        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)

        # Loss objective

        self.loss_object = tf.keras.losses.MAE
        lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.001,
                                                                 decay_steps=1000,
                                                                 alpha=0.1)
        self.optimizer = tf.keras.optimizers.Adam(lr_scheduler)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.acc = 1
        self.num_epochs = num_epochs
        # ....
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=self.model, optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=run_paths["path_ckpts_train"],
                                                  max_to_keep=3)
        # Checkpoint Manager
        # ...

    @tf.function
    def train_step(self, images):
        with tf.GradientTape() as tape:
            images = tf.cast(images, dtype=tf.float32)
            predictions = self.model(images, training=True)
            loss = self.loss_object(images, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)

    @tf.function
    def val_step(self, images):

        images = tf.cast(images, dtype=tf.float32)
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(images, predictions)

        self.val_loss(t_loss)

    def write_scalar_summary(self, step):
        """ Write scalar summary to tensorboard """

        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', self.train_loss.result(), step=step)

        with self.val_summary_writer.as_default():
            tf.summary.scalar('loss', self.val_loss.result(), step=step)

    def train(self):
        logging.info(self.model.summary())
        logging.info('\n================ Starting Training ================')
        self.acc = 0.1
        self.step = 1
        for epoch in range(1, self.num_epochs + 1):
            logging.info(f'\nEpoch {epoch}/{self.num_epochs}:')
            for idx, images in enumerate(self.ds_train):
                self.step += 1
                self.train_step(images)

                if self.step % self.log_interval == 0:
                    # Reset test metrics
                    self.val_loss.reset_states()

                    for val_images in self.ds_val:
                        self.val_step(val_images)

                    template = 'Step {}, train MAE Loss: {}, Validation MSE Accuracy: {}'
                    logging.info(template.format(self.step, self.train_loss.result(), self.val_loss.result()))

                    # Write summary to tensorboard
                    self.write_scalar_summary(self.step)

                    # Reset train metrics
                    self.train_loss.reset_states()

                    yield self.val_loss.result().numpy()

                if self.step % self.ckpt_interval == 0:
                    if self.acc > self.val_loss.result():
                        self.acc = self.val_loss.result()
                        #tf.keras.models.save_model(self.model, self.run_paths["path_ckpts_train"])
                        logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                        path = self.manager.save()
                        print("model saved to %s" % self.run_paths["path_ckpts_train"])

            logging.info(f'Epoch {epoch}/{self.num_epochs} finished.')

        logging.info('\n================ Finished Training ================')


class Example:
    pass
