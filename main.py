import os
import argparse
import random
import logging
import numpy as np
from PIL import Image
from absl import app, flags
import tensorflow as tf
from matplotlib import pyplot as plt

from evaluation.test import evaluate, reconstruct_images
from input_pipeline.dataset import ImageDataPipeline
from train import Trainer
from utilss import utils_params, utils_misc
import gin
from models.autoencoder import autoencoder

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--model', choices=['autoencoder', 'other_model'],
                    default='autoencoder', help='choose model')
parser.add_argument('--mode', choices=['train', 'test'], default='test', help='train or test')
parser.add_argument('--evaluation', choices=['evaluation', 'other_evalution_mathods'],
                    default='evaluation', help='evaluation methods')
parser.add_argument('--checkpoint-file', type=str, default='/home/wenwutang/PycharmProjects/experiments/run_2023-08-05T18-43-44-952339/ckpts',
                    help='Path to checkpoint.')

args = parser.parse_args()


# fix the seed to make the training repeatable
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


setup_seed(66)


@gin.configurable
def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    image_pipeline = ImageDataPipeline()

    # get datasets
    ds_train, ds_val, ds_test = image_pipeline.split_dataset()

    if args.model == 'autoencoder':
        model = autoencoder()
    elif args.model == 'other_model':
        pass
    else:
        print('Error, model does not exist')

    if args.mode == 'train':
        model.summary()
        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue
    elif args.mode == 'test':
        # model = tf.keras.models.load_model(args.checkpoint_file)
        checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=model)
        manager = tf.train.CheckpointManager(checkpoint, directory=args.checkpoint_file,max_to_keep=3)

        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            tf.print("restored")
        else:
            tf.print("Error")

        if args.evaluation == 'evaluation':
            evaluate(model, ds_test)
            for images in ds_test:
                predictions = model(images, training=False).numpy()
                images = images.numpy()
                print(predictions.shape)
                for i in range(32):
                    prediction = predictions[i]
                    plt.imshow(prediction)
                    plt.show()
                predictions = predictions.numpy()
                # pil_image = Image.fromarray((image_np * 255).astype('uint8'))


                # print(f'Saved image {idx + 1}/{batch_size}: {image_filename}')
                # Show the image using PIL
                # pil_image.show()
            # image = ds_test[0][0]
            # plt.imshow(image)
            # plt.show()

            #reconstruct_images(model, ds_test)
            # evaluate(model, ds_test)
            #print('#########################################', loss_threshold)
        elif args.evaluation == 'confusionmatrix':
            pass
            # confusionmatrix(model, ds_test)


if __name__ == "__main__":
    app.run(main)
