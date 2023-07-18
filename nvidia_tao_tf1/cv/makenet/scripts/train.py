# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Makenet training script with protobuf configuration."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
import json
import logging
import os
import sys

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
import six

import tensorflow as tf
from nvidia_tao_tf1.core.utils import set_random_seed
from nvidia_tao_tf1.cv.common.callbacks.loggers import TAOStatusLogger
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.mlops.clearml import get_clearml_task
from nvidia_tao_tf1.cv.common.model_parallelism.parallelize_model import model_parallelism
from nvidia_tao_tf1.cv.common.utils import check_tf_oom, hvd_keras, restore_eff
from nvidia_tao_tf1.cv.common.utils import OneIndexedCSVLogger as CSVLogger
from nvidia_tao_tf1.cv.common.utils import TensorBoard
from nvidia_tao_tf1.cv.makenet.model.model_builder import get_model
from nvidia_tao_tf1.cv.makenet.spec_handling.spec_loader import load_experiment_spec
from nvidia_tao_tf1.cv.makenet.utils import preprocess_crop  # noqa pylint: disable=unused-import
from nvidia_tao_tf1.cv.makenet.utils.callbacks import AdvModelCheckpoint
from nvidia_tao_tf1.cv.makenet.utils.helper import (
    build_lr_scheduler,
    build_optimizer,
    model_io,
    setup_config
)
from nvidia_tao_tf1.cv.makenet.utils.mixup_generator import MixupImageDataGenerator
from nvidia_tao_tf1.cv.makenet.utils.preprocess_input import preprocess_input


ImageFile.LOAD_TRUNCATED_IMAGES = True
FLAGS = tf.app.flags.FLAGS

logger = logging.getLogger(__name__)
verbose = 0
hvd = None


def eval_str(s):
    """If s is a string, return the eval results. Else return itself."""

    if isinstance(s, six.string_types):
        if len(s) > 0:
            return eval(s)
        return None
    return s


def build_command_line_parser(parser=None):
    '''Parse command line arguments.'''
    if parser is None:
        parser = argparse.ArgumentParser(prog='train',
                                         description='Train a classification model.')
    parser.add_argument(
        '-e',
        '--experiment_spec_file',
        type=str,
        required=True,
        help='Path to the experiment spec file.')
    parser.add_argument(
        '-r',
        '--results_dir',
        required=True,
        type=str,
        help='Path to a folder where experiment outputs should be written.'
    )
    parser.add_argument(
        '-k',
        '--key',
        required=False,
        default="",
        type=str,
        help='Key to save or load a .tlt model.'
    )
    parser.add_argument(
        '--init_epoch',
        default=1,
        type=int,
        help='Set resume epoch.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Include this flag in command line invocation for verbose logs.'
    )
    parser.add_argument(
        '-c',
        '--classmap',
        help="Class map file to set the class indices of the model.",
        type=str,
        default=None
    )
    return parser


def parse_command_line(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser(args)
    return parser.parse_args(args)


def setup_callbacks(model_name, results_dir, lr_config,
                    init_epoch, iters_per_epoch, max_epoch, key,
                    hvd, weight_histograms=False):
    """Setup callbacks: tensorboard, checkpointer, lrscheduler, csvlogger.

    Args:
        model_name (str): name of the model used.
        results_dir (str): Path to a folder where various training outputs will
                           be written.
        lr_config: config derived from the Proto config file
        init_epoch (int): The number of epoch to resume training.
        key: encryption key
        hvd: horovod instance
        weight_histograms (bool): Enabled weight histograms in the tensorboard callback.

    Returns:
        callbacks (list of keras.callbacks): list of callbacks.
    """
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                 hvd.callbacks.MetricAverageCallback()]
    max_iterations = iters_per_epoch * max_epoch
    lrscheduler = build_lr_scheduler(lr_config, hvd.size(), max_iterations)
    init_step = (init_epoch - 1) * iters_per_epoch
    lrscheduler.reset(init_step)
    callbacks.append(lrscheduler)

    if hvd.rank() == 0:
        # Set up the checkpointer.
        save_weights_dir = os.path.join(results_dir, 'weights')
        if not os.path.exists(save_weights_dir):
            os.makedirs(save_weights_dir)
        # Save encrypted models
        weight_filename = os.path.join(save_weights_dir,
                                       '%s_{epoch:03d}.hdf5' % model_name)
        checkpointer = AdvModelCheckpoint(weight_filename, key, verbose=1)
        callbacks.append(checkpointer)

        # Set up the custom TensorBoard callback. It will log the loss
        # after every step, and some images and user-set summaries only on
        # the first step of every epoch. Align this with other keras
        # networks.
        tensorboard_dir = os.path.join(results_dir, "events")
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        tensorboard = TensorBoard(
            log_dir=tensorboard_dir,
            weight_hist=weight_histograms
        )
        callbacks.append(tensorboard)

        # Set up the CSV logger, logging statistics after every epoch.
        csvfilename = os.path.join(results_dir, 'training.csv')
        csvlogger = CSVLogger(csvfilename,
                              separator=',',
                              append=False)
        callbacks.append(csvlogger)
        status_logger = TAOStatusLogger(
            results_dir,
            append=True,
            num_epochs=max_epoch,
            is_master=hvd.rank() == 0,
        )
        callbacks.append(status_logger)

    return callbacks


def verify_dataset_classes(dataset_path, classes):
    """Verify whether classes are in the dataset.

    Args:
        dataset_path (str): Path to the dataset.
        classes (list): List of classes.

    Returns:
        No explicit returns.
    """
    dataset_classlist = [
        item for item in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, item))
    ]
    missed_classes = []
    for class_name in classes:
        if class_name not in dataset_classlist:
            missed_classes.append(class_name)
    assert not missed_classes, (
        "Some classes mentioned in the classmap file were "
        f"missing in the dataset at {dataset_path}. "
        f"\n Missed classes are {missed_classes}"
    )


def categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    """Ported tf.keras categorical_crossentropy."""
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing == 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)
    return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)


def load_data(train_data, val_data, preprocessing_func,
              image_height, image_width, batch_size,
              enable_random_crop=False, enable_center_crop=False,
              enable_color_augmentation=False,
              interpolation=0, color_mode="rgb",
              mixup_alpha=0.0, no_horizontal_flip=False,
              classmap=None):
    """Load training and validation data with default data augmentation.

    Args:
        train_data (str): path to the training data.
        val_data (str): path to the validation data.
        preprocessing_func: function to process an image.
        image_height (int): Height of the input image tensor.
        image_width (int): Width of the input image tensor.
        batch_size (int): Number of image tensors per batch.
        enable_random_crop (bool): Flag to enable random cropping in load_img.
        enable_center_crop (bool): Flag to enable center cropping for val.
        enable_color_augmentation(bool): Flag to enable color augmentation.
        interpolation(int): Interpolation method for image resize. 0 means bilinear,
            while 1 means bicubic.
        color_mode (str): Input image read mode as either `rgb` or `grayscale`.
        mixup_alpha (float): mixup alpha.
        no_horizontal_flip(bool): Flag to disable horizontal flip for
            direction-aware datasets.
        classmap (str): Path to classmap file.

    Return:
        train/val Iterators and number of classes in the dataset.
    """
    interpolation_map = {
        0: "bilinear",
        1: "bicubic"
    }
    interpolation = interpolation_map[interpolation]
    classes = None
    if classmap is not None:
        classes = get_classes_from_classmap(classmap)
        verify_dataset_classes(train_data, classes)
        verify_dataset_classes(val_data, classes)
    # set color augmentation properly for train.
    # this global var will not affect validation dataset because
    # the crop method is either "none" or "center" for val dataset,
    # while this color augmentation is only possible for "random" crop.
    if enable_color_augmentation:
        preprocess_crop._set_color_augmentation(True)
    # Initializing data generator : Train
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_func,
        horizontal_flip=not no_horizontal_flip,
        featurewise_center=False)
    train_iterator = MixupImageDataGenerator(
        train_datagen, train_data, batch_size,
        image_height, image_width,
        color_mode=color_mode,
        interpolation=interpolation + ':random' if enable_random_crop else interpolation,
        alpha=mixup_alpha,
        classes=classes
    )
    logger.info('Processing dataset (train): {}'.format(train_data))

    # Initializing data generator: Val
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_func,
        horizontal_flip=False)

    # Initializing data iterator: Val
    val_iterator = val_datagen.flow_from_directory(
        val_data,
        target_size=(image_height, image_width),
        color_mode=color_mode,
        batch_size=batch_size,
        interpolation=interpolation + ':center' if enable_center_crop else interpolation,
        shuffle=False,
        classes=classes,
        class_mode='categorical')
    logger.info('Processing dataset (validation): {}'.format(val_data))

    # Check if the number of classes is > 1
    assert train_iterator.num_classes > 1, \
        "Number of classes should be greater than 1. Consider adding a background class."

    # Check if the number of classes is consistent
    assert train_iterator.num_classes == val_iterator.num_classes, \
        "Number of classes in train and val don't match."
    return train_iterator, val_iterator, train_iterator.num_classes


def get_classes_from_classmap(classmap_path):
    """Get list of classes from classmap file.

    Args:
        classmap_path (str): Path to the classmap file.

    Returns:
        classes (list): List of classes
    """
    if not os.path.exists(classmap_path):
        raise FileNotFoundError(
            f"Class map file wasn't found at {classmap_path}"
        )
    with open(classmap_path, "r") as cmap_file:
        try:
            data = json.load(cmap_file)
        except json.decoder.JSONDecodeError as e:
            print(f"Loading the {classmap_path} failed with error\n{e}")
            sys.exit(-1)
        except Exception as e:
            if e.output is not None:
                print(f"Classification exporter failed with error {e.output}")
            sys.exit(-1)
    if not data:
        return []
    classes = [""] * len(list(data.keys()))
    if not all([isinstance(value, int) for value in data.values()]):
        raise RuntimeError(
            "The values in the classmap file should be int."
            "Please verify the contents of the classmap file."
        )
    if not all([class_index < len(classes)
                and isinstance(class_index, int)
                for class_index in data.values()]):
        raise RuntimeError(
            "Invalid data in the json file. The class index must "
            "be < number of classes and an integer value.")
    for classname, class_index in data.items():
        classes[class_index] = classname
    return classes


def run_experiment(config_path=None, results_dir=None,
                   key=None, init_epoch=1, verbosity=False,
                   classmap=None):
    """Launch experiment that trains the model.

    NOTE: Do not change the argument names without verifying that cluster
          submission works.

    Args:
        config_path (str): Path to a text file containing a complete experiment
                           configuration.
        results_dir (str): Path to a folder where various training outputs will
                           be written.
        If the folder does not already exist, it will be created.
        init_epoch (int): The number of epoch to resume training.
        classmap (str): Path to the classmap file.
    """
    # Horovod: initialize Horovod.
    hvd = hvd_keras()
    hvd.init()
    # Load experiment spec.
    if config_path is not None:
        # Create an experiment_pb2.Experiment object from the input file.
        logger.info("Loading experiment spec at %s.", config_path)
        # The spec in config_path has to be complete.
        # Default spec is not merged into es.
        es = load_experiment_spec(config_path, merge_from_default=False)
    else:
        logger.info("Loading the default experiment spec.")
        es = load_experiment_spec()
    model_config = es.model_config
    train_config = es.train_config
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # check if model parallelism is enabled or not
    if train_config.model_parallelism:
        world_size = len(train_config.model_parallelism)
    else:
        world_size = 1
    gpus = list(range(hvd.local_rank() * world_size, (hvd.local_rank() + 1) * world_size))
    config.gpu_options.visible_device_list = ','.join([str(x) for x in gpus])
    K.set_session(tf.Session(config=config))
    verbose = 1 if hvd.rank() == 0 else 0
    K.set_image_data_format('channels_first')
    # Configure the logger.
    logging.basicConfig(
        format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
        level='DEBUG' if verbosity else 'INFO')

    # Set random seed.
    logger.debug("Random seed is set to {}".format(train_config.random_seed))
    set_random_seed(train_config.random_seed + hvd.local_rank())
    is_master = hvd.rank() == 0
    if is_master and not os.path.exists(results_dir):
        os.makedirs(results_dir)
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            is_master=is_master,
            verbosity=logger.getEffectiveLevel(),
            append=True
        )
    )
    # Configure tf logger verbosity.
    tf.logging.set_verbosity(tf.logging.INFO)

    weight_histograms = False
    if hvd.rank() == 0:
        if train_config.HasField("visualizer"):
            weight_histograms = train_config.visualizer.weight_histograms
            if train_config.visualizer.HasField("clearml_config"):
                clearml_config = train_config.visualizer.clearml_config
                _ = get_clearml_task(clearml_config, "classification")

    # get channel, height and width of the input image
    nchannels, image_height, image_width = map(int, model_config.input_image_size.split(','))
    assert nchannels in [1, 3], "Invalid input image dimension."
    assert image_height >= 16, "Image height should be greater than 15 pixels."
    assert image_width >= 16, "Image width should be greater than 15 pixels."

    if nchannels == 3:
        color_mode = "rgb"
    else:
        color_mode = "grayscale"

    img_mean = train_config.image_mean
    if train_config.preprocess_mode in ['tf', 'torch'] and img_mean:
        logger.info("Custom image mean is only supported in `caffe` mode.")
        logger.info("Custom image mean will be ignored.")
    if train_config.preprocess_mode == 'caffe':
        mode_txt = 'Custom'
        if nchannels == 3:
            if img_mean:
                assert all(c in img_mean for c in ['r', 'g', 'b']) , (
                    "'r', 'g', 'b' should all be present in image_mean "
                    "for images with 3 channels."
                )
                img_mean = [img_mean['b'], img_mean['g'], img_mean['r']]
            else:
                mode_txt = 'Default'
                img_mean = [103.939, 116.779, 123.68]
        else:
            if img_mean:
                assert 'l' in img_mean, (
                    "'l' should be present in image_mean for images "
                    "with 1 channel."
                )
                img_mean = [img_mean['l']]
            else:
                mode_txt = 'Default'
                img_mean = [117.3786]
        logger.info("{} image mean {} will be used.".format(mode_txt, img_mean))

    # Load augmented data
    train_iterator, val_iterator, nclasses = \
        load_data(train_config.train_dataset_path,
                  train_config.val_dataset_path,
                  partial(preprocess_input,
                          data_format='channels_first',
                          mode=train_config.preprocess_mode,
                          img_mean=img_mean,
                          color_mode=color_mode),
                  image_height, image_width,
                  train_config.batch_size_per_gpu,
                  train_config.enable_random_crop,
                  train_config.enable_center_crop,
                  train_config.enable_color_augmentation,
                  model_config.resize_interpolation_method,
                  color_mode=color_mode,
                  mixup_alpha=train_config.mixup_alpha,
                  no_horizontal_flip=train_config.disable_horizontal_flip,
                  classmap=classmap)

    # Creating model
    ka = dict()
    ka['nlayers'] = model_config.n_layers if model_config.n_layers else 18
    ka['use_batch_norm'] = model_config.use_batch_norm
    ka['use_pooling'] = model_config.use_pooling
    ka['freeze_bn'] = model_config.freeze_bn
    ka['use_bias'] = model_config.use_bias
    ka['all_projections'] = model_config.all_projections
    ka['dropout'] = model_config.dropout if model_config.dropout else 0.0
    ka['activation'] = model_config.activation
    freeze_blocks = model_config.freeze_blocks if model_config.freeze_blocks else None
    ka['passphrase'] = key

    final_model = get_model(arch=model_config.arch if model_config.arch else "resnet",
                            input_shape=(nchannels, image_height, image_width),
                            data_format='channels_first',
                            nclasses=nclasses,
                            retain_head=model_config.retain_head,
                            freeze_blocks=freeze_blocks,
                            **ka)

    # Set up BN and regularizer config
    if model_config.HasField("batch_norm_config"):
        bn_config = model_config.batch_norm_config
    else:
        bn_config = None
    final_model = setup_config(
        final_model,
        train_config.reg_config,
        freeze_bn=model_config.freeze_bn,
        bn_config=bn_config,
        custom_objs={}
    )

    if train_config.pretrained_model_path:
        # Decrypt and load pretrained model
        pretrained_model = model_io(train_config.pretrained_model_path, enc_key=key)

        strict_mode = True
        for layer in pretrained_model.layers[1:]:
            # The layer must match up to ssd layers.
            if layer.name == 'predictions':
                strict_mode = False
            try:
                l_return = final_model.get_layer(layer.name)
            except ValueError:
                # Some layers are not there
                continue
            try:
                l_return.set_weights(layer.get_weights())
            except ValueError:
                if strict_mode:
                    # This is a pruned model
                    final_model = setup_config(
                        pretrained_model,
                        train_config.reg_config,
                        bn_config=bn_config
                    )

    # model parallelism, keep the freeze_bn config untouched when building
    # a new parallilized model
    if train_config.model_parallelism:
        final_model = model_parallelism(
            final_model,
            tuple(train_config.model_parallelism),
            model_config.freeze_bn
            )
    # Printing model summary
    final_model.summary()

    if init_epoch > 1 and not train_config.pretrained_model_path:
        raise ValueError("Make sure to load the correct model when setting initial epoch > 1.")

    if train_config.pretrained_model_path and init_epoch > 1:
        opt = pretrained_model.optimizer
    else:
        # Defining optimizer
        opt = build_optimizer(train_config.optimizer)

    # Add Horovod Distributed Optimizer
    opt = hvd.DistributedOptimizer(opt)
    # Compiling model
    cc = partial(categorical_crossentropy, label_smoothing=train_config.label_smoothing)
    cc.__name__ = "categorical_crossentropy"
    final_model.compile(loss=cc, metrics=['accuracy'],
                        optimizer=opt)

    callbacks = setup_callbacks(model_config.arch, results_dir,
                                train_config.lr_config,
                                init_epoch, len(train_iterator) // hvd.size(),
                                train_config.n_epochs, key,
                                hvd, weight_histograms=weight_histograms)
    # Writing out class-map file for inference mapping
    if hvd.rank() == 0:
        with open(os.path.join(results_dir, "classmap.json"), "w") \
             as classdump:
            json.dump(train_iterator.class_indices, classdump)

    # Commencing Training
    final_model.fit_generator(
        train_iterator,
        steps_per_epoch=len(train_iterator) // hvd.size(),
        epochs=train_config.n_epochs,
        verbose=verbose,
        workers=train_config.n_workers,
        validation_data=val_iterator,
        validation_steps=len(val_iterator),
        callbacks=callbacks,
        initial_epoch=init_epoch - 1)

    # Evaluate the model on the full data set.
    status_logging.get_status_logger().write(message="Final model evaluation in progress.")
    score = hvd.allreduce(
                final_model.evaluate_generator(val_iterator,
                                               len(val_iterator),
                                               workers=train_config.n_workers))
    kpi_data = {
        "validation_loss": float(score[0]),
        "validation_accuracy": float(score[1])
    }

    status_logging.get_status_logger().kpi = kpi_data

    status_logging.get_status_logger().write(message="Model evaluation is complete.")
    if verbose:
        logger.info('Total Val Loss: {}'.format(score[0]))
        logger.info('Total Val accuracy: {}'.format(score[1]))
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.SUCCESS,
        message="Training finished successfully."
    )


@check_tf_oom
def main(args=None):
    """Wrapper function for continuous training of MakeNet application.

    Args:
       Dictionary arguments containing parameters defined by command line
       parameters.

    """
    # parse command line
    args = parse_command_line(args)
    try:
        run_experiment(config_path=args.experiment_spec_file,
                       results_dir=args.results_dir,
                       key=args.key,
                       init_epoch=args.init_epoch,
                       verbosity=args.verbose,
                       classmap=args.classmap)
        logger.info("Training finished successfully.")
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Training was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
        logger.info("Training was interrupted.")
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e


if __name__ == '__main__':
    main()
