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

"""Perform continuous MultitaskNet training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import json
import logging

from multiprocessing import cpu_count
import os

import keras
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import TerminateOnNaN

import tensorflow as tf
from nvidia_tao_tf1.cv.common.callbacks.enc_model_saver_callback import KerasModelSaver
from nvidia_tao_tf1.cv.common.callbacks.loggers import TAOStatusLogger
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.mlops.clearml import get_clearml_task
from nvidia_tao_tf1.cv.common.utils import build_lrs_from_config
from nvidia_tao_tf1.cv.common.utils import (
    build_optimizer_from_config,
    build_regularizer_from_config
)
from nvidia_tao_tf1.cv.common.utils import (
    check_tf_oom,
    hvd_keras,
    initialize,
    parse_model_load_from_config
)
from nvidia_tao_tf1.cv.common.utils import OneIndexedCSVLogger as CSVLogger
from nvidia_tao_tf1.cv.common.utils import TensorBoard
from nvidia_tao_tf1.cv.multitask_classification.data_loader.data_generator import (
    MultiClassDataGenerator
)
from nvidia_tao_tf1.cv.multitask_classification.model.model_builder import get_model
from nvidia_tao_tf1.cv.multitask_classification.utils.model_io import load_model
from nvidia_tao_tf1.cv.multitask_classification.utils.spec_loader import load_experiment_spec


logger = logging.getLogger(__name__)
verbose = 0


def build_command_line_parser(parser=None):
    """Build a command line parser for inference."""
    if parser is None:
        parser = argparse.ArgumentParser(
            description="TAO Toolkit Multitask Classification training."
        )
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
        "-v",
        "--verbose",
        action='store_true',
        default=False,
        help="Flag to enable verbose logging."
    )
    return parser


def parse_command_line_arguments(args=None):
    """Parse command line arguments for training."""
    parser = build_command_line_parser()
    return parser.parse_args(args)


def _load_pretrain_weights(pretrain_model, train_model):
    """Load weights in pretrain model to model."""
    strict_mode = True
    for layer in train_model.layers[1:]:
        # The layer must match up to yolo layers.
        if layer.name.find('multitask_') != -1:
            strict_mode = False
        try:
            l_return = pretrain_model.get_layer(layer.name)
        except ValueError:
            if strict_mode and layer.name[-3:] != 'qdq' and len(layer.get_weights()) != 0:
                raise ValueError(layer.name + ' not found in pretrained model.')
            # Ignore QDQ
            continue
        try:
            layer.set_weights(l_return.get_weights())
        except ValueError:
            if strict_mode:
                raise ValueError(layer.name + ' has incorrect shape in pretrained model.')
            continue


def construct_model(model_config, training_config, nclasses_dict, key):
    '''
    Construct a model according to spec file.

    Args:
        model_config: model_config of parsed spec file
        training_config: training_config of parsed spec file
        nclasses_dict: dictionary with task / class information from data loader
        key: TLT encryption / decryption key
    Returns:
        model: built model
        init_epoch: training should start from this epoch
    '''

    # load_path, load_graph, reset_optim, init_epoch = load_config
    load_config = parse_model_load_from_config(training_config)
    load_model_path = load_config[0]
    load_graph = load_config[1]
    reset_optim = load_config[2]
    nchannels, im_height, im_width = map(int, model_config.input_image_size.split(','))

    # Creating model
    ka = dict()
    ka['nlayers'] = model_config.n_layers if model_config.n_layers else 18
    ka['use_batch_norm'] = model_config.use_batch_norm
    ka['use_pooling'] = model_config.use_pooling
    ka['freeze_bn'] = model_config.freeze_bn
    ka['use_bias'] = model_config.use_bias
    ka['all_projections'] = model_config.all_projections
    ka['dropout'] = model_config.dropout if model_config.dropout else 0.0
    ka['freeze_blocks'] = model_config.freeze_blocks if model_config.freeze_blocks else None
    ka['arch'] = model_config.arch if model_config.arch else "resnet"
    ka['data_format'] = 'channels_first'
    ka['nclasses_dict'] = nclasses_dict
    ka['input_shape'] = (nchannels, im_height, im_width)
    ka['kernel_regularizer'] = build_regularizer_from_config(training_config.regularizer)

    if (not load_model_path) or (not load_graph):
        # needs to build a training model
        train_model = get_model(**ka)

        if load_model_path:
            # load pretrain weights
            pretrain_model = load_model(load_model_path, key=key)
            _load_pretrain_weights(pretrain_model, train_model)
    else:
        train_model = load_model(load_model_path, key=key)

        if reset_optim:
            train_model_config = train_model.get_config()
            for layer, layer_config in zip(train_model.layers, train_model_config['layers']):
                if hasattr(layer, 'kernel_regularizer'):
                    layer_config['config']['kernel_regularizer'] = ka['kernel_regularizer']
            reg_model = keras.Model.from_config(train_model_config)
            reg_model.set_weights(train_model.get_weights())
            train_model = reg_model

    if (not load_model_path) or reset_optim:
        optim = build_optimizer_from_config(training_config.optimizer)

        train_model.compile(loss=len(nclasses_dict)*["categorical_crossentropy"],
                            loss_weights=len(nclasses_dict)*[1.0],
                            metrics=["accuracy"], optimizer=optim)
    return train_model, load_config[3]


def run_experiment(config_path, results_dir, key, verbose=False):
    """
    Launch experiment that trains the model.

    NOTE: Do not change the argument names without verifying that cluster submission works.

    Args:
        config_path (str): Path to a text file containing a complete experiment configuration.
        results_dir (str): Path to a folder where various training outputs will be written.
        If the folder does not already exist, it will be created.
    """
    hvd = hvd_keras()
    hvd.init()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.Session(config=config)
    K.set_session(sess)
    verbose = 1 if hvd.rank() == 0 else 0
    # Load experiment spec.
    experiment_spec = load_experiment_spec(config_path)

    initialize(experiment_spec.random_seed, hvd)

    # Setting up keras backend and keras environment
    K.set_image_data_format("channels_first")

    logging.basicConfig(
        format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
        level='DEBUG' if verbose else 'INFO'
    )
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
    # keras.backend.set_learning_phase(1)
    # get channel, height and width of the input image
    model_config = experiment_spec.model_config
    training_config = experiment_spec.training_config
    if is_master:
        if training_config.HasField("visualizer"):
            if training_config.visualizer.HasField("clearml_config"):
                logger.info("Integrating with clearml.")
                clearml_config = training_config.visualizer.clearml_config
                get_clearml_task(
                    clearml_config,
                    "multitask_classification"
                )

    nchannels, im_height, im_width = map(int, model_config.input_image_size.split(','))

    if nchannels == 1:
        color_mode = 'grayscale'
    elif nchannels == 3:
        color_mode = 'rgb'
    else:
        raise ValueError("number of channels must be 1 or 3")

    # Initializing data generator : Train
    train_datagen = MultiClassDataGenerator(preprocessing_function=preprocess_input,
                                            horizontal_flip=True,
                                            featurewise_center=False,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            zoom_range=0.2)

    # Initiliazing data iterator: Train
    data_config = experiment_spec.dataset_config
    batch_size = training_config.batch_size_per_gpu
    train_iterator = train_datagen.flow_from_singledirectory(data_config.image_directory_path,
                                                             data_config.train_csv_path,
                                                             target_size=(im_height, im_width),
                                                             batch_size=batch_size,
                                                             color_mode=color_mode)
    if hvd.rank() == 0:
        print('Processing dataset (train): {}'.format(data_config.train_csv_path))

    # Initializing data generator: Val
    val_datagen = MultiClassDataGenerator(preprocessing_function=preprocess_input,
                                          horizontal_flip=False)

    # Initializing data iterator: Val
    val_iterator = val_datagen.flow_from_singledirectory(data_config.image_directory_path,
                                                         data_config.val_csv_path,
                                                         target_size=(im_height, im_width),
                                                         batch_size=batch_size,
                                                         color_mode=color_mode)
    if hvd.rank() == 0:
        print('Processing dataset (validation): {}'.format(data_config.val_csv_path))

    # Check if the number of classes is consistent
    assert train_iterator.class_dict == val_iterator.class_dict, \
        "Num of classes at train and val don't match"
    nclasses_dict = train_iterator.class_dict

    final_model, init_epoch = construct_model(model_config, training_config, nclasses_dict, key)

    final_model.optimizer = hvd.DistributedOptimizer(final_model.optimizer)

    # Load training parameters
    num_epochs = training_config.num_epochs
    ckpt_interval = training_config.checkpoint_interval

    # Setup callbacks
    iters_per_epoch = len(train_iterator) // hvd.size()

    max_iterations = num_epochs * iters_per_epoch
    lr_scheduler = build_lrs_from_config(training_config.learning_rate, max_iterations, hvd.size())

    init_step = init_epoch * iters_per_epoch
    lr_scheduler.reset(init_step)
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                 hvd.callbacks.MetricAverageCallback(),
                 lr_scheduler,
                 TerminateOnNaN()]

    # Writing out class-map file for inference mapping
    if hvd.rank() == 0:
        # Printing model summary
        final_model.summary()
        reverse_mapping = {task: {v: k for k, v in classes.items()}
                           for task, classes in train_iterator.class_mapping.items()}
        save_dict = {'tasks': train_iterator.tasks_header,
                     'class_mapping': reverse_mapping}
        json.dump(save_dict, open(os.path.join(results_dir, 'class_mapping.json'), 'w'))

        if not os.path.exists(os.path.join(results_dir, 'weights')):
            os.mkdir(os.path.join(results_dir, 'weights'))

        arch_name = model_config.arch
        if model_config.arch in ['resnet', 'darknet', 'cspdarknet', 'vgg']:
            # append nlayers into meta_arch_name
            arch_name += str(model_config.n_layers)

        ckpt_path = str(os.path.join(results_dir, 'weights',
                                     'multitask_cls_' + arch_name + '_epoch_{epoch:03d}.hdf5'))

        # This callback will update model_eval and save the model.
        model_checkpoint = KerasModelSaver(ckpt_path, key, ckpt_interval, verbose=verbose)

        csv_path = os.path.join(results_dir, 'multitask_cls_training_log_' + arch_name + '.csv')
        csv_logger = CSVLogger(filename=csv_path,
                               separator=',',
                               append=False)

        callbacks.append(model_checkpoint)
        callbacks.append(csv_logger)

        # Setting up TAO status logger.
        status_logger = TAOStatusLogger(
            results_dir,
            append=True,
            num_epochs=num_epochs,
            is_master=hvd.rank() == 0,
        )
        callbacks.append(status_logger)

        # Setting up Tensorboard visualizer.
        tensorboard_dir = os.path.join(results_dir, "events")
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        weight_histograms = False
        if training_config.HasField("visualizer"):
            weight_histograms = training_config.visualizer.weight_histograms
        tensorboard = TensorBoard(
            log_dir=tensorboard_dir,
            weight_hist=weight_histograms
        )
        callbacks.append(tensorboard)

    # Commencing Training
    final_model.fit_generator(train_iterator,
                              steps_per_epoch=iters_per_epoch,
                              epochs=num_epochs,
                              verbose=verbose,
                              workers=max(int((cpu_count() - 1) / hvd.size() + 0.5), 1),
                              validation_data=val_iterator,
                              validation_steps=len(val_iterator),
                              callbacks=callbacks,
                              max_queue_size=20,
                              initial_epoch=init_epoch)

    status_logging.get_status_logger().write(message="Final model evaluation in progress.")

    score = hvd.allreduce(
                final_model.evaluate_generator(val_iterator,
                                               len(val_iterator),
                                               workers=training_config.n_workers))
    status_logging.get_status_logger().write(message="Model evaluation in complete.")
    if verbose:
        print('Total Val Loss: {}'.format(score[0]))
        print('Tasks: {}'.format(val_iterator.tasks_header))
        print('Val loss per task: {}'.format(score[1:1 + val_iterator.num_tasks]))
        print('Val acc per task: {}'.format(score[1 + val_iterator.num_tasks:]))

    tasks = val_iterator.tasks_header
    val_accuracies = score[1 + val_iterator.num_tasks:]
    kpi_dict = {key: float(value) for key, value in zip(tasks, val_accuracies)}
    kpi_dict["mean accuracy"] = sum(val_accuracies) / len(val_accuracies)

    status_logging.get_status_logger().kpi.update(kpi_dict)

    status_logging.get_status_logger().write(
        status_level=status_logging.Status.SUCCESS,
        message="Multi-Task classification finished successfully"
    )


@check_tf_oom
def main(args=None):
    """Run the training process."""
    try:
        args = parse_command_line_arguments(args)
        run_experiment(config_path=args.experiment_spec_file,
                       results_dir=args.results_dir,
                       key=args.key,
                       verbose=args.verbose)
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Training was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e


if __name__ == "__main__":
    main()
