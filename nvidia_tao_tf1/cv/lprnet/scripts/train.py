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

"""Perform continuous LPRNet training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import lru_cache
import logging
from math import ceil
import os

import tensorflow as tf

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.mlops.clearml import get_clearml_task

from nvidia_tao_tf1.cv.lprnet.callbacks.ac_callback import LPRAccuracyCallback as ac_callback
from nvidia_tao_tf1.cv.lprnet.callbacks.enc_model_saver import KerasModelSaver
from nvidia_tao_tf1.cv.lprnet.callbacks.loggers import TAOStatusLogger
from nvidia_tao_tf1.cv.lprnet.callbacks.soft_start_annealing import \
    SoftStartAnnealingLearningRateScheduler as LRS
from nvidia_tao_tf1.cv.lprnet.callbacks.tb_callback import LPRNetTensorBoardImage
from nvidia_tao_tf1.cv.lprnet.dataloader.data_sequence import LPRNetDataGenerator
from nvidia_tao_tf1.cv.lprnet.loss.wrap_ctc_loss import WrapCTCLoss
from nvidia_tao_tf1.cv.lprnet.models import model_builder
from nvidia_tao_tf1.cv.lprnet.utils.model_io import load_model_as_pretrain
from nvidia_tao_tf1.cv.lprnet.utils.spec_loader import (
    load_experiment_spec,
    spec_validator,
    TRAIN_EXP_REQUIRED_MSG
)

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')
logger = logging.getLogger(__name__)
verbose = 0


@lru_cache()
def hvd_tf_keras():
    """lazy import horovod."""

    import horovod.tensorflow.keras as hvd

    return hvd


def run_experiment(config_path, results_dir, resume_weights, key, init_epoch=1):
    """
    Launch experiment that trains the model.

    NOTE: Do not change the argument names without verifying that cluster submission works.

    Args:
        config_path (str): Path to a text file containing a complete experiment configuration.
        results_dir (str): Path to a folder where various training outputs will be written.
        If the folder does not already exist, it will be created.
        resume_weights (str): Optional path to a pretrained model file.
        init_epoch (int): The number of epoch to resume training.
    """
    hvd = hvd_tf_keras()
    hvd.init()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    verbose = 1 if hvd.rank() == 0 else 0
    is_master = hvd.rank() == 0
    if is_master and not os.path.exists(results_dir):
        os.makedirs(results_dir)
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            is_master=is_master,
            verbosity=1,
            append=True
        )
    )
    # Load experiment spec.
    experiment_spec = load_experiment_spec(config_path)

    spec_validator(experiment_spec, TRAIN_EXP_REQUIRED_MSG)

    training_config = experiment_spec.training_config
    if hvd.rank() == 0:
        if training_config.HasField("visualizer"):
            if training_config.visualizer.HasField("clearml_config"):
                clearml_config = training_config.visualizer.clearml_config
                get_clearml_task(clearml_config, "lprnet")
    # Load training parameters
    num_epochs = experiment_spec.training_config.num_epochs
    batch_size_per_gpu = experiment_spec.training_config.batch_size_per_gpu
    ckpt_interval = experiment_spec.training_config.checkpoint_interval or 5
    # config kernel regularizer
    reg_type = experiment_spec.training_config.regularizer.type
    reg_weight = experiment_spec.training_config.regularizer.weight
    kr = None
    br = None
    if reg_type:
        if reg_type > 0:
            assert 0 < reg_weight < 1, \
                "Weight decay should be no less than 0 and less than 1"
            if reg_type == 1:
                kr = tf.keras.regularizers.l1(reg_weight)
                br = tf.keras.regularizers.l1(reg_weight)
            else:
                kr = tf.keras.regularizers.l2(reg_weight)
                br = tf.keras.regularizers.l2(reg_weight)

    # configure optimizer and loss
    optimizer = tf.keras.optimizers.SGD(lr=0.0001,
                                        momentum=0.9,
                                        decay=0.0,
                                        nesterov=False)

    max_label_length = experiment_spec.lpr_config.max_label_length
    ctc_loss = WrapCTCLoss(max_label_length)

    # build train/eval model
    if resume_weights is not None:
        if init_epoch == 1:
            resume_from_training = False
        else:
            resume_from_training = True

        logger.info("Loading pretrained weights. This may take a while...")
        model, model_eval, time_step = \
            load_model_as_pretrain(resume_weights,
                                   experiment_spec,
                                   key, kr, br,
                                   resume_from_training)

        if init_epoch == 1:
            print("Initialize optimizer")
            model.compile(optimizer=hvd.DistributedOptimizer(optimizer),
                          loss=ctc_loss.compute_loss)
        else:
            print("Resume optimizer from pretrained model")
            model.compile(optimizer=hvd.DistributedOptimizer(model.optimizer),
                          loss=ctc_loss.compute_loss)
    else:
        model, model_eval, time_step = \
            model_builder.build(experiment_spec,
                                kernel_regularizer=kr,
                                bias_regularizer=br)

        print("Initialize optimizer")
        model.compile(optimizer=hvd.DistributedOptimizer(optimizer),
                      loss=ctc_loss.compute_loss)

    # build train / eval dataset:
    train_data = LPRNetDataGenerator(experiment_spec=experiment_spec,
                                     is_training=True,
                                     shuffle=True,
                                     time_step=time_step)

    val_data = LPRNetDataGenerator(experiment_spec=experiment_spec,
                                   is_training=False,
                                   shuffle=False)

    # build learning rate scheduler
    lrconfig = experiment_spec.training_config.learning_rate.soft_start_annealing_schedule
    total_num = train_data.n_samples
    iters_per_epoch = int(ceil(total_num / batch_size_per_gpu) // hvd.size())
    max_iterations = num_epochs * iters_per_epoch
    lr_scheduler = LRS(base_lr=lrconfig.max_learning_rate * hvd.size(),
                       min_lr_ratio=lrconfig.min_learning_rate / lrconfig.max_learning_rate,
                       soft_start=lrconfig.soft_start,
                       annealing_start=lrconfig.annealing,
                       max_iterations=max_iterations)
    init_step = (init_epoch - 1) * iters_per_epoch
    lr_scheduler.reset(init_step)

    terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()

    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                 hvd.callbacks.MetricAverageCallback(),
                 lr_scheduler,
                 terminate_on_nan]

    # build logger and checkpoint saver for master GPU:
    if hvd.rank() == 0:
        model.summary()
        logger.info("Number of images in the training dataset:\t{:>6}"
                    .format(train_data.n_samples))
        logger.info("Number of images in the validation dataset:\t{:>6}"
                    .format(val_data.n_samples))
        if not os.path.exists(os.path.join(results_dir, 'weights')):
            os.mkdir(os.path.join(results_dir, 'weights'))

        ckpt_path = os.path.join(results_dir, 'weights',
                                 "lprnet_epoch-{epoch:03d}.hdf5")
        model_checkpoint = KerasModelSaver(ckpt_path, key, ckpt_interval, last_epoch=num_epochs,
                                         verbose=verbose)
        callbacks.append(model_checkpoint)

        status_logger = TAOStatusLogger(
            results_dir,
            append=True,
            num_epochs=num_epochs,
            is_master=hvd.rank() == 0,
        )
        callbacks.append(status_logger)

        if val_data.n_samples > 0:
            eval_interval = experiment_spec.eval_config.validation_period_during_training
            tf.keras.backend.set_learning_phase(0)
            ac_checkpoint = ac_callback(eval_model=model_eval,
                                        eval_interval=eval_interval,
                                        val_dataset=val_data,
                                        verbose=verbose)
            callbacks.append(ac_checkpoint)
            tf.keras.backend.set_learning_phase(1)

        csv_logger = tf.keras.callbacks.CSVLogger(filename=os.path.join(results_dir,
                                                                        "lprnet_training_log.csv"),
                                                  separator=",",
                                                  append=False)

        callbacks.append(csv_logger)

    # init early stopping:
    if experiment_spec.training_config.HasField("early_stopping"):
        es_config = experiment_spec.training_config.early_stopping
        es_cb = tf.keras.callbacks.EarlyStopping(monitor=es_config.monitor,
                                                 min_delta=es_config.min_delta,
                                                 patience=es_config.patience,
                                                 verbose=True)
        callbacks.append(es_cb)

    if hvd.rank() == 0:
        if experiment_spec.training_config.visualizer.enabled:
            tb_log_dir = os.path.join(results_dir, "events")
            tb_cb = tf.keras.callbacks.TensorBoard(tb_log_dir, write_graph=False)
            callbacks.append(tb_cb)
            tbimg_cb = LPRNetTensorBoardImage(tb_log_dir,
                                              experiment_spec.training_config.visualizer.num_images)
            fetches = [tf.assign(tbimg_cb.img, model.inputs[0], validate_shape=False)]
            model._function_kwargs = {'fetches': fetches}
            callbacks.append(tbimg_cb)

    train_steps = int(ceil(train_data.n_samples / batch_size_per_gpu) // hvd.size())
    model.fit(x=train_data,
              steps_per_epoch=train_steps,
              epochs=num_epochs,
              callbacks=callbacks,
              workers=4,
              use_multiprocessing=False,
              initial_epoch=init_epoch - 1,
              verbose=verbose
              )

    status_logging.get_status_logger().write(
        status_level=status_logging.Status.SUCCESS,
        message="Training finished successfully."
    )


def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
            parser mechanism.

    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            description='Train a LPRNet'
        )

    parser.add_argument(
        '-e',
        '--experiment_spec_file',
        type=str,
        required=True,
        help='Path to spec file. Absolute path or relative to working directory. \
            If not specified, default spec from spec_loader.py is used.')
    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        required=True,
        help='Path to a folder where experiment outputs should be written.'
    )
    parser.add_argument(
        '-k',
        '--key',
        default="",
        type=str,
        required=False,
        help='Key to save or load a .tlt model.'
    )
    parser.add_argument(
        '-m',
        '--resume_model_weights',
        type=str,
        default=None,
        help='Path to a model to continue training.'
    )
    parser.add_argument(
        '--initial_epoch',
        type=int,
        default=1,
        help='Set resume epoch'
    )

    return parser


def parse_command_line_arguments(args=None):
    """
    Parse command-line flags passed to the training script.

    Returns:
      Namespace with all parsed arguments.
    """
    parser = build_command_line_parser()
    return parser.parse_args(args)


def main(args=None):
    """Run the training process."""
    args = parse_command_line_arguments(args)
    try:
        run_experiment(config_path=args.experiment_spec_file,
                       results_dir=args.results_dir,
                       resume_weights=args.resume_model_weights,
                       init_epoch=args.initial_epoch,
                       key=args.key)
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


if __name__ == "__main__":
    main()
