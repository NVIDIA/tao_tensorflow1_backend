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
"""FasterRCNN train script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys

from keras import backend as K
import tensorflow as tf

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utils import hvd_keras
from nvidia_tao_tf1.cv.faster_rcnn.models.utils import build_or_resume_model
from nvidia_tao_tf1.cv.faster_rcnn.spec_loader import spec_loader, spec_wrapper
from nvidia_tao_tf1.cv.faster_rcnn.utils import utils


def build_command_line_parser(parser=None):
    """Build a command line parser for training."""
    if parser is None:
        parser = argparse.ArgumentParser(description='Train or retrain a Faster-RCNN model.')
    parser.add_argument("-e",
                        "--experiment_spec",
                        type=str,
                        required=True,
                        help="Experiment spec file has all the training params.")
    parser.add_argument("-k",
                        "--enc_key",
                        type=str,
                        required=False,
                        help="TLT encoding key, can override the one in the spec file.")
    parser.add_argument("-r",
                        "--results_dir",
                        type=str,
                        default=None,
                        required=True,
                        help="Path to the files where the logs are stored.")
    return parser


def parse_args(args_in=None):
    """Parser arguments."""
    parser = build_command_line_parser()
    return parser.parse_known_args(args_in)[0]


def main(args=None):
    """Train or retrain a model."""
    options = parse_args(args)
    spec = spec_loader.load_experiment_spec(options.experiment_spec)
    # enc key in CLI will override the one in the spec file.
    if options.enc_key is not None:
        spec.enc_key = options.enc_key
    spec = spec_wrapper.ExperimentSpec(spec)
    hvd = hvd_keras()
    hvd.init()
    results_dir = options.results_dir
    is_master = hvd.rank() == 0
    if is_master and not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            is_master=is_master,
            verbosity=1,
            append=True
        )
    )
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # check if model parallelism is enabled or not
    if spec.training_config.model_parallelism:
        world_size = len(spec.training_config.model_parallelism)
    else:
        world_size = 1
    gpus = list(range(hvd.local_rank() * world_size, (hvd.local_rank() + 1) * world_size))
    config.gpu_options.visible_device_list = ','.join([str(x) for x in gpus])
    K.set_session(tf.Session(config=config))
    K.set_image_data_format('channels_first')
    K.set_learning_phase(1)
    utils.set_random_seed(spec.random_seed+hvd.rank())
    verbosity = 'INFO'
    if spec.verbose:
        verbosity = 'DEBUG'
    logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                        level=verbosity)
    logger = logging.getLogger(__name__)
    # Returning the clearml task incase it's needed to be closed.
    model, iters_per_epoch, initial_epoch, _ = build_or_resume_model(spec, hvd, logger, results_dir)
    if hvd.rank() == 0:
        model.summary()
    K.get_session().run(utils.get_init_ops())
    model.train(spec.epochs, iters_per_epoch, initial_epoch)
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.SUCCESS,
        message="Training finished successfully."
    )


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    try:
        main()
        logger.info("Training finished successfully.")
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Training was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
        logger.info("Training was interrupted.")
    except tf.errors.ResourceExhaustedError:
        status_logging.get_status_logger().write(
            message=(
                "Ran out of GPU memory, please lower the batch size, use a smaller input "
                "resolution, use a smaller backbone or try model parallelism. See "
                "documentation on how to enable model parallelism for FasterRCNN."
            ),
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
        logger.error(
            "Ran out of GPU memory, please lower the batch size, use a smaller input "
            "resolution, use a smaller backbone or try model parallelism. See "
            "documentation on how to enable model parallelism for FasterRCNN."
        )
        sys.exit(1)
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
