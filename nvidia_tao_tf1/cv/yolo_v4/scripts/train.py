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

"""Perform continuous YOLO training on a tfrecords or keras dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import warnings
from keras import backend as K
import tensorflow as tf

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utils import check_tf_oom, hvd_keras, initialize
from nvidia_tao_tf1.cv.yolo_v3.utils.tensor_utils import get_init_ops
from nvidia_tao_tf1.cv.yolo_v4.models.utils import build_training_pipeline
from nvidia_tao_tf1.cv.yolo_v4.utils.spec_loader import load_experiment_spec


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')
logger = logging.getLogger(__name__)
verbose = 0
warnings.filterwarnings(action="ignore", category=UserWarning)


def run_experiment(config_path, results_dir, key):
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
    K.set_image_data_format('channels_first')
    K.set_learning_phase(1)
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
    status_logging.get_status_logger().write(
        data=None,
        status_level=status_logging.Status.STARTED,
        message="Starting Yolo_V4 Training job"
    )
    # Load experiment spec.
    spec = load_experiment_spec(config_path)
    initialize(spec.random_seed, hvd)
    # build training model and dataset
    model = build_training_pipeline(
        spec,
        results_dir,
        key,
        hvd,
        sess,
        verbose
    )
    if hvd.rank() == 0:
        model.summary()
    sess.run(get_init_ops())
    model.train(verbose)
    status_logging.get_status_logger().write(
        data=None,
        status_level=status_logging.Status.SUCCESS,
        message="YOLO_V4 training finished successfully."
    )


def build_command_line_parser(parser=None):
    '''build parser.'''
    if parser is None:
        parser = argparse.ArgumentParser(prog='train', description='Train an YOLOv4 model.')
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
        type=str,
        default="",
        required=False,
        help='Key to save or load a .tlt model.'
    )
    return parser


def parse_command_line(args):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser()
    return parser.parse_args(args)


@check_tf_oom
def main(args=None):
    """Run the training process."""
    args = parse_command_line(args)
    try:
        run_experiment(
            config_path=args.experiment_spec_file,
            results_dir=args.results_dir,
            key=args.key
        )
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


if __name__ == "__main__":
    main()
