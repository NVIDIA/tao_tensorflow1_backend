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

"""Perform EfficientDet training on a tfrecords dataset."""

import argparse
import logging
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.util import deprecation

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
import nvidia_tao_tf1.cv.common.no_warning # noqa pylint: disable=W0611
from nvidia_tao_tf1.cv.efficientdet.dataloader import dataloader
from nvidia_tao_tf1.cv.efficientdet.executer import distributed_executer
from nvidia_tao_tf1.cv.efficientdet.models import det_model_fn
from nvidia_tao_tf1.cv.efficientdet.utils import hparams_config
from nvidia_tao_tf1.cv.efficientdet.utils.model_loader import decode_tlt_file
from nvidia_tao_tf1.cv.efficientdet.utils.spec_loader import (
    generate_params_from_spec,
    load_experiment_spec
)
from nvidia_tao_tf1.cv.efficientdet.utils.distributed_utils import MPI_is_distributed
from nvidia_tao_tf1.cv.efficientdet.utils.distributed_utils import MPI_rank

deprecation._PRINT_DEPRECATION_WARNINGS = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["TF_CPP_VMODULE"] = 'non_max_suppression_op=0,generate_box_proposals_op=0,executor=0'
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')
logger = logging.getLogger(__name__)


def main(args=None):
    """Launch EfficientDet training."""
    disable_eager_execution()
    tf.autograph.set_verbosity(0)
    # parse CLI and config file
    args = parse_command_line_arguments(args)
    print("Loading experiment spec at %s.", args.experiment_spec_file)
    spec = load_experiment_spec(args.experiment_spec_file, merge_from_default=False)

    # set up config
    MODE = 'train'
    # Parse and override hparams
    config = hparams_config.get_detection_config(spec.model_config.model_name)
    params = generate_params_from_spec(config, spec, MODE)
    config.update(params)
    # Update config with parameters in args
    config.key = args.key
    config.model_dir = args.model_dir
    if not MPI_is_distributed() or MPI_rank() == 0:
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
    if config.checkpoint:
        config.checkpoint = decode_tlt_file(config.checkpoint, args.key)
    if config.pruned_model_path:
        config.pruned_model_path = decode_tlt_file(config.pruned_model_path, args.key)

    # Set up dataloader
    train_dataloader = dataloader.InputReader(
        str(Path(spec.dataset_config.training_file_pattern)),
        is_training=True,
        use_fake_data=spec.dataset_config.use_fake_data,
        max_instances_per_image=config.max_instances_per_image)

    eval_dataloader = dataloader.InputReader(
        str(Path(spec.dataset_config.validation_file_pattern)),
        is_training=False,
        max_instances_per_image=config.max_instances_per_image)
    try:
        run_executer(config, train_dataloader, eval_dataloader)
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


def run_executer(runtime_config, train_input_fn=None, eval_input_fn=None):
    """Runs EfficientDet on distribution strategy defined by the user."""
    executer = distributed_executer.EstimatorExecuter(runtime_config,
                                                      det_model_fn.efficientdet_model_fn)
    executer.train_and_eval(train_input_fn=train_input_fn, eval_input_fn=eval_input_fn)
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
        parser = argparse.ArgumentParser(prog='train', description='Train an EfficientDet model.')

    parser.add_argument(
        '-e',
        '--experiment_spec_file',
        type=str,
        required=True,
        help='Path to spec file. Absolute path or relative to working directory. \
                If not specified, default spec from spec_loader.py is used.')
    parser.add_argument(
        '-d',
        '--model_dir',
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
    return parser


def parse_command_line_arguments(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser(args)
    return parser.parse_args(args)


if __name__ == '__main__':
    main()
