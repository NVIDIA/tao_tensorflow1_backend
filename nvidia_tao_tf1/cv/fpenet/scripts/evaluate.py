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
"""FpeNet model evaluation script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os

import tensorflow as tf
import yaml

import nvidia_tao_tf1.core as tao_core
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utilities.path_processing import mkdir_p
import nvidia_tao_tf1.cv.fpenet  # noqa # pylint: disable=unused-import


def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
            parser mechanism
    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            prog='evaluate', description='Run FpeNet evaluation.')

    parser.add_argument(
        '-type',
        '--eval_type',
        type=str,
        choices=['kpi_testing'],
        default='kpi_testing',
        help='Type of evaluation to run.')

    parser.add_argument(
        '-m',
        '--model_folder_path',
        type=str,
        required=True,
        help='Path to the folder where the model to be evaluated'
             'is in, or the model file itself.')

    parser.add_argument(
        '-e',
        '--experiment_spec_filename',
        type=str,
        default='experiment_spec.yaml',
        help='Filename of yaml experiment spec to be used for evaluation.')

    parser.add_argument(
        '-ll',
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level.')

    parser.add_argument(
        '-k',
        '--key',
        default="",
        type=str,
        required=False,
        help='The key to load pretrained weights and save intermediate snapshopts and final model.'
    )
    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        default=None,
        help='Path to a folder where experiment outputs will be created, or specify in spec file.')

    return parser


def parse_command_line(args=None):
    """Simple function to parse command line arguments.

    Args:
        args (list): List of strings used as command line arguments.
            If None, sys.argv is used.

    Returns:
        args_parsed: Parsed arguments.
    """
    parser = build_command_line_parser()
    args_parsed = parser.parse_args(args)
    return args_parsed


def main(args=None):
    """Launch the model evaluation process."""
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_command_line(args)

    results_dir = args.results_dir

    if results_dir:
        mkdir_p(results_dir)

    # Writing out status file for TAO.
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            is_master=True,
            verbosity=1,
            append=True
        )
    )

    status_logging.get_status_logger().write(
        data=None,
        status_level=status_logging.Status.STARTED,
        message="Starting evaluation."
    )

    model_folder_path = args.model_folder_path
    experiment_spec = args.experiment_spec_filename
    eval_type = args.eval_type
    key = args.key

    # Build logger file.
    logging.basicConfig(
        format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s')
    logger = logging.getLogger(__name__)
    logger_tf = logging.getLogger('tensorflow')
    logger.setLevel(args.log_level)
    logger_tf.setLevel(args.log_level)

    # Load experiment spec.
    config_path = os.path.join(model_folder_path, experiment_spec)
    print('config_path: ', config_path)
    if not os.path.isfile(config_path):
        raise ValueError("Experiment spec file cannot be found.")
    with open(config_path, 'r') as yaml_file:
        spec = yaml.load(yaml_file.read())

    # Build the model saving directory.
    if not os.path.exists(model_folder_path):
        raise FileNotFoundError(f"Model path doesn't exist at {model_folder_path}")
    spec['checkpoint_dir'] = model_folder_path
    trainer_build_kwargs = {}
    if os.path.isfile(model_folder_path):
        spec['checkpoint_dir'] = os.path.dirname(model_folder_path)
        trainer_build_kwargs["eval_model_path"] = model_folder_path

    # Add key
    if key is not None:
        spec['key'] = key

    # Build trainer with on evaluation mode.
    trainer = tao_core.coreobject.deserialize_tao_object(spec)
    trainer.build(eval_mode=eval_type, **trainer_build_kwargs)
    logger.info("Trainer built.")
    logger.info("Starting evaluation")
    trainer.run_testing()


if __name__ == "__main__":
    try:
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Evaluation finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Evaluation was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
