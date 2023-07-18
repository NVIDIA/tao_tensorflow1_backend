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
"""FpeNet training script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os

formatter = logging.Formatter(
    "%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s")  # noqa
handler = logging.StreamHandler()  # noqa
handler.setFormatter(formatter)  # noqa
logging.basicConfig(
    level='INFO'
)  # noqa
# Replace existing handlers with ours to avoid duplicate messages.
logging.getLogger().handlers = []  # noqa
logging.getLogger().addHandler(handler)  # noqa

import tensorflow as tf
import yaml

import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.core import distribution
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utilities.path_processing import mkdir_p
import nvidia_tao_tf1.cv.fpenet # noqa # pylint: disable=unused-import

logger = logging.getLogger(__name__)


def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
            parser mechanism.
    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(prog='train', description='Run FpeNet training.')

    parser.add_argument(
        '-e',
        '--experiment_spec_file',
        type=str,
        default='nvidia_tao_tf1/cv/fpenet/experiment_specs/default.yaml',
        help='Path to a single file containing a complete experiment spec.')

    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        default=None,
        help='Path to a folder where experiment outputs will be created, or specify in spec file.')

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


def main(cl_args=None):
    """Launch the training process."""
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_command_line(cl_args)
    config_path = args.experiment_spec_file
    results_dir = args.results_dir
    key = args.key

    # Load experiment spec.
    if not os.path.isfile(config_path):
        raise ValueError("Experiment spec file cannot be found.")
    with open(config_path, 'r') as yaml_file:
        spec = yaml.load(yaml_file.read())

    # Build the model saving directory.
    if results_dir is not None:
        spec['checkpoint_dir'] = results_dir
    elif spec['checkpoint_dir']:
        results_dir = spec['checkpoint_dir']
    else:
        raise ValueError('Checkpoint directory not specified, please specify it through -r or'
                         'through the checkpoint_dir field in your model config.')
    mkdir_p(results_dir)

    # Add key
    if key is not None:
        spec['key'] = key

    # Use Horovod distributor for multi-gpu training.
    distribution.set_distributor(distribution.HorovodDistributor())

    is_master = distribution.get_distributor().is_master()
    # Set logger level
    if is_master:
        logger.setLevel(args.log_level)

    # Writing out status file for TAO.
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            is_master=is_master,
            verbosity=1,
            append=True
        )
    )
    # Build trainer from spec.
    trainer = tao_core.coreobject.deserialize_tao_object(spec)
    trainer.build()
    logger.info('Build trainer finished. Starting training...')

    status_logging.get_status_logger().write(
        data=None,
        status_level=status_logging.Status.STARTED,
        message="Starting fpenet training."
    )

    # Save the training spec in the results directory.
    if distribution.get_distributor().is_master():
        trainer.to_yaml(os.path.join(results_dir, 'experiment_spec.yaml'))

    trainer.train()
    status_logging.get_status_logger().write(
        data=None,
        status_level=status_logging.Status.SUCCESS,
        message="Fpenet training finished successfully."
    )


if __name__ == "__main__":
    try:
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Training finished successfully."
        )
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
