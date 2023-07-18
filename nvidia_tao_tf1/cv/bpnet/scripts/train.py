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
"""BpNet training script."""

import argparse
import logging
import os

import tensorflow as tf
import yaml

import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.core import distribution
import nvidia_tao_tf1.cv.bpnet # noqa # pylint: disable=unused-import
from nvidia_tao_tf1.cv.bpnet.dataio.coco_dataset import COCODataset
from nvidia_tao_tf1.cv.bpnet.trainers.bpnet_trainer import MODEL_EXTENSION
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
import nvidia_tao_tf1.cv.common.utilities.path_processing as io_utils
from nvidia_tao_tf1.cv.common.utilities.path_processing import mkdir_p
from nvidia_tao_tf1.cv.common.utilities.tlt_utils import get_latest_tlt_model

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

logger = logging.getLogger(__name__)


def build_command_line_parser(parser=None):
    """
    Parse command-line flags passed to the training script.

    Args:
        args (list of str): Command line arguments list.

    Returns:
        Namespace with members for all parsed arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(prog='train', description='Run BpNet train.')

    parser.add_argument(
        '-e',
        '--experiment_spec_file',
        type=str,
        default='nvidia_tao_tf1/cv/bpnet/experiment_specs/experiment_spec.yaml',
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

    # Build logger file.
    logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s')
    logger = logging.getLogger(__name__)
    logger_tf = logging.getLogger('tensorflow')

    # If not on DEBUG, set logging level to 'WARNING' to suppress outputs from other processes.
    level = 'DEBUG' if args.log_level == 'DEBUG' else 'WARNING'
    logger.setLevel(level)

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

    distribution.set_distributor(distribution.HorovodDistributor())
    is_master = distribution.get_distributor().is_master()

    if is_master:
        logger.setLevel(args.log_level)
        logger_tf.setLevel(args.log_level)

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
    logger.info('done')
    trainer.build()

    logger.info('training')
    status_logging.get_status_logger().write(
        data=None,
        status_level=status_logging.Status.STARTED,
        message="Starting BPnet training."
    )

    trainer.train()
    logger.info('Training has finished...')
    status_logging.get_status_logger().write(
        data=None,
        status_level=status_logging.Status.RUNNING,
        message="BPnet training loop finished."
    )

    # Save the training spec in the results directory.
    if is_master:
        trainer.to_yaml(os.path.join(results_dir, 'experiment_spec.yaml'))

    status_logging.get_status_logger().write(
        data=None,
        status_level=status_logging.Status.SUCCESS,
        message="BPnet training experimenent finished successfully."
    )


if __name__ == "__main__":
    try:
        main()
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
