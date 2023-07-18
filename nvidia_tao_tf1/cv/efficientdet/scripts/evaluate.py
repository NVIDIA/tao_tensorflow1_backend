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

"""EfficientDet evaluation script."""

import argparse
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

deprecation._PRINT_DEPRECATION_WARNINGS = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["TF_CPP_VMODULE"] = 'non_max_suppression_op=0,generate_box_proposals_op=0,executor=0'


def main(args=None):
    """Launch EfficientDet training."""
    disable_eager_execution()
    tf.autograph.set_verbosity(0)
    # parse CLI and config file
    args = parse_command_line_arguments(args)
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    status_file = os.path.join(args.results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            is_master=True,
            verbosity=1,
            append=True
        )
    )
    s_logger = status_logging.get_status_logger()
    s_logger.write(
        status_level=status_logging.Status.STARTED,
        message="Starting EfficientDet evaluation."
    )
    print("Loading experiment spec at %s.", args.experiment_spec)
    spec = load_experiment_spec(args.experiment_spec, merge_from_default=False)

    # set up config
    MODE = 'eval'
    # Parse and override hparams
    config = hparams_config.get_detection_config(spec.model_config.model_name)
    params = generate_params_from_spec(config, spec, MODE)
    config.update(params)
    config.key = args.key
    config.model_path = args.model_path
    if config.pruned_model_path:
        config.pruned_model_path = decode_tlt_file(config.pruned_model_path, args.key)
    # Set up dataloader
    eval_dataloader = dataloader.InputReader(
        str(Path(spec.dataset_config.validation_file_pattern)),
        is_training=False,
        max_instances_per_image=config.max_instances_per_image)

    eval_results = run_executer(config, eval_dataloader)
    for k, v in eval_results.items():
        s_logger.kpi[k] = float(v)
    s_logger.write(
        status_level=status_logging.Status.SUCCESS,
        message="Evaluation finished successfully."
    )


def run_executer(runtime_config, eval_input_fn=None):
    """Runs EfficientDet on distribution strategy defined by the user."""
    executer = distributed_executer.EstimatorExecuter(runtime_config,
                                                      det_model_fn.efficientdet_model_fn)
    eval_results = executer.eval(eval_input_fn=eval_input_fn)
    return eval_results


def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
                parser mechanism.
    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(prog='eval', description='Evaluate an EfficientDet model.')

    parser.add_argument(
        '-e',
        '--experiment_spec',
        type=str,
        required=True,
        help='Path to spec file. Absolute path or relative to working directory. \
                If not specified, default spec from spec_loader.py is used.')
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        required=True,
        help='Path to a trained EfficientDet model.'
    )
    parser.add_argument(
        '-k',
        '--key',
        type=str,
        default="",
        required=False,
        help='Key to save or load a .tlt model.'
    )
    parser.add_argument(
        '-i',
        '--image_dir',
        type=str,
        required=False,
        default=None,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        default='/tmp',
        required=False,
        help='Output directory where the status log is saved.'
    )
    return parser


def parse_command_line_arguments(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser(args)
    return parser.parse_args(args)


if __name__ == '__main__':
    try:
        main()
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
