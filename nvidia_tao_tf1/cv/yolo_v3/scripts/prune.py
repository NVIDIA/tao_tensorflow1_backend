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

"""MagNet pruning wrapper for classification/detection models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime as dt
import logging
import os

from nvidia_tao_tf1.core.pruning.pruning import prune
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utils import (
    get_model_file_size,
    get_num_params
)
from nvidia_tao_tf1.cv.yolo_v3.utils.model_io import load_model, save_model
from nvidia_tao_tf1.cv.yolo_v3.utils.spec_loader import load_experiment_spec

logger = logging.getLogger(__name__)


def build_command_line_parser(parser=None):
    '''build parser.'''
    if parser is None:
        parser = argparse.ArgumentParser(description="TLT pruning script")
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        help="Path to the target model for pruning",
                        required=True,
                        default=None)
    parser.add_argument("-o",
                        "--output_file",
                        type=str,
                        help="Output file path for pruned model",
                        required=True,
                        default=None)
    parser.add_argument("-e",
                        "--experiment_spec_path",
                        type=str,
                        help="Path to experiment spec file",
                        required=True)
    parser.add_argument('-k',
                        '--key',
                        required=False,
                        default="",
                        type=str,
                        help='Key to load a .tlt model')
    parser.add_argument('-n',
                        '--normalizer',
                        type=str,
                        default='max',
                        help="`max` to normalize by dividing each norm by the \
                        maximum norm within a layer; `L2` to normalize by \
                        dividing by the L2 norm of the vector comprising all \
                        kernel norms. (default: `max`)")
    parser.add_argument('-eq',
                        '--equalization_criterion',
                        type=str,
                        default='union',
                        help="Criteria to equalize the stats of inputs to an \
                        element wise op layer. Options are \
                        [arithmetic_mean, geometric_mean, union, \
                        intersection]. (default: `union`)")
    parser.add_argument("-pg",
                        "--pruning_granularity",
                        type=int,
                        help="Pruning granularity: number of filters to remove \
                        at a time. (default:8)",
                        default=8)
    parser.add_argument("-pth",
                        "--pruning_threshold",
                        type=float,
                        help="Threshold to compare normalized norm against \
                        (default:0.1)", default=0.1)
    parser.add_argument("-nf",
                        "--min_num_filters",
                        type=int,
                        help="Minimum number of filters to keep per layer. \
                        (default:16)", default=16)
    parser.add_argument("-el",
                        "--excluded_layers", action='store',
                        type=str, nargs='*',
                        help="List of excluded_layers. Examples: -i item1 \
                        item2", default=[])
    parser.add_argument("--results_dir",
                        type=str,
                        default=None,
                        help="Path to the files where the logs are stored.")
    parser.add_argument("-v",
                        "--verbose",
                        action='store_true',
                        help="Include this flag in command line invocation for \
                        verbose logs.")
    return parser


def parse_command_line(args):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser()
    return parser.parse_args(args)


def run_pruning(args=None):
    """Prune an encrypted Keras model."""
    results_dir = args.results_dir
    if results_dir is not None:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        timestamp = int(dt.timestamp(dt.now()))
        filename = "status.json"
        if results_dir == "/workspace/logs":
            filename = f"status_prune_{timestamp}.json"
        status_file = os.path.join(results_dir, filename)
        status_logging.set_status_logger(
            status_logging.StatusLogger(
                filename=status_file,
                is_master=True
            )
        )
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.STARTED,
            message="Starting YOLO pruning"
        )

    # Set up logger verbosity.
    verbosity = 'INFO'
    if args.verbose:
        verbosity = 'DEBUG'
    # Configure the logger.
    logging.basicConfig(
                format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                level=verbosity)
    assert args.equalization_criterion in \
        ['arithmetic_mean', 'geometric_mean', 'union', 'intersection'], \
        "Equalization criterion are [arithmetic_mean, geometric_mean, union, \
         intersection]."
    assert args.normalizer in ['L2', 'max'], \
        "normalizer options are [L2, max]."
    experiment_spec = load_experiment_spec(args.experiment_spec_path)
    img_height = experiment_spec.augmentation_config.output_height
    img_width = experiment_spec.augmentation_config.output_width
    n_channels = experiment_spec.augmentation_config.output_channel
    final_model = load_model(
        args.model,
        experiment_spec,
        (n_channels, img_height, img_width),
        key=args.key
    )
    if verbosity == 'DEBUG':
        # Printing out the loaded model summary
        logger.debug("Model summary of the unpruned model:")
        logger.debug(final_model.summary())
    # Exckuded layers for YOLOv3 / v4
    force_excluded_layers = [
        'conv_big_object',
        'conv_mid_object',
        'conv_sm_object'
    ]
    force_excluded_layers += final_model.output_names
    # Pruning trained model
    pruned_model = prune(
        model=final_model,
        method='min_weight',
        normalizer=args.normalizer,
        criterion='L2',
        granularity=args.pruning_granularity,
        min_num_filters=args.min_num_filters,
        threshold=args.pruning_threshold,
        equalization_criterion=args.equalization_criterion,
        excluded_layers=args.excluded_layers + force_excluded_layers)
    if verbosity == 'DEBUG':
        # Printing out pruned model summary
        logger.debug("Model summary of the pruned model:")
        logger.debug(pruned_model.summary())
    pruning_ratio = pruned_model.count_params() / final_model.count_params()
    logger.info("Pruning ratio (pruned model / original model): {}".format(
            pruning_ratio
        )
    )
    # Save the encrypted pruned model
    save_model(pruned_model, args.output_file, args.key, save_format='.hdf5')
    if results_dir is not None:
        s_logger = status_logging.get_status_logger()
        s_logger.kpi = {
            "pruning_ratio": pruning_ratio,
            "size": get_model_file_size(args.output_file),
            "param_count": get_num_params(pruned_model)
        }
        s_logger.write(
            message="Pruning ratio (pruned model / original model): {}".format(
                pruning_ratio
            )
        )


def main(args=None):
    """Wrapper function for pruning."""
    # Apply patch to correct keras 2.2.4 bug
    try:
        # parse command line
        args = parse_command_line(args)
        run_pruning(args)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Pruning finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Pruning was interrupted",
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
