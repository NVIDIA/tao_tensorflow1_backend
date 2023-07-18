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

# we have to import keras here although we do not use it at all
# to avoid the circular import error in the two patches below.
# circular import issue: keras -> third_party.keras.mixed_precision -> keras
# TODO(@zhimengf): Ideally, we have to patch the keras patches in the keras __init__.py
# instead of calling third_party.keras.mixed_precision in the iva code base,
# as it is the way in dazel.
import keras  # noqa pylint: disable=F401, W0611

from nvidia_tao_tf1.core.pruning.pruning import prune
from nvidia_tao_tf1.core.utils.path_utils import expand_path
import nvidia_tao_tf1.cv.common.no_warning # noqa pylint: disable=W0611
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utils import (
    encode_from_keras,
    get_model_file_size,
    get_num_params,
    model_io,
    restore_eff
)
from nvidia_tao_tf1.cv.yolo_v4.layers.split import Split

logger = logging.getLogger(__name__)


def build_command_line_parser(parser=None):
    """Build a command line parser for pruning."""
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
    parser.add_argument("--results_dir",
                        type=str,
                        default=None,
                        help="Path to where the status log is generated.")
    parser.add_argument('-k',
                        '--key',
                        required=False,
                        type=str,
                        default="",
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
    parser.add_argument("-v",
                        "--verbose",
                        action='store_true',
                        help="Include this flag in command line invocation for \
                        verbose logs.")
    return parser


def parse_command_line_arguments(args=None):
    """Parse command line arguments for pruning."""
    parser = build_command_line_parser()
    return parser.parse_args(args)


def run_pruning(args=None):
    """Prune an encrypted Keras model."""
    # Set up logger verbosity.
    verbosity = 'INFO'
    if args.verbose:
        verbosity = 'DEBUG'
    # Configure the logger.
    logging.basicConfig(
        format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
        level=verbosity
    )
    results_dir = args.results_dir
    if results_dir is not None:
        if not os.path.exists(expand_path(results_dir)):
            os.makedirs(expand_path(results_dir))
        timestamp = int(dt.timestamp(dt.now()))
        filename = "status.json"
        if results_dir == "/workspace/logs":
            filename = f"status_prune_{timestamp}.json"
        status_file = os.path.join(expand_path(results_dir), filename)
        status_logging.set_status_logger(
            status_logging.StatusLogger(
                filename=status_file,
                is_master=True
            )
        )

    assert args.equalization_criterion in \
        ['arithmetic_mean', 'geometric_mean', 'union', 'intersection'], \
        "Equalization criterion are [arithmetic_mean, geometric_mean, union, \
         intersection]."
    assert args.normalizer in ['L2', 'max'], \
        "normalizer options are [L2, max]."

    custom_objs = {}

    # Decrypt and load the pretrained model
    final_model = model_io(
        expand_path(args.model),
        args.key,
        custom_objects=custom_objs,
        compile=False
    )

    if verbosity == 'DEBUG':
        # Printing out the loaded model summary
        logger.debug("Model summary of the unpruned model:")
        logger.debug(final_model.summary())

    # Excluded layers for FRCNN
    force_excluded_layers = ['rpn_out_class',
                             'rpn_out_regress',
                             'dense_class_td',
                             'dense_regress_td']

    # Excluded layers for SSD
    force_excluded_layers += ['ssd_conf_0', 'ssd_conf_1', 'ssd_conf_2',
                              'ssd_conf_3', 'ssd_conf_4', 'ssd_conf_5',
                              'ssd_loc_0', 'ssd_loc_1', 'ssd_loc_2',
                              'ssd_loc_3', 'ssd_loc_4', 'ssd_loc_5',
                              'ssd_predictions']

    # Exckuded layers for YOLOv3 / v4
    force_excluded_layers += ['conv_big_object', 'conv_mid_object',
                              'conv_sm_object']

    # Excluded layers for RetinaNet
    force_excluded_layers += ['retinanet_predictions',
                              'retinanet_loc_regressor',
                              'retinanet_conf_regressor']
    # For CSPDarkNetTiny backbone
    # Cannot prune input layers of Split layer
    for layer in final_model.layers:
        if type(layer) == Split:
            basename = layer.name[:-8]
            name = basename + "_conv_0"
            force_excluded_layers.append(name)
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
    logger.info(
        "Pruning ratio (pruned model / original model): {}".format(
            pruning_ratio
        )
    )

    # Save the pruned model.
    output_file = args.output_file
    if not output_file.endswith(".hdf5"):
        output_file = f"{output_file}.hdf5"
    encode_from_keras(
        pruned_model,
        output_file,
        args.key,
        custom_objects=custom_objs
    )
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
    # parse command line
    args = parse_command_line_arguments(args)
    run_pruning(args)


if __name__ == "__main__":
    main()
