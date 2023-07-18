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

"""BpNet pruning wrapper."""

import argparse
import logging
import os

from nvidia_tao_tf1.core.pruning.pruning import prune

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
import nvidia_tao_tf1.cv.common.utilities.path_processing as io_utils
from nvidia_tao_tf1.cv.common.utilities.tlt_utils import model_io

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
        parser = argparse.ArgumentParser(description="Run BpNet pruning.")
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
    parser.add_argument("-v",
                        "--verbose",
                        action='store_true',
                        help="Include this flag in command line invocation for \
                        verbose logs.")
    parser.add_argument('-r',
                        '--results_dir',
                        type=str,
                        default=None,
                        help='Path to a folder where experiment outputs will be created, \
                        or specify in spec file.')
    return parser


def parse_command_line(args):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser()
    return parser.parse_args(args)


def run_pruning(args=None):
    """Prune an encrypted Keras model."""
    results_dir = args.results_dir
    output_file = args.output_file

    # Make results dir if it doesn't already exist
    if results_dir:
        io_utils.mkdir_p(results_dir)
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
        message="Starting pruning."
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

    final_model = model_io(args.model, enc_key=args.key)

    # Fix bug 3869039
    try:
        final_model = final_model.get_layer("model_1")
    except Exception:
        pass

    # Make results dir if it doesn't already exist
    if not os.path.exists(os.path.dirname(output_file)):
        io_utils.mkdir_p(os.path.dirname(output_file))

    # TODO: Set shape and print summary to understand
    # the reduction in channels after pruning
    # logger.info("Unpruned model summary")
    # final_model.summary() # Disabled for TLT release
    # Printing out the loaded model summary
    force_excluded_layers = []
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
        excluded_layers=args.excluded_layers + force_excluded_layers,
        output_layers_with_outbound_nodes=force_excluded_layers)
    # logger.info("Model summary of the pruned model")
    # pruned_model.summary() # Disabled for TLT release
    logger.info("Number of params in original model): {}".format(
        final_model.count_params()))
    logger.info("Number of params in pruned model): {}".format(
        pruned_model.count_params()))
    logger.info("Pruning ratio (pruned model / original model): {}".format(
        pruned_model.count_params() / final_model.count_params()))

    # Save the encrypted pruned model
    if not output_file.endswith(".hdf5"):
        output_file = f"{output_file}.hdf5"

    # Save decrypted pruned model.
    pruned_model.save(output_file, overwrite=True)


def main(args=None):
    """Wrapper function for pruning."""

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
