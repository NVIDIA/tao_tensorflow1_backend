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
"""Modulus pruning.

This module includes APIs to prune a Keras models.
Pruning is currently supported only for sequential models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import logging.config
import os
import sys
import time

from nvidia_tao_tf1.core.models.import_keras import keras as keras_fn
from nvidia_tao_tf1.core.pruning.pruning import prune
from nvidia_tao_tf1.core.utils.path_utils import expand_path

keras = keras_fn()
"""Root logger for pruning app."""
logger = logging.getLogger(__name__)


def prune_app(
    input_filename,
    output_filename,
    verbose,
    method,
    normalizer,
    criterion,
    granularity,
    min_num_filters,
    threshold,
    excluded_layers,
    equalization_criterion,
    output_layers_with_outbound_nodes
):
    """Wrapper around :any:`modulus.pruning.pruning.prune`.

    Args:
        input_filename (str): path to snapshot of model to prune
        output_filename (str): output filename (defaults to $(input).pruned)
        verbose (boolean): whether to print debug messages

    See :any:`modulus.pruning.pruning.prune` for more information on the other arguments.
    """
    start_time = time.time()

    # Set up logging.
    verbosity = "DEBUG" if verbose else "INFO"

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=verbosity
    )

    logger.info("Loading model from %s" % (input_filename))

    # Load model from disk.
    model = keras.models.load_model(input_filename, compile=False)

    logger.info("Original model - param count: %d" % model.count_params())

    # Create list of exclude layers from command-line, if provided.
    if excluded_layers is not None:
        excluded_layers = excluded_layers.split(",")

    # Create list of output layers with outbound nodes from command-line, if provided.
    if output_layers_with_outbound_nodes is not None:
        output_layers_with_outbound_nodes = output_layers_with_outbound_nodes.split(",")

    # Prune model given specified parameters.
    new_model = prune(
        model,
        method,
        normalizer,
        criterion,
        granularity,
        min_num_filters,
        threshold,
        excluded_layers=excluded_layers,
        equalization_criterion=equalization_criterion,
        output_layers_with_outbound_nodes=output_layers_with_outbound_nodes,
    )

    logger.info("New model - param count: %d" % new_model.count_params())

    if output_filename is None:
        output_filename = input_filename + ".pruned"

    logger.info("Saving pruned model into %s" % (output_filename))

    # Save pruned model to disk.
    dirname = os.path.dirname(output_filename)
    if not os.path.exists(expand_path(dirname)):
        os.makedirs(expand_path(dirname))
    new_model.save(output_filename)

    logger.debug("Done after %s seconds" % (time.time() - start_time,))


def main(args=None):
    """Pruning application.

    If MagLev was installed through ``pip`` then this application can be
    run from a shell. For example::

        $ maglev-prune model.h5 --threshold 0.1

    See command-line help for more information.

    Args:
        args (list): Arguments to parse.
    """
    # Reduce TensorFlow verbosity
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser(description="Prune a string of conv/fc nodes")

    # Positional arguments.
    parser.add_argument("input_filename", help="Input file (.h5 Keras snapshot)")

    # Optional arguments.
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file (defaults to $(input_filename).pruned)",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="min_weight",
        help="Pruning method (currently only 'min_weight' is supported)",
    )

    parser.add_argument(
        "-n",
        "--normalizer",
        type=str,
        default="max",
        help="Normalizer type (off, L2, max)",
    )

    parser.add_argument(
        "-c", "--criterion", type=str, default="L2", help="Criterion (L2, activations)"
    )

    parser.add_argument(
        "-e",
        "--excluded_layers",
        type=str,
        default=None,
        help="Comma separated list of layers to be excluded from pruning.",
    )

    parser.add_argument(
        "--output_layers_with_outbound_nodes",
        type=str,
        default=None,
        help="Comma separated list of output layers that have outbound nodes.",
    )

    parser.add_argument(
        "--equalization_criterion",
        type=str,
        help="Equalization criterion to be used for inputs to an element-wise op.",
        choices=["union", "intersection", "arithmetic_mean", "geometric_mean"],
    )

    parser.add_argument("-g", "--granularity", type=int, default=8, help="Granularity")

    parser.add_argument(
        "-m", "--min_num_filters", type=int, default=16, help="Min number of filters"
    )

    parser.add_argument(
        "-t", "--threshold", type=float, default=0.01, help="Pruning threshold"
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose messages")

    if not args:
        args = sys.argv[1:]
    args = vars(parser.parse_args(args))

    prune_app(
        args["input_filename"],
        args["output"],
        args["verbose"],
        args["method"],
        args["normalizer"],
        args["criterion"],
        args["granularity"],
        args["min_num_filters"],
        args["threshold"],
        args["excluded_layers"],
        args["equalization_criterion"],
        args["output_layers_with_outbound_nodes"],
    )


if __name__ == "__main__":
    main()
