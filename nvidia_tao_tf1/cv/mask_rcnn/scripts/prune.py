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

"""MaskRCNN pruning script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import shutil
import tempfile
from zipfile import ZipFile



import tensorflow as tf
from tensorflow import keras  # noqa pylint: disable=F401, W0611

from nvidia_tao_tf1.core.pruning.pruning import prune
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
import nvidia_tao_tf1.cv.common.no_warning # noqa pylint: disable=W0611
from nvidia_tao_tf1.cv.common.utils import get_model_file_size
from nvidia_tao_tf1.cv.mask_rcnn.utils import model_loader
from nvidia_tao_tf1.encoding import encoding

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
                        "--output_dir",
                        type=str,
                        help="Output directory for pruned model",
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
    return parser


def parse_command_line_arguments(args=None):
    """Parse command line arguments for pruning."""
    parser = build_command_line_parser()
    return parser.parse_args(args)


def run_pruning(args=None):
    """Prune an encrypted MRCNN model."""
    # Set up logger verbosity.
    verbosity = 'INFO'
    if args.verbose:
        verbosity = 'DEBUG'
    # Configure the logger.
    logging.basicConfig(
                format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                level=verbosity)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        assert not os.path.exists(os.path.join(args.output_dir, 'model.tlt')), \
            "model.tlt already exists in the output dir. Please verify. \
            Recommend to specify a different output path"
    status_file = os.path.join(args.output_dir, "status.json")
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
        message="Starting MaskRCNN pruning."
    )
    assert args.equalization_criterion in \
        ['arithmetic_mean', 'geometric_mean', 'union', 'intersection'], \
        "Equalization criterion are [arithmetic_mean, geometric_mean, union, \
         intersection]."
    assert args.normalizer in ['L2', 'max'], \
        "normalizer options are [L2, max]."

    temp_dir = tempfile.mkdtemp()

    def prune_graph(args, graph_name='train_graph.json'):
        """Prune MRCNN model graphs with checkpoint."""
        # Load the unpruned model graph from json
        model_dir = os.path.dirname(args.model)
        final_model = model_loader.load_json_model(
            os.path.join(model_dir, graph_name))
        # Decrypt and restore checkpoint
        ckpt_path = model_loader.load_mrcnn_tlt_model(args.model, args.key)
        sess = keras.backend.get_session()
        tf.global_variables_initializer()
        tf.compat.v1.train.Saver().restore(sess, ckpt_path)

        if verbosity == 'DEBUG':
            # Printing out the loaded model summary
            logger.debug("Model summary of the unpruned model:")
            logger.debug(final_model.summary())

        # Excluded layers for MRCNN
        force_excluded_layers = ['class-predict',
                                 'box-predict',
                                 'mask_fcn_logits',
                                 'post_hoc_d2',
                                 'post_hoc_d3',
                                 'post_hoc_d4',
                                 'post_hoc_d5',
                                 'p6']

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
            pruning_ratio))

        if 'train' in graph_name:
            pruned_model.save(os.path.join(temp_dir, 'pruned_model.hdf5'))

        with open(os.path.join(temp_dir, graph_name), "w") as jf:
            jo = json.loads(pruned_model.to_json())
            json.dump(jo, jf, indent=4)
        return pruning_ratio, pruned_model.count_params()

    pruning_ratio, param_count = prune_graph(args, "train_graph.json")
    tf.compat.v1.reset_default_graph()
    pruning_ratio, param_count = prune_graph(args, "eval_graph.json")
    tf.compat.v1.reset_default_graph()

    model_loader.load_keras_model(os.path.join(temp_dir, 'pruned_model.hdf5'))
    km_weights = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=None)
    with open(os.path.join(temp_dir, 'pruned.ckpt'), 'wb') as f:
        tf.compat.v1.train.Saver(km_weights).save(tf.keras.backend.get_session(), f.name)

    # save to etlt
    prev_dir = os.getcwd()
    os.chdir(temp_dir)
    output_model = os.path.join(args.output_dir, "model.tlt")
    with ZipFile(output_model, 'w') as zip_object:
        for model_file in os.listdir(temp_dir):
            if 'hdf5' not in model_file:
                zip_object.write(model_file)
    # Restore previous execution directory and remove tmp files/directories.
    os.chdir(prev_dir)
    shutil.rmtree(temp_dir)
    s_logger.kpi.update(
        {'pruning_ratio': float(pruning_ratio),
         'param_count': param_count,
         'size': get_model_file_size(os.path.join(args.output_dir, 'model.tlt'))})
    s_logger.write(
        status_level=status_logging.Status.SUCCESS,
        message="Pruning finished successfully."
    )


def main(args=None):
    """Wrapper function for pruning."""
    try:
        # parse command line
        args = parse_command_line_arguments(args)
        run_pruning(args)
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
