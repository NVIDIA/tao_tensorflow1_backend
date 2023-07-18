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

"""EfficientDet pruning script."""

import argparse
import json
import logging
import os
import shutil
import tempfile
import time
from zipfile import ZipFile

import tensorflow as tf
from tensorflow import keras  # noqa pylint: disable=F401, W0611

from nvidia_tao_tf1.core.pruning.pruning import prune
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
import nvidia_tao_tf1.cv.common.no_warning # noqa pylint: disable=W0611
from nvidia_tao_tf1.cv.common.utils import get_model_file_size
from nvidia_tao_tf1.cv.efficientdet.utils.model_loader import dump_json, load_json_model
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


def extract_zipfile_ckpt(zip_path):
    """Extract the contents of an efficientdet ckpt zip file.

    Args:
        zip_path (str): Path to a zipfile.

    Returns:
        checkpoint_path (str): Path to the checkpoint extracted.
    """
    temp_ckpt_dir = tempfile.mkdtemp()
    with ZipFile(zip_path, 'r') as zip_object:
        for member in zip_object.namelist():
            zip_object.extract(member, path=temp_ckpt_dir)
            if member.startswith('model.ckpt-'):
                step = int(member.split('model.ckpt-')[-1].split('.')[0])
    return os.path.join(temp_ckpt_dir, "model.ckpt-{}".format(step))


def extract_ckpt(encoded_checkpoint, key):
    """Get unencrypted checkpoint from tlt file."""
    logging.info("Loading weights from {}".format(encoded_checkpoint))
    try:
        # Load an unencrypted checkpoint as 5.0.
        checkpoint_path = extract_zipfile_ckpt(encoded_checkpoint)
    except BadZipFile:
        # Decrypt and load the checkpoint.
        os_handle, temp_zip_path = tempfile.mkstemp()
        os.close(os_handle)

        # Decrypt the checkpoint file.
        with open(encoded_checkpoint, 'rb') as encoded_file, open(temp_zip_path, 'wb') as tmp_zipf:
            encoding.decode(encoded_file, tmp_zipf, key.encode())
        encoded_file.closed
        tmp_zipf.closed
        checkpoint_path = extract_zipfile_ckpt(temp_zip_path)
        os.remove(temp_zip_path)
    return checkpoint_path


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
    DEBUG_MODE = False
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

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    elif len(os.listdir(args.output_dir)) > 0:
        raise ValueError("Output directory is not empty. \
            Please specify a new directory or clean up the current one.")
    output_file = os.path.join(args.output_dir, 'model.tlt')

    """Prune MRCNN model graphs with checkpoint."""
    # Load the unpruned model
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
        message="Starting EfficientDet pruning."
    )
    model_dir = os.path.dirname(args.model)
    final_model = load_json_model(
        os.path.join(model_dir, 'graph.json'))
    # Decrypt and restore checkpoint
    ckpt_path = extract_ckpt(args.model, args.key)
    if DEBUG_MODE:
        # selectively restore checkpoint
        reader = tf.compat.v1.train.NewCheckpointReader(ckpt_path)
        restore_dict = {}
        for v in tf.trainable_variables():
            tensor_name = v.name.split(':')[0]
            if reader.has_tensor(tensor_name):
                restore_dict[tensor_name] = v
            else:
                print(tensor_name)
        saver = tf.compat.v1.train.Saver(restore_dict)
        sess = keras.backend.get_session()
        tf.global_variables_initializer()
        saver.restore(sess, ckpt_path)

    # restore checkpoint
    sess = keras.backend.get_session()
    tf.global_variables_initializer()
    tf.compat.v1.train.Saver().restore(sess, ckpt_path)

    if verbosity == 'DEBUG':
        # Printing out the loaded model summary
        logger.debug("Model summary of the unpruned model:")
        logger.debug(final_model.summary())

    # Excluded layers for Effdet
    force_excluded_layers = []
    force_excluded_layers += final_model.output_names
    t0 = time.time()
    logger.info("Pruning process will take some time. Please wait...")
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
    logger.debug("Elapsed time: {}".format(time.time() - t0))
    # zip pruned hdf5 and save to tlt file
    temp_dir = tempfile.mkdtemp()
    pruned_model.save(os.path.join(temp_dir, "pruned_model.hdf5"))

    # save train graph in json
    dump_json(pruned_model, os.path.join(temp_dir, "pruned_train.json"))
    # generate eval graph for exporting. (time saving hack)
    with open(os.path.join(temp_dir, "pruned_train.json"), 'r') as f:
        pruned_json = json.load(f)
        for layer in pruned_json['config']['layers']:
            if layer['class_name'] == 'PatchedBatchNormalization':
                if layer['inbound_nodes'][0][0][-1]:
                    layer['inbound_nodes'][0][0][-1]['training'] = False
    with open(os.path.join(temp_dir, "pruned_eval.json"), 'w') as jf:
        json.dump(pruned_json, jf)
    # save to tlt
    prev_dir = os.getcwd()
    os.chdir(temp_dir)
    with ZipFile(output_file, 'w') as zip_object:
        written_flag = False
        for model_file in os.listdir(temp_dir):
            if 'pruned_' in model_file:
                zip_object.write(model_file)
                written_flag = True
        assert written_flag, "The pruned model is not saved probably. \
            Please rerun the pruning script."
    # Restore previous execution directory and remove tmp files/directories.
    os.chdir(prev_dir)
    s_logger.kpi.update(
        {
            'pruning_ratio': float(pruning_ratio),
            'param_count': pruned_model.count_params(),
            'size': get_model_file_size(output_file)
        }
    )
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
