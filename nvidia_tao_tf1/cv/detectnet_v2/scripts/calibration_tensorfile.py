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

"""Utilities for dumping dataset tensors to TensorFile for int8 calibration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys

import tensorflow as tf
from tqdm import trange

from nvidia_tao_tf1.core.export.data import TensorFile
from nvidia_tao_tf1.cv.detectnet_v2.common.graph import get_init_ops
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.build_dataloader import build_dataloader
from nvidia_tao_tf1.cv.detectnet_v2.spec_handler.spec_loader import load_experiment_spec
from nvidia_tao_tf1.cv.detectnet_v2.training.utilities import initialize
from nvidia_tao_tf1.cv.detectnet_v2.utilities.timer import time_function

logger = logging.getLogger(__name__)


def dump_dataset_images_to_tensorfile(experiment_spec, output_path, training, max_batches):
    """Dump dataset images to a nvidia_tao_tf1.core.data.TensorFile object and store it to disk.

    The file can be used as an input to e.g. INT8 calibration.

    Args:
        experiment_spec: experiment_pb2.Experiment object containing experiment parameters.
        output_path (str): Path for the TensorFile to be created.
        training (bool): Whether to dump images from the training or validation set.
        max_batches (int): Maximum number of minibatches to dump.

    Returns:
        tensor_file: nvidia_tao_tf1.core.data.TensorFile object.
    """
    dataset_config = experiment_spec.dataset_config
    augmentation_config = experiment_spec.augmentation_config
    batch_size = experiment_spec.training_config.batch_size_per_gpu

    dataloader = build_dataloader(dataset_config, augmentation_config)

    images, _, num_samples = dataloader.get_dataset_tensors(batch_size,
                                                            training=training,
                                                            enable_augmentation=False,
                                                            repeat=True)

    batches_in_dataset = num_samples // batch_size

    # If max_batches is not supplied, then dump the whole dataset.
    max_batches = batches_in_dataset if max_batches == -1 else max_batches

    if max_batches > batches_in_dataset:
        raise ValueError("The dataset contains %d minibatches, while the requested amount is %d." %
                         (batches_in_dataset, max_batches))

    tensor_file = dump_to_tensorfile(images, output_path, max_batches)

    return tensor_file


def dump_to_tensorfile(tensor, output_path, max_batches):
    """Dump iterable tensor to a TensorFile.

    Args:
        tensor: Tensor that can be iterated over.
        output_path: Path for the TensorFile to be created.
        max_batches: Maximum number of minibatches to dump.

    Returns:
        tensor_file: TensorFile object.
    """
    output_root = os.path.dirname(output_path)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    else:
        if os.path.exists(output_path):
            raise ValueError("A previously generated tensorfile already exists in the output path."
                             " Please delete this file before writing a new one.")

    tensor_file = TensorFile(output_path, 'w')

    tr = trange(max_batches, file=sys.stdout)
    tr.set_description("Writing calibration tensorfile")

    with tf.Session() as session:
        session.run(get_init_ops())

        for _ in tr:
            batch_tensors = session.run(tensor)
            tensor_file.write(batch_tensors)

    return tensor_file


def build_command_line_parser(parser=None):
    """Simple function to build a command line parser."""
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="calibration_tensorfile",
            description="Tool to generate random batches of train/val data for calibration."
        )
    parser.add_argument(
        '-e',
        '--experiment_spec_file',
        type=str,
        help='Absolute path to the experiment spec file.'
    )
    parser.add_argument(
        '-o',
        '--output_path',
        type=str,
        help='Path to the TensorFile that will be created.'
    )
    parser.add_argument(
        '-m',
        '--max_batches',
        type=int,
        default=-1,
        help='Maximum number of minibatches to dump. The default is to dump the whole dataset.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Set verbosity level for the logger.'
    )
    parser.add_argument(
        '--use_validation_set',
        action='store_true',
        help='If set, then validation images are dumped. Otherwise, training images are dumped.'
    )
    return parser


def parse_command_line_arguments(cl_args=None):
    """Parse command line arguments."""
    parser = build_command_line_parser()
    return parser.parse_args(cl_args)


@time_function(__name__)
def main(args=None):
    """Run the dataset dump."""
    args = parse_command_line_arguments(args)

    # Set up logger verbosity.
    verbosity = 'INFO'
    if args.verbose:
        verbosity = 'DEBUG'

    # Configure the logger.
    logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                        level=verbosity)

    logger.info(
        "This method is soon to be deprecated. Please use the -e option in the export command "
        "to instantiate the dataloader and generate samples for calibration from the "
        "training dataloader."
    )
    experiment_spec = load_experiment_spec(args.experiment_spec_file, merge_from_default=False,
                                           validation_schema="train_val")
    training = not args.use_validation_set
    output_path = args.output_path
    max_batches = args.max_batches

    # Set seed. Training precision left untouched as it is irrelevant here.
    initialize(random_seed=experiment_spec.random_seed, training_precision=None)

    tensorfile = dump_dataset_images_to_tensorfile(experiment_spec,
                                                   output_path, training,
                                                   max_batches)
    tensorfile.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if type(e) == tf.errors.ResourceExhaustedError:
            logger.error(
                "Ran out of GPU memory, please lower the batch size, use a smaller input "
                "resolution, or use a smaller backbone."
            )
            exit(1)
        else:
            # throw out the error as-is if they are not OOM error
            raise e
