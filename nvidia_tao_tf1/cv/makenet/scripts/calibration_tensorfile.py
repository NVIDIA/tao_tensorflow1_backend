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
"""Dump training samples for INT8 inference."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
import logging
import os
import sys

from keras.preprocessing.image import ImageDataGenerator

from tqdm import trange

from nvidia_tao_tf1.core.export import TensorFile
from nvidia_tao_tf1.cv.makenet.spec_handling.spec_loader import load_experiment_spec
from nvidia_tao_tf1.cv.makenet.utils import preprocess_crop  # noqa pylint: disable=unused-import
from nvidia_tao_tf1.cv.makenet.utils.preprocess_input import preprocess_input

# Defining multiple image extensions.
SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".png", ".jpeg"]

logger = logging.getLogger(__name__)


def build_command_line_parser(parser=None):
    """Function to build command line parser."""
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Dump training samples for INT8")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to the output TensorFile", default=None)
    parser.add_argument("-e", "--experiment_spec", type=str, required=True,
                        help="Path to the experiment spec file.", default=None)
    parser.add_argument("-m", "--max_batches", type=int,
                        help="Number of batches", default=1)
    parser.add_argument("-v", "--verbose", action='store_true',
                        help='Set the verbosity of the log.')
    parser.add_argument("--use_validation_set", action='store_true',
                        help="Set to use only validation set.")
    return parser


def parse_command_line(args=None):
    """Parse command line arguments for dumping image samples for INT8 calibration."""
    parser = build_command_line_parser(parser=None)
    arguments = vars(parser.parse_args(args))
    return arguments


def dump_samples(output_path, config_file=None,
                 n_batches=1, verbosity=False,
                 use_validation_set=False):
    """Main wrapper function for dumping image samples.

    Args:
        output_path (str): Directory the output tensorfile should be written.
        config_file (str): Path to experiment config file.
        n_batches (int): Number of batches to be dumped.
        verbosity (bool): Enable verbose logs.
        use_validation_set (bool): Flag to use training or validation set.
    """
    # Configure the logger.
    logging.basicConfig(
        format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
        level='DEBUG' if verbosity else 'INFO')

    if config_file is not None:
        # Create an experiment_pb2.Experiment object from the input file.
        assert os.path.exists(config_file), "Config file not found at {}".format(config_file)
        logger.info("Loading experiment spec at {}".format(config_file))
        # The spec file in the config path has to be complete.
        # Default spec is not merged into es.
        es = load_experiment_spec(config_file, merge_from_default=False)
    else:
        logger.info("Loading the default experiment spec file.")
        es = load_experiment_spec()

    # Extract the training config.
    train_config = es.train_config
    model_config = es.model_config

    # Define data dimensions.
    image_shape = model_config.input_image_size.split(',')
    n_channel = int(image_shape[0])
    image_height = int(image_shape[1])
    image_width = int(image_shape[2])
    assert n_channel in [1, 3], "Invalid input image dimension."
    assert image_height >= 16, "Image height should be greater than 15 pixels."
    assert image_width >= 16, "Image width should be greater than 15 pixels."

    img_mean = es.train_config.image_mean
    if n_channel == 3:
        if img_mean:
            assert all(c in img_mean for c in ['r', 'g', 'b']) , (
                "'r', 'g', 'b' should all be present in image_mean "
                "for images with 3 channels."
            )
            img_mean = [img_mean['b'], img_mean['g'], img_mean['r']]
        else:
            img_mean = [103.939, 116.779, 123.68]
    else:
        if img_mean:
            assert 'l' in img_mean, (
                "'l' should be present in image_mean for images "
                "with 1 channel."
            )
            img_mean = [img_mean['l']]
        else:
            img_mean = [117.3786]

    # Define path to dataset.
    train_data = train_config.train_dataset_path
    if use_validation_set:
        train_data = train_config.val_dataset_path
    batch_size = train_config.batch_size_per_gpu

    # Setting dataloader color_mode.
    color_mode = "rgb"
    if n_channel == 1:
        color_mode = "grayscale"

    preprocessing_func = partial(
        preprocess_input,
        data_format="channels_first",
        mode=train_config.preprocess_mode,
        color_mode=color_mode,
        img_mean=img_mean)

    # Initialize the data generator.
    logger.info("Setting up input generator.")
    train_datagen = ImageDataGenerator(preprocessing_function=preprocessing_func,
                                       horizontal_flip=False,
                                       featurewise_center=False)
    logger.debug("Setting up iterator.")
    train_iterator = train_datagen.flow_from_directory(train_data,
                                                       target_size=(image_height,
                                                                    image_width),
                                                       batch_size=batch_size,
                                                       class_mode='categorical',
                                                       color_mode=color_mode)

    # Total_num_samples.
    num_samples = train_iterator.n
    num_avail_batches = int(num_samples / batch_size)
    assert n_batches <= num_avail_batches, "n_batches <= num_available_batches, n_batches={}, " \
                                           "num_available_batches={}".format(n_batches,
                                                                             num_avail_batches)

    # Make the output directory.
    dir_path = os.path.dirname(output_path)
    if not os.path.exists(dir_path):
        logger.info("Output directory not found. Creating at {}".format(dir_path))
        os.makedirs(dir_path)

    # Write data per batch.
    if os.path.exists(output_path):
        raise ValueError("A previously generated tensorfile already exists in the output path. "
                         "Please delete this file before writing a new one.")

    tensorfile = TensorFile(os.path.join(output_path), 'w')

    # Setting tqdm iterator.
    tr = trange(n_batches, file=sys.stdout)
    tr.set_description("Writing calibration tensorfile")

    for _ in tr:
        image, _ = next(train_iterator)
        tensorfile.write(image)
    tensorfile.close()

    logger.info("Calibration tensorfile written.")


def main(cl_args=None):
    """Main function for the trt calibration samples."""
    args = parse_command_line(args=cl_args)
    dump_samples(args["output"],
                 args["experiment_spec"],
                 args["max_batches"],
                 args['verbose'],
                 args['use_validation_set'])


if __name__ == '__main__':
    main()
