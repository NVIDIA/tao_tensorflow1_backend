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

"""Command line interface for converting pose datasets to TFRecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import yaml

from nvidia_tao_tf1.cv.bpnet.dataio.build_converter import build_converter
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
import nvidia_tao_tf1.cv.common.utilities.path_processing as io_utils


def build_command_line_parser(parser=None):
    """
    Convert a pose dataset to TFRecords.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
            parser mechanism.

    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(prog='dataset_convert',
                                         description='Convert pose datasets to TFRecords')
    parser.add_argument(
        '-d',
        '--dataset_spec',
        required=True,
        help='Path to the dataset spec containing config for exporting .tfrecords.')

    parser.add_argument(
        '-o',
        '--output_filename',
        required=True,
        help='Output file name.')

    parser.add_argument(
        '-m',
        '--mode',
        required=False,
        default='train',
        help='Converter mode: train/test.')

    parser.add_argument(
        '-p',
        '--num_partitions',
        type=int,
        required=False,
        default=1,
        help='Number of partitions (folds).')

    parser.add_argument(
        '-s',
        '--num_shards',
        type=int,
        required=False,
        default=0,
        help='Number of shards.')

    parser.add_argument(
        '--generate_masks',
        action='store_true',
        help='Generate and save masks of regions with unlabeled people - used for training.')

    parser.add_argument(
        '--check_files',
        action='store_true',
        help='Check if the files including images and masks exist in the given root data dir.')

    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        default=None,
        help='Path to a folder where experiment outputs will be created, or specify in spec file.')

    return parser


def parse_command_line_args(cl_args=None):
    """Parser command line arguments to the trainer.

    Args:
        cl_args (list): List of strings used as command line arguments.

    Returns:
        args_parsed: Parsed arguments.
    """
    parser = build_command_line_parser()
    args = parser.parse_args(cl_args)
    return args


def main(cl_args=None):
    """Generate tfrecords based on user arguments.

    Args:
        args(list): list of arguments to be parsed if called from another module.
    """
    args = parse_command_line_args(cl_args)

    results_dir = args.results_dir
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
        message="Starting Dataset convert."
    )

    # Load config file
    if args.dataset_spec.endswith(".json"):
        with open(args.dataset_spec, "r") as f:
            dataset_spec = json.load(f)
    elif args.dataset_spec.endswith(".yaml"):
        with open(args.dataset_spec, 'r') as f:
            dataset_spec = yaml.load(f.read())
    else:
        raise ValueError("Experiment spec file extension not supported.")

    converter = build_converter(
        dataset_spec,
        args.output_filename,
        mode=args.mode,
        num_partitions=args.num_partitions,
        num_shards=args.num_shards,
        generate_masks=args.generate_masks,
        check_if_images_and_masks_exist=args.check_files)

    converter.convert()


if __name__ == '__main__':
    try:
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Dataset convert finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Dataset convert was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
