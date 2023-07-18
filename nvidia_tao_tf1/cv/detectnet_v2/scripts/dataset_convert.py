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

"""Command line interface for converting detection datasets to TFRecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os

from google.protobuf.text_format import Merge as merge_text_proto

import tensorflow as tf

from nvidia_tao_tf1.core.utils.path_utils import expand_path
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.detectnet_v2.dataio.build_converter import build_converter
import nvidia_tao_tf1.cv.detectnet_v2.proto.dataset_export_config_pb2 as dataset_export_config_pb2

logger = logging.getLogger(__name__)


def build_command_line_parser(parser=None):
    """Build command line parser for dataset_convert."""
    if parser is None:
        parser = argparse.ArgumentParser(
            prog='dataset_converter',
            description='Convert object detection datasets to TFRecords.'
        )
    parser.add_argument(
        '-d',
        '--dataset_export_spec',
        required=True,
        help='Path to the detection dataset spec containing config for exporting .tfrecords.')
    parser.add_argument(
        '-o',
        '--output_filename',
        required=True,
        help='Output file name.')
    parser.add_argument(
        '-f',
        '--validation_fold',
        type=int,
        default=None,
        help='Indicate the validation fold in 0-based indexing. \
            This is required when modifying the training set but otherwise optional.')
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help="Flag to get detailed logs during the conversion process."
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        type=str,
        default=None,
        help="Path to the results directory"
    )
    return parser


def parse_command_line_args(cl_args=None):
    """Parse sys.argv arguments from commandline.

    Args:
        cl_args: List of command line arguments.

    Returns:
        args: list of parsed arguments.
    """
    parser = build_command_line_parser()
    args = parser.parse_args(cl_args)
    return args


def main(args=None):
    """
    Convert an object detection dataset to TFRecords.

    Args:
        args(list): list of arguments to be parsed if called from another module.
    """
    args = parse_command_line_args(cl_args=args)

    verbosity = 'INFO'
    if args.verbose:
        verbosity = 'DEBUG'
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=verbosity)

    # Defining the results directory.
    if args.results_dir is not None:
        results_dir = expand_path(args.results_dir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        status_file = os.path.join(results_dir, "status.json")
        status_logging.set_status_logger(
            status_logging.StatusLogger(
                filename=status_file,
                is_master=True,
                verbosity=logger.getEffectiveLevel(),
                append=False
            )
        )
        status_logging.get_status_logger().write(
            data=None,
            message="Starting Object Detection Dataset Convert.",
            status_level=status_logging.Status.STARTED
        )

    # Load config from the proto file.
    dataset_export_config = dataset_export_config_pb2.DatasetExportConfig()
    with open(expand_path(args.dataset_export_spec), "r") as f:
        merge_text_proto(f.read(), dataset_export_config)

    converter = build_converter(dataset_export_config, args.output_filename, args.validation_fold)
    converter.convert()


if __name__ == '__main__':
    try:
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
        if type(e) == tf.errors.ResourceExhaustedError:
            logger = logging.getLogger(__name__)
            logger.error(
                "Ran out of GPU memory, please lower the batch size, use a smaller input "
                "resolution, or use a smaller backbone."
            )
            status_logging.get_status_logger().write(
                message="Ran out of GPU memory, please lower the batch size, use a smaller input "
                        "resolution, or use a smaller backbone.",
                verbosity_level=status_logging.Verbosity.INFO,
                status_level=status_logging.Status.FAILURE
            )
            exit(1)
        else:
            # throw out the error as-is if they are not OOM error
            status_logging.get_status_logger().write(
                message=str(e),
                status_level=status_logging.Status.FAILURE
            )
            raise e
