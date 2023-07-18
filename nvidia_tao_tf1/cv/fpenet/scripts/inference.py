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
"""FpeNet inference script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import yaml

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utilities.path_processing import mkdir_p
from nvidia_tao_tf1.cv.fpenet.inferencer.fpenet_inferencer import FpeNetInferencer

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
        parser = argparse.ArgumentParser(prog='infer', description='Run FpeNet inference.')

    parser.add_argument("-e",
                        "--experiment_spec",
                        default=None,
                        type=str,
                        help="Path to inferencer spec file.",
                        required=True)
    parser.add_argument('-i',
                        '--input_data_json_path',
                        help='The json file with paths to input images and ground truth face box.',
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument('-m',
                        '--model_path',
                        help='The trained model path to infer images with.',
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument("-k",
                        "--key",
                        default="",
                        help="Key to load the model.",
                        type=str,
                        required=False)
    parser.add_argument('-o',
                        '--output_folder',
                        help='The directory to the output images and predictions.',
                        type=str,
                        required=True,
                        default=None)
    parser.add_argument('-r',
                        '--image_root_path',
                        help='parent directory (if any) for the image paths in json.',
                        type=str,
                        required=False,
                        default='')

    return parser


def parse_command_line_args(cl_args=None):
    """Parser command line arguments to the inferencer.

    Args:
        cl_args(sys.argv[1:]): Arg from the command line.

    Returns:
        args: Parsed arguments using argparse.
    """
    parser = build_command_line_parser(parser=None)
    args = parser.parse_args(cl_args)
    return args


def main(args=None):
    """Wrapper function for running inference on a single image or collection of images.

    Args:
       Dictionary arguments containing parameters defined by command line parameters
    """
    arguments = parse_command_line_args(args)

    if arguments.output_folder:
        mkdir_p(arguments.output_folder)

    # Writing out status file for TAO.
    status_file = os.path.join(arguments.output_folder, "status.json")
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
        message="Starting inference."
    )

    # Load experiment spec.
    config_path = arguments.experiment_spec
    if not os.path.isfile(config_path):
        raise ValueError("Experiment spec file cannot be found.")
    with open(config_path, 'r') as yaml_file:
        spec = yaml.load(yaml_file.read())

    inferencer = FpeNetInferencer(experiment_spec=spec,
                                  data_path=arguments.input_data_json_path,
                                  output_folder=arguments.output_folder,
                                  model_path=arguments.model_path,
                                  image_root_path=arguments.image_root_path,
                                  key=arguments.key)
    # load pre-trained model
    inferencer.load_model()
    # load test data and run inference
    inferencer.infer_model()
    # save inference results
    inferencer.save_results()


if __name__ == "__main__":
    try:
        main()
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Inference finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Inference was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
