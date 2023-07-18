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
"""FPE DataIO pipeline script which generates tfrecords."""

import argparse
import os
from yaml import load

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utilities.path_processing import mkdir_p
from nvidia_tao_tf1.cv.fpenet.dataio.generate_dataset import tfrecord_manager


def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
            parser mechanism
    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            prog='convert_datasets',
            description='Convert FpeNet ground truth jsons to tfrecords.'
        )
    parser.add_argument('-e', '--experiment_spec_file',
                        type=str,
                        required=True,
                        help='Config file with dataio inputs.')
    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        default=None,
        help='Path to a folder where experiment outputs will be created, or specify in spec file.')

    return parser


def parse_command_line(args=None):
    """Simple function to parse command line arguments.

    Args:
        args (list): List of strings used as command line arguments.

    Returns:
        args_parsed: Parsed arguments.
    """

    parser = build_command_line_parser()
    args_parsed = parser.parse_args(args)

    return args_parsed


def main(cl_args=None):
    '''Main function to parse use arguments and call tfrecord manager.'''

    try:
        args = parse_command_line(cl_args)

        results_dir = args.results_dir

        if results_dir:
            mkdir_p(results_dir)

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
            message="Starting dataset convert."
        )

        config_path = args.experiment_spec_file
        with open(config_path, 'r') as f:
            args = load(f)

        tfrecord_manager(args)
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


if __name__ == '__main__':
    main()
