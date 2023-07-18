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
'''Export trained FPENet Keras model to UFF format.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utilities.path_processing import mkdir_p

from nvidia_tao_tf1.cv.fpenet.exporter.fpenet_exporter import FpeNetExporter


DEFAULT_MAX_WORKSPACE_SIZE = 1 * (1 << 30)
DEFAULT_MAX_BATCH_SIZE = 1


def build_command_line_parser(parser=None):
    '''Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
            parser mechanism.
    Returns:
        parser
    '''
    if parser is None:
        parser = argparse.ArgumentParser(prog='export', description='Encrypted UFF exporter.')

    parser.add_argument(
        '-m',
        '--model_filename',
        type=str,
        required=True,
        help='Absolute path to the model file to export.'
    )
    parser.add_argument(
        '-k',
        '--key',
        required=False,
        type=str,
        default="",
        help='Key to save or load a model.')
    parser.add_argument(
        '-o',
        '--out_file',
        required=False,
        type=str,
        default=None,
        help='Path to the output .etlt file.')
    parser.add_argument(
        '-t',
        '--target_opset',
        required=False,
        type=int,
        default=10,
        help='Target opset version to use for onnx conversion.')
    parser.add_argument(
        '--cal_data_file',
        default='',
        type=str,
        help='Tensorfile to run calibration for int8 optimization.')
    parser.add_argument(
        '--cal_image_dir',
        default='',
        type=str,
        help='Directory of images to run int8 calibration if data file is unavailable')
    parser.add_argument(
        '--data_type',
        type=str,
        default='fp32',
        help='Data type for the TensorRT export.',
        choices=['fp32', 'fp16', 'int8'])
    parser.add_argument(
        '-s',
        '--strict_type_constraints',
        action='store_true',
        default=False,
        help='Apply TensorRT strict_type_constraints or not for INT8 mode.')
    parser.add_argument(
        '--cal_cache_file',
        default='./cal.bin',
        type=str,
        help='Calibration cache file to write to.')
    parser.add_argument(
        '--batches',
        type=int,
        default=10,
        help='Number of batches to calibrate over.')
    parser.add_argument(
        '--max_workspace_size',
        type=int,
        default=DEFAULT_MAX_WORKSPACE_SIZE,
        help='Max size of workspace to be set for TensorRT engine builder.')
    parser.add_argument(
        '--max_batch_size',
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
        help='Max batch size for TensorRT engine builder.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Number of images per batch.')
    parser.add_argument(
        '--engine_file',
        type=str,
        default=None,
        help='Path to the exported TRT engine.')
    parser.add_argument(
        '--static_batch_size',
        type=int,
        default=-1,
        help='Set a static batch size for exported etlt model. \
        Default is -1(dynamic batch size).')
    parser.add_argument(
        '--opt_batch_size',
        type=int,
        default=1,
        help="Optimium batch size to use for int8 calibration.")
    parser.add_argument(
        '-d',
        '--input_dims',
        type=str,
        default='1,80,80',
        help='Input dims: channels_first(CHW) or channels_last (HWC).')
    parser.add_argument(
        '-b',
        '--backend',
        choices=['uff', 'tfonnx', 'onnx'],
        type=str,
        default='tfonnx',
        help='Model type to export to.')
    parser.add_argument(
        '-ll',
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level.'
    )
    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        default=None,
        help='Path to a folder where experiment outputs will be created, or specify in spec file.')

    return parser


def parse_command_line(args=None):
    '''Simple function to parse command line arguments.

    Args:
        args (list): List of strings used as command line arguments.
            If None, sys.argv is used.

    Returns:
        args_parsed: Parsed arguments.
    '''
    parser = build_command_line_parser()
    args_parsed = parser.parse_args(args)
    return args_parsed


def run_export(args=None):
    '''Wrapper to run export of tlt models.

    Args:
        args (dict): Dictionary of parsed arguments to run export.

    Returns:
        None.
    '''
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
        message="Starting export."
    )

    # Parsing command line arguments.
    model_name = args.model_filename
    key = args.key
    output_filename = args.out_file
    backend = args.backend
    input_dims = [int(i) for i in args.input_dims.split(',')]
    assert len(input_dims) == 3, "Input dims need to have three values."

    target_opset = args.target_opset
    log_level = args.log_level

    # Calibrator configuration.
    cal_cache_file = args.cal_cache_file
    cal_image_dir = args.cal_image_dir
    cal_data_file = args.cal_data_file
    batch_size = args.batch_size
    n_batches = args.batches
    data_type = args.data_type
    strict_type = args.strict_type_constraints
    engine_file_name = args.engine_file
    max_workspace_size = args.max_workspace_size
    max_batch_size = args.max_batch_size
    static_batch_size = args.static_batch_size
    opt_batch_size = args.opt_batch_size

    save_engine = False
    if engine_file_name is not None:
        save_engine = True
    # Build logger file.
    logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    logger.warning('Please verify the input dimension and input name before using this code!')

    # Set default output filename if the filename
    # isn't provided over the command line.
    output_extension = backend
    if backend in ["onnx", "tfonnx"]:
        output_extension = "onnx"
    if output_filename is None:
        split_name = os.path.splitext(model_name)[0]
        output_filename = f"{split_name}.{output_extension}"
    if not output_filename.endswith(output_extension):
        output_filename = f"{output_filename}.{output_extension}"
    logger.info("Saving exported model to {}".format(output_filename))

    # Make an output directory if necessary.
    output_root = os.path.dirname(os.path.realpath(output_filename))
    if not os.path.exists(output_root):
        os.makedirs(output_root)


    # Build exporter instance
    exporter = FpeNetExporter(model_name,
                              key,
                              backend=backend,
                              data_type=data_type,
                              strict_type=strict_type)

    # Export the model to etlt file and build the TRT engine.
    exporter.export(input_dims,
                    output_filename,
                    backend,
                    data_file_name=cal_data_file,
                    calibration_cache=os.path.realpath(cal_cache_file),
                    n_batches=n_batches,
                    batch_size=batch_size,
                    target_opset=target_opset,
                    save_engine=save_engine,
                    engine_file_name=engine_file_name,
                    calibration_images_dir=cal_image_dir,
                    max_batch_size=max_batch_size,
                    static_batch_size=static_batch_size,
                    opt_batch_size=opt_batch_size,
                    max_workspace_size=max_workspace_size)

    logger.info('Model exported at : %s' % output_filename)


def main(cl_args=None):
    """Run exporting."""
    try:
        args = parse_command_line(cl_args)
        run_export(args)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Export finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Export was interrupted",
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
