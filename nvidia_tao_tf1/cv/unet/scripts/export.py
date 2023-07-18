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

"""Script to export a trained UNet model to an ETLT file for deployment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime as dt
import logging
import os

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.unet.export.unet_exporter import UNetExporter as Exporter

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKSPACE_SIZE = 1 * (1 << 30)
DEFAULT_MAX_BATCH_SIZE = 1
DEFAULT_MIN_BATCH_SIZE = 1
DEFAULT_OPT_BATCH_SIZE = 1


def build_command_line_parser(parser=None):
    """Build a command line parser."""
    if parser is None:
        parser = argparse.ArgumentParser(description='Export a trained TLT model')

    parser.add_argument("-m",
                        "--model",
                        help="Path to the model file.",
                        type=str,
                        required=True,
                        default=None)
    parser.add_argument("-k",
                        "--key",
                        help="Key to load the model.",
                        type=str,
                        default="",
                        required=False)
    parser.add_argument("-e",
                        "--experiment_spec",
                        type=str,
                        default=None,
                        required=True,
                        help="Path to the experiment spec file.")
    parser.add_argument("-o",
                        "--output_file",
                        type=str,
                        default=None,
                        help="Output file (defaults to $(input_filename).onnx)")
    parser.add_argument("--data_type",
                        type=str,
                        default="fp32",
                        help="Data type for the TensorRT export.",
                        choices=["fp32", "fp16", "int8"])
    parser.add_argument("--max_workspace_size",
                        type=int,
                        default=DEFAULT_MAX_WORKSPACE_SIZE,
                        # help="Max size of workspace to be set for TensorRT engine builder.")
                        help=argparse.SUPPRESS)
    parser.add_argument("--max_batch_size",
                        type=int,
                        default=DEFAULT_MAX_BATCH_SIZE,
                        # help="Max batch size for TensorRT engine builder.")
                        help=argparse.SUPPRESS)
    parser.add_argument("--min_batch_size",
                        type=int,
                        default=DEFAULT_MIN_BATCH_SIZE,
                        # help="Min batch size for TensorRT engine builder.")
                        help=argparse.SUPPRESS)
    parser.add_argument("--opt_batch_size",
                        type=int,
                        default=DEFAULT_OPT_BATCH_SIZE,
                        # help="Opt batch size for TensorRT engine builder.")
                        help=argparse.SUPPRESS)
    parser.add_argument("--gen_ds_config",
                        action="store_true",
                        default=False,
                        help="Generate a template DeepStream related configuration elements. "
                             "This config file is NOT a complete configuration file and requires "
                             "the user to update the sample config files in DeepStream with the "
                             "parameters generated from here.")
    parser.add_argument("--engine_file",
                        type=str,
                        default=None,
                        # help="Path to the exported TRT engine.")
                        help=argparse.SUPPRESS)
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        default=False,
                        help="Verbosity of the logger.")
    parser.add_argument("-s",
                        "--strict_type_constraints",
                        action="store_true",
                        default=False,
                        # help="Apply TensorRT strict_type_constraints or not")
                        help=argparse.SUPPRESS)
    # Int8 calibration arguments.
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        # help="Number of images per batch for calibration.")
                        help=argparse.SUPPRESS)
    parser.add_argument("--cal_data_file",
                        default="",
                        type=str,
                        # help="Tensorfile to run calibration for int8 optimization.")
                        help=argparse.SUPPRESS)
    parser.add_argument("--cal_image_dir",
                        default="",
                        type=str,
                        # help="Directory of images to run int8 calibration if "
                        #      "data file is unavailable")
                        help=argparse.SUPPRESS)
    parser.add_argument("--cal_json_file",
                        default="",
                        type=str,
                        help="Dictionary containing tensor scale for QAT models.")
    parser.add_argument('--cal_cache_file',
                        default='./cal.bin',
                        type=str,
                        # help='Calibration cache file to write to.')
                        help=argparse.SUPPRESS)
    parser.add_argument("--batches",
                        type=int,
                        default=10,
                        # help="Number of batches to calibrate over.")
                        help=argparse.SUPPRESS)
    parser.add_argument("--results_dir",
                        type=str,
                        default=None,
                        help="Path to the files where the logs are stored.")
    parser.add_argument("--force_ptq",
                        action="store_true",
                        default=False,
                        # help="Flag to force post training quantization for QAT models.")
                        help=argparse.SUPPRESS)

    return parser


def parse_command_line(args=None):
    """Simple function to parse arguments."""
    parser = build_command_line_parser()
    args = vars(parser.parse_args(args))
    return args


def build_exporter(model_path, key,
                   experiment_spec="",
                   data_type="fp32",
                   strict_type=False):
    """Simple function to build exporter instance."""
    constructor_kwargs = {'model_path': model_path,
                          'key': key,
                          "experiment_spec_path": experiment_spec,
                          'data_type': data_type,
                          'strict_type': strict_type}
    return Exporter(**constructor_kwargs)


def main(cl_args=None):
    """CLI wrapper to run export.

    This function parses the command line interface for tlt-export, instantiates the respective
    exporter and serializes the trained model to an etlt file. The tools also runs optimization
    to the int8 backend.

    Args:
        cl_args(list): Arguments to parse.

    Returns:
        No explicit returns.
    """

    args = parse_command_line(args=cl_args)
    run_export(args)


def run_export(args):
    """Wrapper to run export of tlt models.

    Args:
        args (dict): Dictionary of parsed arguments to run export.

    Returns:
        No explicit returns.
    """
    results_dir = args["results_dir"]

    # Parsing command line arguments.
    model_path = args['model']
    key = args['key']
    data_type = args['data_type']
    output_file = args['output_file']
    experiment_spec = args['experiment_spec']
    engine_file_name = args['engine_file']
    max_workspace_size = args["max_workspace_size"]
    max_batch_size = args["max_batch_size"]
    strict_type = args['strict_type_constraints']
    cal_data_file = args["cal_data_file"]
    cal_image_dir = args["cal_image_dir"]
    cal_cache_file = args["cal_cache_file"]
    n_batches = args["batches"]
    batch_size = args["batch_size"]
    gen_ds_config = args["gen_ds_config"]
    min_batch_size = args["min_batch_size"]
    opt_batch_size = args["opt_batch_size"]
    force_ptq = args["force_ptq"]
    cal_json_file = args["cal_json_file"]

    save_engine = False
    if engine_file_name is not None:
        save_engine = True

    log_level = "INFO"
    if args['verbose']:
        log_level = "DEBUG"

    # Configure the logger.
    logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                        level=log_level)

    # Set default output filename if the filename
    # isn't provided over the command line.
    if output_file is None:
        split_name = model_path.replace(".tlt","")
        output_file = "{}.onnx".format(split_name)

    # Warn the user if an exported file already exists.
    assert not os.path.exists(output_file), "Default output file {} already "\
        "exists".format(output_file)

    # Make an output directory if necessary.
    output_root = os.path.dirname(os.path.realpath(output_file))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    if not results_dir:
        results_dir = output_root
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    timestamp = int(dt.timestamp(dt.now()))
    filename = "status.json"
    if results_dir == "/workspace/logs":
        filename = f"status_export_{timestamp}.json"
    status_file = os.path.join(results_dir, filename)
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            is_master=True
        )
    )
    status_logger = status_logging.get_status_logger()
    # Build exporter instance
    status_logger.write(message="Building exporter object.")
    exporter = build_exporter(model_path, key,
                              experiment_spec=experiment_spec,
                              data_type=data_type,
                              strict_type=strict_type)

    # Export the model to etlt file and build the TRT engine.
    status_logger.write(message="Exporting the model.")
    exporter.export(output_file_name=output_file,
                    backend="onnx",
                    save_engine=save_engine,
                    engine_file_name=engine_file_name,
                    max_batch_size=max_batch_size,
                    min_batch_size=min_batch_size,
                    opt_batch_size=opt_batch_size,
                    max_workspace_size=max_workspace_size,
                    data_file_name=cal_data_file,
                    calib_json_file=cal_json_file,
                    calibration_images_dir=cal_image_dir,
                    calibration_cache=cal_cache_file,
                    n_batches=n_batches,
                    batch_size=batch_size,
                    gen_ds_config=gen_ds_config,
                    force_ptq=force_ptq)

    status_logger.write(
        data=None,
        status_level=status_logging.Status.SUCCESS,
        message="Unet export job complete."
    )


if __name__ == "__main__":
    try:
        main()
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
