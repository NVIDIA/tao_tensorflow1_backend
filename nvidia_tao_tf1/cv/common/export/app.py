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

"""Script to export a trained TLT model to an ETLT file for deployment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime as dt
import logging
import os

import keras  # noqa pylint: disable=F401, W0611

import nvidia_tao_tf1.cv.common.logging.logging as status_logging

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKSPACE_SIZE = 2 * (1 << 30)
DEFAULT_MAX_BATCH_SIZE = 1
DEFAULT_OPT_BATCH_SIZE = 1
DEFAULT_MIN_BATCH_SIZE = 1


def build_command_line_parser(parser=None):
    """Simple function to parse arguments."""
    if parser is None:
        parser = argparse.ArgumentParser(description='Export a TLT model.')
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
                        required=False,
                        default="")
    parser.add_argument("-o",
                        "--output_file",
                        type=str,
                        default=None,
                        help="Output file. Defaults to $(input_filename).$(backend)")
    parser.add_argument("--force_ptq",
                        action="store_true",
                        default=False,
                        # help="Flag to force post training quantization for QAT models.")
                        help=argparse.SUPPRESS)
    # Int8 calibration arguments.
    parser.add_argument("--cal_data_file",
                        default="",
                        type=str,
                        # help="Tensorfile to run calibration for int8 optimization.")
                        help=argparse.SUPPRESS)
    parser.add_argument("--cal_image_dir",
                        default="",
                        type=str,
                        # help="Directory of images to run int8 calibration if "
                        #              "data file is unavailable")
                        help=argparse.SUPPRESS)
    parser.add_argument("--cal_json_file",
                        default="",
                        type=str,
                        help="Dictionary containing tensor scale for QAT models.")
    parser.add_argument("--data_type",
                        type=str,
                        default="fp32",
                        help=argparse.SUPPRESS,
                        # help="Data type for the TensorRT export.",
                        choices=["fp32", "fp16", "int8"])
    parser.add_argument("-s",
                        "--strict_type_constraints",
                        action="store_true",
                        default=False,
                        # help="Apply TensorRT strict_type_constraints or not for INT8 mode.")
                        help=argparse.SUPPRESS)
    parser.add_argument("--gen_ds_config",
                        action="store_true",
                        default=False,
                        help="Generate a template DeepStream related configuration elements. "
                             "This config file is NOT a complete configuration file and requires "
                             "the user to update the sample config files in DeepStream with the "
                             "parameters generated from here.")
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
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        # help="Number of images per batch.")
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
    parser.add_argument("--onnx_route",
                        type=str,
                        default="keras2onnx",
                        help=argparse.SUPPRESS)
    parser.add_argument("-e",
                        "--experiment_spec",
                        type=str,
                        default=None,
                        help="Path to the experiment spec file.")
    parser.add_argument("--engine_file",
                        type=str,
                        default=None,
                        # help="Path to the exported TRT engine.")
                        help=argparse.SUPPRESS)
    parser.add_argument("--static_batch_size",
                        type=int,
                        default=-1,
                        help=(
                            "Set a static batch size for exported etlt model. "
                            "Default is -1(dynamic batch size)."
                            "This option is only relevant for ONNX based model."
                        ))
    parser.add_argument("--target_opset",
                        type=int,
                        default=12,
                        help="Target opset for ONNX models.")
    parser.add_argument("--results_dir",
                        type=str,
                        default=None,
                        help="Path to the files where the logs are stored.")
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        default=False,
                        help="Verbosity of the logger.")
    return parser


def parse_command_line(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser()
    return vars(parser.parse_known_args(args)[0])


def run_export(Exporter, args, backend="uff"):
    """Wrapper to run export of tlt models.

    Args:
        Exporter(object): The exporter class instance.
        args (dict): Dictionary of parsed arguments to run export.
        backend(str): Exported model backend, either 'uff' or 'onnx'.

    Returns:
        No explicit returns.
    """
    # Parsing command line arguments.
    model_path = args['model']
    key = args['key']
    # Calibrator configuration.
    cal_cache_file = args['cal_cache_file']
    cal_image_dir = args['cal_image_dir']
    cal_data_file = args['cal_data_file']
    batch_size = args['batch_size']
    n_batches = args['batches']
    data_type = args['data_type']
    strict_type = args['strict_type_constraints']
    output_file = args['output_file']
    experiment_spec = args['experiment_spec']
    engine_file_name = args['engine_file']
    max_workspace_size = args["max_workspace_size"]
    max_batch_size = args["max_batch_size"]
    static_batch_size = args["static_batch_size"]
    target_opset = args["target_opset"]
    force_ptq = args["force_ptq"]
    gen_ds_config = args["gen_ds_config"]
    min_batch_size = args["min_batch_size"]
    opt_batch_size = args["opt_batch_size"]
    cal_json_file = args.get("cal_json_file", None)
    # This parameter is only relevant for classification.
    classmap_file = args.get("classmap_json", None)
    # Status logger for the UI. By default this will be populated in /workspace/logs.
    results_dir = args.get("results_dir", None)
    onnx_route = args.get("onnx_route", "keras2onnx")

    # Add warning if static_batch_size != -1, we will override whatever you have,
    # that batch size will be used for calibration also
    # and max_batch_size won't matter
    if static_batch_size != -1:
        logger.warning("If you set static batch size for your ONNX, "
                       "the calibration batch size will also be the "
                       "static batch size you provided.")

    if results_dir is not None:
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
    save_engine = False
    if engine_file_name is not None:
        save_engine = True

    log_level = "INFO"
    if args['verbose']:
        log_level = "DEBUG"

    # Configure the logger.
    logging.basicConfig(
        format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
        level=log_level)

    # Set default output filename if the filename
    # isn't provided over the command line.
    if output_file is None:
        split_name = os.path.splitext(model_path)[0]
        output_file = f"{split_name}.{backend}"

    if not (backend in output_file):
        output_file = f"{output_file}.{backend}"

    logger.info("Saving exported model to {}".format(output_file))

    # Warn the user if an exported file already exists.
    assert not os.path.exists(output_file), "Default output file {} already "\
        "exists".format(output_file)

    # Make an output directory if necessary.
    output_root = os.path.dirname(os.path.realpath(output_file))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Build exporter instance
    status_logger.write(message="Building exporter object.")
    exporter = Exporter(model_path, key,
                        backend=backend,
                        experiment_spec_path=experiment_spec,
                        data_type=data_type,
                        strict_type=strict_type,
                        classmap_file=classmap_file,
                        target_opset=target_opset,
                        onnx_route=onnx_route)
    exporter.set_session()
    exporter.set_keras_backend_dtype()
    # Export the model to etlt file and build the TRT engine.
    status_logger.write(message="Exporting the model.")
    exporter.export(output_file,
                    backend,
                    data_file_name=cal_data_file,
                    calibration_cache=os.path.realpath(cal_cache_file),
                    n_batches=n_batches,
                    batch_size=batch_size,
                    save_engine=save_engine,
                    engine_file_name=engine_file_name,
                    calibration_images_dir=cal_image_dir,
                    calib_json_file=cal_json_file,
                    max_batch_size=max_batch_size,
                    min_batch_size=min_batch_size,
                    opt_batch_size=opt_batch_size,
                    static_batch_size=static_batch_size,
                    max_workspace_size=max_workspace_size,
                    force_ptq=force_ptq,
                    gen_ds_config=gen_ds_config)


def launch_export(Exporter, args=None, backend="uff"):
    """CLI wrapper to run export.

    This function should be included inside package scripts/export.py

    # import build_command_line_parser as this is needed by entrypoint
    from nvidia_tao_tf1.cv.common.export.app import build_command_line_parser  # noqa pylint: disable=W0611
    from nvidia_tao_tf1.cv.common.export.app import launch_export
    from nvidia_tao_tf1.cv.X.export.X_exporter import XExporter as Exporter

    if __name__ == "__main__":
        launch_export(Exporter)
    """

    args = parse_command_line(args)
    run_export(Exporter, args, backend)


def main():
    """Raise deprecation warning."""
    raise DeprecationWarning(
        "This command has been deprecated in this version of TLT. "
        "Please run \n <model> export <cli_args>"
    )


if __name__ == "__main__":
    main()
