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
import json
import logging
import os

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.multitask_classification.export.mclassification_exporter import (
    MClassificationExporter
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKSPACE_SIZE = 2 * (1 << 30)
DEFAULT_MAX_BATCH_SIZE = 1


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
                        default="")
    parser.add_argument("-o",
                        "--output_file",
                        type=str,
                        default=None,
                        help="Output file (defaults to $(input_filename).etlt)")
    parser.add_argument("--force_ptq",
                        action="store_true",
                        default=False,
                        help=argparse.SUPPRESS)
    # Int8 calibration arguments.
    parser.add_argument("--cal_data_file",
                        default="",
                        type=str,
                        help=argparse.SUPPRESS)
    parser.add_argument("--cal_image_dir",
                        default="",
                        type=str,
                        help=argparse.SUPPRESS)
    parser.add_argument("--data_type",
                        type=str,
                        default="fp32",
                        help=argparse.SUPPRESS,
                        choices=["fp32", "fp16", "int8"])
    parser.add_argument("-s",
                        "--strict_type_constraints",
                        action="store_true",
                        default=False,
                        help=argparse.SUPPRESS)
    parser.add_argument('--cal_cache_file',
                        default='./cal.bin',
                        type=str,
                        help=argparse.SUPPRESS)
    parser.add_argument("--batches",
                        type=int,
                        default=10,
                        help=argparse.SUPPRESS)
    parser.add_argument("--max_workspace_size",
                        type=int,
                        default=DEFAULT_MAX_WORKSPACE_SIZE,
                        help=argparse.SUPPRESS)
    parser.add_argument("--max_batch_size",
                        type=int,
                        default=DEFAULT_MAX_BATCH_SIZE,
                        help=argparse.SUPPRESS)
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help=argparse.SUPPRESS)
    parser.add_argument("--backend",
                        type=str,
                        default="onnx",
                        help=argparse.SUPPRESS,
                        choices=["onnx", "uff"])
    parser.add_argument("-cm",
                        "--class_map",
                        type=str,
                        help="Path to the classmap JSON file.")
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
                        help=argparse.SUPPRESS)
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


def run_export(args):
    """Wrapper to run export of tlt models.

    Args:
        args (dict): Dictionary of parsed arguments to run export.

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
    engine_file_name = args['engine_file']
    max_workspace_size = args["max_workspace_size"]
    max_batch_size = args["max_batch_size"]
    force_ptq = args["force_ptq"]
    # Status logger for the UI.
    results_dir = args.get("results_dir", None)
    gen_ds_config = args["gen_ds_config"]
    backend = args["backend"]
    save_engine = False
    if engine_file_name is not None:
        save_engine = True

    log_level = "INFO"
    if args['verbose']:
        log_level = "DEBUG"

    # Status logger initialization
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

    # Configure the logger.
    logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
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

    output_tasks = json.load(open(args['class_map'], 'r'))['tasks']
    # Build exporter instance
    status_logger.write(message="Building exporter object.")
    exporter = MClassificationExporter(output_tasks, model_path, key,
                                       backend=backend,
                                       data_type=data_type,
                                       strict_type=strict_type)
    exporter.set_session()
    exporter.set_keras_backend_dtype()
    # Export the model to etlt file and build the TRT engine.
    status_logger.write(message="Exporting the model.")
    exporter.export(output_file, backend,
                    data_file_name=cal_data_file,
                    calibration_cache=os.path.realpath(cal_cache_file),
                    n_batches=n_batches,
                    batch_size=batch_size,
                    save_engine=save_engine,
                    engine_file_name=engine_file_name,
                    calibration_images_dir=cal_image_dir,
                    max_batch_size=max_batch_size,
                    max_workspace_size=max_workspace_size,
                    force_ptq=force_ptq,
                    gen_ds_config=gen_ds_config)
    status_logger.write(
        status_level=status_logging.Status.SUCCESS,
        message="Exporting finished.")


if __name__ == "__main__":
    try:
        args = parse_command_line()
        run_export(args)
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
