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
"""Export Keras model to etlt format."""

import argparse
import logging
import os

from nvidia_tao_tf1.cv.bpnet.exporter.bpnet_exporter import BpNetExporter
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
import nvidia_tao_tf1.cv.common.utilities.path_processing as io_utils

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKSPACE_SIZE = 2 * (1 << 30)
DEFAULT_MAX_BATCH_SIZE = 1


def build_command_line_parser(parser=None):
    """Simple function to parse arguments."""
    if parser is None:
        parser = argparse.ArgumentParser(description='Export a Bpnet TLT model.')
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
                        help="Output file (defaults to $(input_filename).etlt)")
    parser.add_argument("--force_ptq",
                        action="store_true",
                        default=False,
                        help="Flag to force post training quantization for QAT models.")
    # Int8 calibration arguments.
    parser.add_argument("--cal_data_file",
                        default="",
                        type=str,
                        help="Tensorfile to run calibration for int8 optimization.")
    parser.add_argument("--cal_image_dir",
                        default="",
                        type=str,
                        help="Directory of images to run int8 calibration if "
                             "data file is unavailable")
    parser.add_argument("--data_type",
                        type=str,
                        default="fp32",
                        help="Data type for the TensorRT export.",
                        choices=["fp32", "fp16", "int8"])
    parser.add_argument("-s",
                        "--strict_type_constraints",
                        action="store_true",
                        default=False,
                        help="Apply TensorRT strict_type_constraints or not for INT8 mode.")
    parser.add_argument('--cal_cache_file',
                        default='./cal.bin',
                        type=str,
                        help='Calibration cache file to write to.')
    parser.add_argument("--batches",
                        type=int,
                        default=10,
                        help="Number of batches to calibrate over.")
    parser.add_argument("--max_workspace_size",
                        type=int,
                        default=DEFAULT_MAX_WORKSPACE_SIZE,
                        help="Max size of workspace to be set for TensorRT engine builder.")
    parser.add_argument("--max_batch_size",
                        type=int,
                        default=DEFAULT_MAX_BATCH_SIZE,
                        help="Max batch size for TensorRT engine builder.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="Number of images per batch.")
    parser.add_argument("-e",
                        "--experiment_spec",
                        type=str,
                        default=None,
                        help="Path to the experiment spec file.")
    parser.add_argument("--engine_file",
                        type=str,
                        default=None,
                        help="Path to the exported TRT engine.")
    parser.add_argument("--static_batch_size",
                        type=int,
                        default=-1,
                        help="Set a static batch size for exported etlt model. \
                        Default is -1(dynamic batch size).")
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        default=False,
                        help="Verbosity of the logger.")
    parser.add_argument('-d',
                        '--input_dims',
                        type=str,
                        default='256,256,3',
                        help='Input dims: channels_first(CHW) or channels_last (HWC).')
    parser.add_argument('--sdk_compatible_model',
                        action='store_true',
                        help='Generate SDK (TLT CV Infer / DS) compatible model.')
    parser.add_argument('-u',
                        '--upsample_ratio',
                        type=int,
                        default=4,
                        help='[NMS][CustomLayers] Upsampling factor.')
    parser.add_argument('-i',
                        '--data_format',
                        choices=['channels_last', 'channels_first'],
                        type=str,
                        default='channels_last',
                        help='Channel Ordering, channels_first(NCHW) or channels_last (NHWC).')
    parser.add_argument('-t',
                        '--backend',
                        choices=['onnx', 'uff', 'tfonnx'],
                        type=str,
                        default='onnx',
                        help="Model type to export to.")
    parser.add_argument('--opt_batch_size',
                        type=int,
                        default=1,
                        help="Optimium batch size to use for int8 calibration.")
    parser.add_argument('-r',
                        '--results_dir',
                        type=str,
                        default=None,
                        help='Path to a folder where experiment outputs will be created, \
                        or specify in spec file.')
    return parser


def parse_command_line(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser()
    return vars(parser.parse_known_args(args)[0])


def run_export(Exporter, args):
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
    max_workspace_size = args['max_workspace_size']
    max_batch_size = args['max_batch_size']
    static_batch_size = args['static_batch_size']
    opt_batch_size = args['opt_batch_size']
    force_ptq = args['force_ptq']
    sdk_compatible_model = args['sdk_compatible_model']
    upsample_ratio = args['upsample_ratio']
    data_format = args['data_format']
    backend = args['backend']
    results_dir = args['results_dir']
    input_dims = [int(i) for i in args["input_dims"].split(',')]
    assert len(input_dims) == 3, "Input dims need to have three values."
    save_engine = False
    if engine_file_name is not None:
        save_engine = True

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
        message="Starting export."
    )

    log_level = "INFO"
    if args['verbose']:
        log_level = "DEBUG"

    # Configure the logger.
    logging.basicConfig(
        format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
        level=log_level
    )

    # Set default output filename if the filename
    # isn't provided over the command line.
    output_extension = backend
    if backend in ["onnx", "tfonnx"]:
        output_extension = "onnx"

    if output_file is None:
        split_name = os.path.splitext(model_path)[0]
        output_file = f"{split_name}.{output_extension}"

    if not output_file.endswith(output_extension):
        output_file = f"{output_file}.{output_extension}"
    logger.info("Saving exported model to {}".format(output_file))

    # Warn the user if an exported file already exists.
    assert not os.path.exists(output_file), "Default output file {} already "\
        "exists".format(output_file)

    # Make an output directory if necessary.
    output_root = os.path.dirname(os.path.realpath(output_file))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Build exporter instance
    exporter = Exporter(model_path, key,
                        backend=backend,
                        experiment_spec_path=experiment_spec,
                        data_type=data_type,
                        strict_type=strict_type,
                        data_format=data_format)

    # Export the model to etlt file and build the TRT engine.
    exporter.export(input_dims,
                    output_file,
                    backend,
                    data_file_name=cal_data_file,
                    calibration_cache=os.path.realpath(cal_cache_file),
                    n_batches=n_batches,
                    batch_size=batch_size,
                    save_engine=save_engine,
                    engine_file_name=engine_file_name,
                    calibration_images_dir=cal_image_dir,
                    max_batch_size=max_batch_size,
                    static_batch_size=static_batch_size,
                    max_workspace_size=max_workspace_size,
                    force_ptq=force_ptq,
                    sdk_compatible_model=sdk_compatible_model,
                    upsample_ratio=upsample_ratio,
                    opt_batch_size=opt_batch_size)


def main(cl_args=None):
    """Run exporting."""
    try:
        args = parse_command_line(cl_args)
        run_export(BpNetExporter, args)
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


if __name__ == "__main__":
    main()
