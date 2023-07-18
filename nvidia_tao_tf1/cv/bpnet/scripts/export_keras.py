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
"""Export Keras model to other formats."""

import argparse
import logging
import os
import keras
from keras import backend as K

from nvidia_tao_tf1.core.export import keras_to_caffe, keras_to_onnx, keras_to_uff
import nvidia_tao_tf1.cv.bpnet.utils.export_utils as export_utils
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.common.utilities.export_utils import convertKeras2TFONNX
import nvidia_tao_tf1.cv.common.utilities.path_processing as io_utils


def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
            parser mechanism.

    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(prog='export', description='Encrypted UFF exporter.')

    parser.add_argument(
                        '-m',
                        '--model_filename',
                        type=str,
                        required=True,
                        default=None,
                        help="Absolute path to Keras model file \
                        (could be .h5, .hdf5 format).")
    parser.add_argument('-o',
                        '--output_filename',
                        required=False,
                        type=str,
                        default=None,
                        help='Path to the output file (without extension).')
    parser.add_argument(
                        '-t',
                        '--export_type',
                        choices=['onnx', 'tfonnx', 'uff', 'caffe'],
                        type=str,
                        default='uff',
                        help="Model type to export to."
    )
    parser.add_argument(
                        '-ll',
                        '--log_level',
                        type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO',
                        help='Set logging level.'
    )
    parser.add_argument(
                        '--sdk_compatible_model',
                        action='store_true',
                        help='Generate SDK compatible model.'
    )
    parser.add_argument(
                        '-ur',
                        '--upsample_ratio',
                        type=int,
                        default=4,
                        help='[NMS][CustomLayers] Upsampling factor.'
    )
    parser.add_argument(
                        '-df',
                        '--data_format',
                        type=str,
                        default='channels_last',
                        help='Channel Ordering, channels_first(NCHW) or channels_last (NHWC).'
    )
    parser.add_argument(
                        '-s',
                        '--target_opset',
                        required=False,
                        type=int,
                        default=10,
                        help='Target opset version to use for onnx conversion.'
    )
    parser.add_argument(
                        '-r',
                        '--results_dir',
                        type=str,
                        default=None,
                        help='Path to a folder where experiment outputs will be created, \
                        or specify in spec file.')

    return parser


def parse_command_line(args=None):
    """Simple function to parse command line arguments.

    Args:
        args (list): List of strings used as command line arguments.
            If None, sys.argv is used.

    Returns:
        args_parsed: Parsed arguments.
    """
    parser = build_command_line_parser()
    args_parsed = parser.parse_args(args)
    return args_parsed


def main(cl_args=None):
    """Run exporting."""
    args_parsed = parse_command_line(cl_args)

    results_dir = args_parsed.results_dir

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

    model_name = args_parsed.model_filename
    if args_parsed.output_filename is None:
        output_filename_noext = model_name
    else:
        output_filename_noext = args_parsed.output_filename
    target_opset = args_parsed.target_opset

    # Build logger file.
    logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(args_parsed.log_level)

    # Set channels ordering in keras backend
    K.set_image_data_format(args_parsed.data_format)

    # Load Keras model from file.
    model = keras.models.load_model(model_name)
    logger.info('model summary:')
    model.summary()

    model, custom_objects = export_utils.update_model(
        model,
        sdk_compatible_model=args_parsed.sdk_compatible_model,
        upsample_ratio=args_parsed.upsample_ratio
    )

    # Export to UFF.
    if args_parsed.export_type == 'uff':
        output_filename = output_filename_noext + '.uff'
        _, out_tensor_name, _ = keras_to_uff(model,
                                             output_filename,
                                             None,
                                             custom_objects=custom_objects)
        logger.info('Output tensor names are: ' + ', '.join(out_tensor_name))

    # Export to Caffe.
    if args_parsed.export_type == 'caffe':
        output_filename = output_filename_noext + '.caffe'
        prototxt_filename = output_filename_noext + '.proto'
        _, out_tensor_name = keras_to_caffe(
            model,
            prototxt_filename,
            output_filename,
            output_node_names=None)
        logger.info('Output tensor names are: ' + ', '.join(out_tensor_name))

    # Export to  onnx
    if args_parsed.export_type == 'onnx':
        output_filename = output_filename_noext + '.onnx'

        (in_tensor, out_tensor, in_tensor_shape) = \
            keras_to_onnx(
                model,
                output_filename,
                custom_objects=custom_objects,
                target_opset=target_opset)
        logger.info('In: "%s" dimension of %s Out "%s"' % (in_tensor, in_tensor_shape, out_tensor))

    # Export through keras->tf->onnx path
    if args_parsed.export_type == 'tfonnx':
        # Create froxen graph as .pb file.
        convertKeras2TFONNX(model,
                            output_filename_noext,
                            output_node_names=None,
                            target_opset=target_opset,
                            custom_objects=custom_objects,
                            logger=logger)


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
