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
"""Modulus export application.

This module includes APIs to export a Keras model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import logging.config
import os
import sys
import tempfile
import time

import h5py
import keras
import numpy as np

from nvidia_tao_tf1.core.export._quantized import (
    check_for_quantized_layers,
    process_quantized_layers,
)
from nvidia_tao_tf1.core.export._tensorrt import keras_to_tensorrt
from nvidia_tao_tf1.core.export._uff import _reload_model_for_inference, keras_to_uff
from nvidia_tao_tf1.core.export.caffe import keras_to_caffe
from nvidia_tao_tf1.core.export.data import TensorFile
from nvidia_tao_tf1.core.utils.path_utils import expand_path
from third_party.keras.tensorflow_backend import limit_tensorflow_GPU_mem

if sys.version_info >= (3, 0):
    from nvidia_tao_tf1.core.export._onnx import (  # pylint: disable=C0412
        keras_to_onnx,
        validate_onnx_inference,
    )


"""Root logger for export app."""
logger = logging.getLogger(__name__)


def get_input_dims(tensor_filename):
    """Return sample input dimensions (excluding batch dimension)."""
    with TensorFile(tensor_filename, "r") as data_file:
        batch = data_file.read()
        input_dims = np.array(batch).shape[1:]
    return input_dims


def get_model_input_dtype(keras_hdf5_file):
    """Return input data type of a Keras model."""
    with h5py.File(keras_hdf5_file, mode="r") as f:
        model_config = f.attrs.get("model_config")
        model_config = json.loads(model_config.decode("utf-8"))

    input_layer_name = model_config["config"]["input_layers"][0][0]
    layers = model_config["config"]["layers"]
    input_layer = next(layer for layer in layers if layer["name"] == input_layer_name)
    data_type = str(input_layer["config"]["dtype"])
    if not data_type:
        raise RuntimeError(
            "Missing input layer data type in {}".format(keras_hdf5_file)
        )
    return data_type


def export_app(args):
    """Wrapper around export APIs.

    Args:
        args (dict): command-line arguments.
    """
    # Limit TensorFlow GPU memory usage.
    limit_tensorflow_GPU_mem(gpu_fraction=0.5)

    start_time = time.time()

    input_filename = args["input_file"]
    output_filename = args["output_file"]
    output_format = args["format"]
    input_dims = args["input_dims"]
    output_node_names = args["outputs"]
    max_workspace_size = args["max_workspace_size"]
    max_batch_size = args["max_batch_size"]
    data_type = args["data_type"]
    data_filename = args["data_file"]
    calibration_cache_filename = args["cal_cache"]
    batch_size = args["batch_size"]
    batches = args["batches"]
    fp32_layer_names = args["fp32_layer_names"]
    fp16_layer_names = args["fp16_layer_names"]
    parser = args["parser"]
    random_data = args["random_data"]
    verbose = args["verbose"]
    custom_objects = args.get("custom_objects")

    # Create list of exclude layers from command-line, if provided.
    if fp32_layer_names is not None:
        fp32_layer_names = fp32_layer_names.split(",")
    if fp16_layer_names is not None:
        fp16_layer_names = fp16_layer_names.split(",")

    if output_filename and "/" in output_filename:
        dirname = os.path.dirname(output_filename)
        if not os.path.exists(expand_path(dirname)):
            os.makedirs(expand_path(dirname))

    # Set up logging.
    verbosity = "DEBUG" if verbose else "INFO"

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=verbosity
    )

    logger.info("Loading model from %s", input_filename)

    # Configure backend floatx according to the model input layer.
    model_input_dtype = get_model_input_dtype(input_filename)
    keras.backend.set_floatx(model_input_dtype)

    keras.backend.set_learning_phase(0)

    # Load model from disk.
    model = keras.models.load_model(
        input_filename, compile=False, custom_objects=custom_objects
    )

    tensor_scale_dict = None
    if check_for_quantized_layers(model):
        assert data_type != "int8", (
            "Models with QuantizedConv2D layer are mixed-precision models."
            " Set data_type to fp32 or fp16 for non-quantized layers."
            " QuantizedConv2D layers will be handled automatically."
        )
        if calibration_cache_filename == "cal.bin":
            calibration_cache_filename = input_filename + ".cal.bin"
        calib_json_file = calibration_cache_filename + ".json"
        model, tensor_scale_dict = process_quantized_layers(
            model, output_format, calibration_cache_filename, calib_json_file
        )

    # Output node names may be explicitly specified if there are
    # more than one output node. Otherwise, the output node will
    # default to the last layer in the Keras model.
    if output_node_names is not None:
        output_node_names = output_node_names.split(",")

    input_shapes = []

    if output_format == "caffe":
        if output_filename is None:
            output_filename = input_filename

        prototxt_filename = "%s.%s" % (output_filename, "prototxt")
        model_filename = "%s.%s" % (output_filename, "caffemodel")

        in_tensor_name, out_tensor_names = keras_to_caffe(
            model,
            prototxt_filename,
            model_filename,
            output_node_names=output_node_names,
        )

        logger.info("Exported model definition was saved into %s", prototxt_filename)
        logger.info("Exported model weights were saved into %s", model_filename)
    elif output_format == "uff":
        if output_filename is None:
            output_filename = "%s.%s" % (input_filename, "uff")

        in_tensor_name, out_tensor_names, input_shapes = keras_to_uff(
            model,
            output_filename,
            output_node_names=output_node_names,
            custom_objects=custom_objects,
        )

        logger.info("Exported model was saved into %s", output_filename)
    elif output_format == "onnx":
        if sys.version_info < (3, 0):
            raise ValueError(
                "Exporting to onnx format is only supported under Python 3."
            )
        if output_node_names:
            raise ValueError(
                "Only exporting the entire keras model -> onnx is supported. Can't select"
                "custom output layers"
            )
        if output_filename is None:
            output_filename = "%s.%s" % (input_filename, "onnx")

        model = _reload_model_for_inference(model, custom_objects=custom_objects)
        in_tensor_name, out_tensor_names, input_shapes = keras_to_onnx(
            model,
            output_filename,
            custom_objects=custom_objects,
            target_opset=args["target_opset"],
        )

        success, error_str = validate_onnx_inference(
            keras_model=model, onnx_model_file=output_filename
        )
        if not success:
            logger.warning(
                "Validation of model with onnx-runtime failed. Error:{}".format(
                    error_str
                )
            )

        logger.info("Exported model was saved into %s", output_filename)
    elif output_format == "tensorrt":
        # Get input dimensions from data file if one was specified.
        if data_filename is not None:
            input_dims = get_input_dims(data_filename)
        else:
            # In the absence of a data file, get input dimensions from
            # the command line.
            if input_dims is None:
                raise ValueError(
                    "Input dimensions must be specified for the export to "
                    "TensorRT format."
                )
            input_dims = [int(item) for item in input_dims.split(",")]

        if random_data:
            os_handle, data_filename = tempfile.mkstemp(suffix=".tensorfile")
            os.close(os_handle)
            with TensorFile(data_filename, "w") as f:
                for _ in range(batches):
                    f.write(np.random.sample((batch_size,) + tuple(input_dims)))

        if output_filename is None:
            output_filename = "%s.%s" % (input_filename, "trt")

        if data_type == "int8" and data_filename is None:
            raise ValueError(
                "A calibration data file must be provided for INT8 export."
            )

        in_tensor_name, out_tensor_names, engine = keras_to_tensorrt(
            model,
            input_dims,
            output_node_names=output_node_names,
            dtype=data_type,
            max_workspace_size=max_workspace_size,
            max_batch_size=max_batch_size,
            calibration_data_filename=data_filename,
            calibration_cache_filename=calibration_cache_filename,
            calibration_n_batches=batches,
            calibration_batch_size=batch_size,
            fp32_layer_names=fp32_layer_names,
            fp16_layer_names=fp16_layer_names,
            parser=parser,
            tensor_scale_dict=tensor_scale_dict,
        )

        # Infer some test images if a data file was specified. This will
        # also print timing information if verbose mode was turned ON.
        if data_filename is not None:
            with TensorFile(data_filename, "r") as data_file:
                data_generator = (data_file.read()[:batch_size] for _ in range(batches))
                for _ in engine.infer_iterator(data_generator):
                    pass
            if random_data and os.path.exists(expand_path(data_filename)):
                os.remove(expand_path(data_filename))

        engine.save(output_filename)
        logger.info("Exported model was saved into %s", output_filename)
    else:
        raise ValueError("Unknown output format: %s" % output_format)

    logger.info("Input node: %s", in_tensor_name)
    logger.info("Output node(s): %s", out_tensor_names)

    logger.debug("Done after %s seconds", time.time() - start_time)

    return {
        "inputs": in_tensor_name,
        "outputs": out_tensor_names,
        "input_shapes": input_shapes,
    }


def add_parser_arguments(parser):
    """Adds the modulus export supported command line arguments to the given parser.

    Args:
        parser (argparse.ArgumentParser): The parser to add the arguemnts to.
    """
    # Positional arguments.
    parser.add_argument("input_file", help="Input file (Keras .h5 or TensorRT .uff).")

    # Optional arguments.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size to use for calibration and inference testing.",
    )

    parser.add_argument(
        "--batches",
        type=int,
        default=10,
        help="Number of batches to use for calibration and inference testing.",
    )

    parser.add_argument(
        "--cal_cache", default="cal.bin", help="Calibration cache file to write to."
    )

    parser.add_argument(
        "--data_type",
        type=str,
        default="fp32",
        help="Data type to use for TensorRT export.",
        choices=["fp32", "fp16", "int8"],
    )

    parser.add_argument(
        "--data_file",
        default=None,
        help="TensorFile of data to use for calibration and inference testing.",
    )

    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="uff",
        help="Output format",
        choices=["caffe", "onnx", "uff", "tensorrt"],
    )

    parser.add_argument(
        "--input_dims",
        type=str,
        default=None,
        help="Comma-separated list of input dimensions. This is "
        "needed for the export to TensorRT format. If a data file is "
        "provided the input dimensions will be inferred from the file.",
    )

    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=16,
        help="Maximum batch size of TensorRT engine in case of export to "
        "TensorRT format.",
    )

    parser.add_argument(
        "--max_workspace_size",
        type=int,
        default=(1 << 30),
        help="Maximum workspace size of TensorRT engine in case of export to "
        "TensorRT format.",
    )

    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default=None,
        help="Output file (defaults to $(input_filename).$(format)).",
    )

    parser.add_argument(
        "--outputs",
        type=str,
        default=None,
        help="Comma-separated list of output blob names.",
    )

    parser.add_argument(
        "--fp32_layer_names",
        type=str,
        default=None,
        help="Comma separated list of layers to be float32 precision.",
    )

    parser.add_argument(
        "--fp16_layer_names",
        type=str,
        default=None,
        help="Comma separated list of layers to be float16 precision.",
    )

    parser.add_argument(
        "--parser",
        type=str,
        default="uff",
        choices=["caffe", "onnx", "uff"],
        help="Parser to use for intermediate model representation "
        "in case of TensorRT export. Note, using onnx as parser is still under test,"
        " please be aware of the risk.",
    )

    parser.add_argument(
        "--random_data",
        action="store_true",
        help="Use random data during calibration and inference.",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose messages."
    )

    parser.add_argument(
        "--target_opset",
        type=int,
        default=None,
        help="target_opset for ONNX converter. default=<default used by"
        "the current keras2onnx package.",
    )


def main(args=None):
    """Export application.

    If MagLev was installed through ``pip`` then this application can be
    run from a shell. For example::

        $ maglev-export model.h5

    See command-line help for more information.

    Args:
        args (list): Arguments to parse.
    """
    if not args:
        args = sys.argv[1:]

    # Reduce TensorFlow verbosity.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser(
        description="Export a MagLev model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_parser_arguments(parser)

    args = vars(parser.parse_args(args))

    export_app(args)


if __name__ == "__main__":
    main()
