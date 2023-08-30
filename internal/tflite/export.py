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

import argparse
import json
import logging
import os
import sys
import tempfile

from nvidia_tao_tf1.core.export import keras_to_pb
from nvidia_tao_tf1.core.export._quantized import check_for_quantized_layers, process_quantized_layers
from nvidia_tao_tf1.cv.common.utils import CUSTOM_OBJS, model_io

import keras
from keras.utils import CustomObjectScope
import tensorflow as tf

logger = logging.getLogger(__name__)


def reset_keras(fn):
    """Simple function to define the keras decorator.
    
    This decorator clears any previously existing sessions
    and sets up a new session.
    """
    def _fn_wrapper(*args, **kwargs):
        """Clear the keras session."""
        keras.backend.clear_session()
        set_keras_session()
        keras.backend.set_learning_phase(0)
        return fn(*args, **kwargs)
    return _fn_wrapper


def set_keras_session():
    """Set the keras and Tensorflow sessions."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    keras.backend.set_session(tf.Session(config=config))


@reset_keras
def load_model(model_path: str, key=""):
    """Load the keras model.

    Args:
        model_path(str): Path to the model.
        key(str): The key to load the model.
    """
    model = model_io(
        model_path,
        enc_key=key
    )
    return model


def resolve_path(path_string: str):
    """Simple function to resolve paths.

    Args:
        path_string (str): Path to model string.
    """
    return os.path.abspath(os.path.expanduser(path_string))


def save_model(model, output_path: str):
    """Save the keras model.

    Args:
        model (keras.models.Model): Path to the keras model to be saved.
        output_path (str): Path to save the model.
    """
    with CustomObjectScope(CUSTOM_OBJS):
        model.save(resolve_path(output_path))


def extract_model_scales(model,
                         backend: str = "onnx"):
    """Remove QDQ and Quantized* layers and extract the scales.

    Args:
        model (keras.model.Model): Model to inspect and extract scales.
        backend (str): "onnx,uff" model backend.
    """
    model, tensor_scale_dict = process_quantized_layers(
        model, backend,
        calib_cache=None,
        calib_json=None)
    logger.info(
        "Extracting tensor scale: {tensor_scale_dict}".format(
        tensor_scale_dict=tensor_scale_dict
        )
    )
    logger.info("Extracting quantized scales")
    os_handle, tmp_keras_model = tempfile.mkstemp(suffix=".hdf5")
    os.close(os_handle)
    with CustomObjectScope(CUSTOM_OBJS):
        model.save(tmp_keras_model)
    new_model = load_model(tmp_keras_model)
    return new_model


def convert_to_pb(model, output_node_names=None):
    """Convert the model to graphdef protobuf.

    Args:
        model (keras.model.Model): Keras model object to serialize.
        output_node_names (dict): Name of the output nodes of the model.
    
    Returns:
        tmp_pb_file (str): Path to the protobuf file containing tf.graphDef.
        input_tensor_names (list): Names of the input tensors.
        output_tensor_names (list): Name of the output tensors.
    """
    os_handle, tmp_pb_file = tempfile.mkstemp(
        suffix=".pb"
    )
    os.close(os_handle)
    input_tensor_names, out_tensor_names, _ = keras_to_pb(
        model,
        tmp_pb_file,
        None,
        custom_objects=CUSTOM_OBJS
    )
    if output_node_names:
        out_tensor_names = output_node_names
    return tmp_pb_file, input_tensor_names, out_tensor_names


def parse_command_line(cl_args="None"):
    """Parse command line args."""
    parser = argparse.ArgumentParser(
        prog="export_tflite",
        description="Export keras models to tflite."
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="Path to a model file."
    )
    parser.add_argument(
        "--key",
        type=str,
        default="",
        help="Key to load the model."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Path to the output model file."
    )
    args = vars(parser.parse_args(cl_args))
    return args


def main(cl_args=None):
    """Model converter."""
    # Convert the model
    args = parse_command_line(cl_args=cl_args)
    input_model_file = args["model_file"]
    output_model_file = args["output_file"]
    key = args["key"]
    tensor_scale_dict = None
    if not output_model_file:
        output_model_file = f"{os.path.splitext(input_model_file)[0]}.tflite"

    model = load_model(
        input_model_file, key
    )
    quantized_model = check_for_quantized_layers(model)
    logger.info("Quantized model: {quantized_model}".format(quantized_model=quantized_model))
    if quantized_model:
        model, tensor_scale_dict = extract_model_scales(
            model, backend="onnx"
        )
        tensor_scale_file = os.path.join(
            os.path.dirname(output_model_file),
            "calib_scale.json"
        )
        with open(tensor_scale_file, "w") as scale_file:
            json.dump(
                tensor_scale_dict, scale_file, indent=4
            )
    graph_def_file, input_arrays, output_arrays = convert_to_pb(
        model
    )

    # Convert the model to TFLite.
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file, input_arrays, output_arrays
    )
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()
    with open(output_model_file, "wb") as tflite_file:
        model_size = tflite_file.write(tflite_model)
    print(
        f"Output tflite model of size {model_size} bytes "
        f"was written at {output_model_file}"
    )


if __name__ == "__main__":
    main(cl_args=sys.argv[1:])
