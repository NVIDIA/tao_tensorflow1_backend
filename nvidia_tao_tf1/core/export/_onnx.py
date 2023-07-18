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
"""Modulus export APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys

import keras
import numpy as np

from nvidia_tao_tf1.core.export._quantized import (
    check_for_quantized_layers,
    process_quantized_layers,
)
from nvidia_tao_tf1.core.export._uff import _reload_model_for_inference

import onnx

if sys.version_info >= (3, 0):
    import keras2onnx
    import onnxruntime as rt


"""Logger for ONNX export APIs."""
logger = logging.getLogger(__name__)


def keras_to_onnx(model, output_filename, custom_objects=None, target_opset=None):
    """Export a Keras model to ONNX format.

    Args:
        model (Model): Keras model to export.
        output_filename (str): file to write exported model to.
        custom_objects (dict): dictionary mapping names (strings) to custom
        classes or functions to be considered during deserialization for export.
        target_opset (int): Target opset version to use, default=<default opset for
        the current keras2onnx installation>
    Returns:
        tuple<in_tensor_name(s), out_tensor_name(s), in_tensor_shape(s)>:
        in_tensor_name(s): The name(s) of the input nodes. If there is only one name, it will be
                           returned as a single string, otherwise a list of strings.
        out_tensor_name(s): The name(s) of the output nodes. If there is only one name, it will be
                            returned as a single string, otherwise a list of strings.
        in_tensor_shape(s): The shape(s) of the input tensors for this network. If there is only
                            one input tensor, it will be returned as a single list<int>, otherwise
                            a list<list<int>>.

        These must be passed to the TensorRT optimization tool to identify input and output blobs.
    """
    if check_for_quantized_layers(model):
        calib_json = output_filename + ".json"
        model, _ = process_quantized_layers(model, "onnx", calib_json=calib_json)
        model = _reload_model_for_inference(model, custom_objects=custom_objects)

    onnx_model = keras2onnx.convert_keras(
        model,
        model.name,
        custom_op_conversions=custom_objects,
        target_opset=target_opset,
    )

    logger.debug("Model converted to ONNX, checking model validity with onnx.checker.")
    # onnx.checker.check_model(onnx_model)
    onnx.save_model(onnx_model, output_filename)

    out_name = []
    for keras_out in model.outputs:
        for i in range(len(onnx_model.graph.output)):
            name = onnx_model.graph.output[i].name
            if keras_out._op.name in name or name in keras_out._op.name:
                out_name.append(name)
                break
    out_name = out_name[0] if len(out_name) == 1 else out_name

    in_name = []
    in_shape = []

    # ONNX graph inputs contain inputs that are not keras input layers.
    for keras_in in model.inputs:
        for i in range(len(onnx_model.graph.input)):
            name = onnx_model.graph.input[i].name
            # The ONNX input layer names are named the same as in keras, but contain extra
            # information like "_01" at the end or vice versa.
            if keras_in._op.name in name or name in keras_in._op.name:
                in_name.append(name)
                in_shape.append(keras.backend.int_shape(keras_in))

    in_name = in_name[0] if len(in_name) == 1 else in_name
    in_shape = in_shape[0] if len(in_shape) else in_shape

    return in_name, out_name, in_shape


def validate_onnx_inference(
    keras_model: keras.models.Model, onnx_model_file: str, tolerance=1e-4
) -> (bool, str):
    """Validate onnx model with onnx runtime..

    Args:
        keras_model (Model): Loaded Keras model.
        onnx_model_file (str): ONNX model filepath.
    Returns:
    Tuple (success, error_str)
        success(bool): True for success, False for failure.
        error_str(str): String which describes the error in case of failure.
    """
    sess = rt.InferenceSession(onnx_model_file)
    ort_inputs = sess.get_inputs()
    ort_outputs = sess.get_outputs()

    for i, (dim_onnx, dim_keras) in enumerate(
        zip(ort_outputs[0].shape, keras_model.outputs[0].get_shape().as_list())
    ):
        if dim_onnx is not None and dim_keras is not None:
            if dim_onnx != dim_keras:
                return (
                    False,
                    "The {} dim of onnx runtime parsed model does not match"
                    "the keras model. Onnx runtime model dims:{},"
                    "keras model dims:{}".format(
                        len(ort_outputs[0].shape) - i,
                        ort_outputs[0].shape,
                        keras_model.outputs[0].get_shape().as_list(),
                    ),
                )

    # Sample inference run with onnxruntime to verify model validity.
    # If the model is defined with a fixed batch size, pick that as
    # the input batch_size of validating the model.
    # For whatever reason, if batch dimension is not specified in the model,
    # the onnx runtime returns it as a string `None` istead of a python
    # inbuilt None.

    if isinstance(ort_inputs[0].shape[0], int):
        test_batch_size = ort_inputs[0].shape[0]
    else:
        test_batch_size = 8
    model_inputs = [
        np.random.uniform(size=([test_batch_size] + ort_input.shape[1:])).astype(
            np.float32
        )
        for ort_input in ort_inputs
    ]

    pred_onnx = sess.run(
        [ort_output.name for ort_output in ort_outputs],
        {ort_input.name: model_inputs[i] for i, ort_input in enumerate(ort_inputs)},
    )

    if list(pred_onnx[0].shape) != ([test_batch_size] + ort_outputs[0].shape[1:]):
        return (
            False,
            "Onnx runtime prediction not of expected shape"
            "Expected shape:{}, onnx runtime prediction shape:{}".format(
                [test_batch_size] + ort_outputs[0].shape[1:], pred_onnx[0].shape
            ),
        )
    pred_keras = keras_model.predict(model_inputs, batch_size=test_batch_size)

    # Check keras and ort predictions are close enough.
    mse = sum(
        [
            np.square(pred_onnx[i] - pred_keras[i]).mean(axis=None)
            for i in range(len(pred_onnx))
        ]
    )
    if mse > 1e-4:
        return (
            False,
            "Onnx-runtime and keras model predictions differ"
            " by mean squared error {}.".format(mse),
        )

    return True, ""
