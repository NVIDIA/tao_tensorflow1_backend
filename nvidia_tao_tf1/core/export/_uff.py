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
import os
import tempfile

import keras
from nvidia_tao_tf1.core.export._quantized import (
    check_for_quantized_layers,
    process_quantized_layers,
)

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.python.training import saver as saver_lib
import uff
from uff.model.utils import convert_to_str


"""Logger for UFF export APIs."""
logger = logging.getLogger(__name__)


def _reload_model_for_inference(model, custom_objects=None):
    """Reload a model specifically for doing inference.

    In order to export a model we need remove training-specific
    parts of the graph. For example, BatchNormalization layers
    may feature conditional branching to do training and inference
    alternately. This confused the UFF export tool.

    NOTE: the current Keras session is cleared in this function.
    Do not use this function during training.

    Args:
        model (Model): Keras model to reload in inference mode.
        custom_objects (dict): dictionary mapping names (strings) to custom
            classes or functions to be considered during deserialization for export.
    Returns:
        A model that can be used for inference only.
    """
    # Save model to a temp file so we can reload it later.
    os_handle, tmp_model_file_name = tempfile.mkstemp(suffix=".h5")
    os.close(os_handle)
    model.save(tmp_model_file_name)

    # Make sure Keras session is clean and tuned for inference.
    keras.backend.clear_session()
    keras.backend.set_learning_phase(0)

    @classmethod
    def apply_fused_padding(cls, tf_node, inputs, tf_nodes):
        tf_padding = convert_to_str(tf_node.attr["padding"].s)
        padding = None
        fields = {}
        if tf_padding == "SAME":
            fields["implicit_padding"] = "same"
        elif tf_padding == "VALID":
            fields["implicit_padding"] = None
            tf_lhs_node = tf_nodes[inputs[0]]
            if tf_lhs_node.op == "Pad":
                tf_padding_node = tf_nodes[tf_lhs_node.input[1]]
                p = cls.convert_tf2numpy_const_node(tf_padding_node)
                before, after = p[:, 0].tolist(), p[:, 1].tolist()
                if before == after:
                    padding = before
                    inputs[0] = tf_lhs_node.input[0]
                    if tf_nodes[inputs[0]].op == "Identity":
                        logger.info("Modulus patch identity layer in padding inputs.")
                        inputs[0] = tf_nodes[inputs[0]].input[0]
        else:
            raise ValueError("Padding mode %s not supported" % tf_padding)
        return inputs, padding, fields

    def compose_call(prev_call_method):
        def call(self, inputs, training=False):
            return prev_call_method(self, inputs, training)

        return call

    def dropout_patch_call(self, inputs, training=False):
        # Just return the input tensor. Keras will map this to ``keras.backend.identity``,
        # which the TensorRT 3.0 UFF parser supports.
        return inputs

    # Patch BatchNormalization and Dropout call methods so they don't create
    # the training part of the graph.
    prev_batchnorm_call = keras.layers.normalization.BatchNormalization.call
    prev_dropout_call = keras.layers.Dropout.call

    logger.debug("Patching keras BatchNormalization...")
    keras.layers.normalization.BatchNormalization.call = compose_call(
        prev_batchnorm_call
    )

    logger.debug("Patching keras Dropout...")
    keras.layers.Dropout.call = dropout_patch_call

    logger.debug("Patching UFF TensorFlow converter apply_fused_padding...")
    uff.converters.tensorflow.converter.TensorFlowToUFFConverter.apply_fused_padding = (
        apply_fused_padding
    )

    # Reload the model.
    model = keras.models.load_model(
        tmp_model_file_name, compile=False, custom_objects=custom_objects
    )

    # Unpatch Keras.
    logger.debug("Unpatching keras BatchNormalization layer...")
    keras.layers.normalization.BatchNormalization.call = prev_batchnorm_call

    logger.debug("Unpatching keras Dropout layer...")
    keras.layers.Dropout.call = prev_dropout_call

    # Delete temp file.
    os.remove(tmp_model_file_name)

    return model


def keras_to_pb(model, output_filename, output_node_names, custom_objects=None):
    """Export a Keras model to Protobuf format.

    The Protobuf format is a TensorFlow-specific representation
    of the model.

    NOTE: the current Keras session is cleared in this function.
    Do not use this function during training.

    Args:
        model (Model): Keras model to export.
        output_filename (str): file to write exported model to.
        output_node_names (list of str): list of model output node names as
        returned by model.layers[some_idx].get_output_at(0).name.split(':')[0].
        If None, then the model output layers are used as output nodes.
        custom_objects (dict): dictionary mapping names (strings) to custom
        classes or functions to be considered during deserialization for export.
    Returns:
        tuple<in_tensor_name(s), out_tensor_name(s), in_tensor_shape(s)>:
        in_tensor_name(s): The name(s) of the input nodes. If there is only one name, it will be
                           returned as a single string, otherwise a list of strings.
        out_tensor_name(s): The name(s) of the output nodes. If there is only one name, it will be
                            returned as a single string, otherwise a list of strings.
        in_tensor_shape(s): The shape(s) of the input tensors for this network. If there is only
                            one input tensor, it will be returned as a single list<int>, otherwise
                            a list<list<int>>.
    """
    model = _reload_model_for_inference(model, custom_objects=custom_objects)

    layers_with_external_state_io = [
        layer for layer in model.layers if hasattr(layer, "is_stateful")
    ]

    def get_layer_name(layer):
        _layer_outputs = layer.get_output_at(0)
        if isinstance(_layer_outputs, list):
            return [lo.name.split(":")[0] for lo in _layer_outputs]
        return _layer_outputs.name.split(":")[0]

    # Get names of input and output nodes.
    in_tensors = model.inputs
    in_tensor_shape = keras.backend.int_shape(in_tensors[0])
    in_name = in_tensors[0].op.name

    if layers_with_external_state_io:
        in_name = [in_name]
        in_tensor_shape = [in_tensor_shape]
        for layer in layers_with_external_state_io:
            if layer.is_stateful:
                in_name.append(layer.state_input_name)
            else:
                # Add feature maps of past frames for stateless models
                in_name.extend(layer._past_feature_names)
            shape = layer.input_shape
            shape = shape if shape[0] is None or isinstance(shape[0], int) else shape[0]
            in_tensor_shape.append(shape)

    if output_node_names is None:
        output_node_names = [t.op.name for t in model.outputs]

        # Replace the sliced output node with original output layers. For example, an output node
        # named `sliced_output_cov/Sigmoid` will be replaced with `output_cov/Sigmoid`
        layer_output_names = [get_layer_name(layer) for layer in model.layers]
        original_output_names = []
        for name in output_node_names:
            # For each sliced output node, search its original node by name and use the original
            # node to replace the sliced output node.
            if name.startswith("sliced_output_"):
                original_output_name_prefix = name.split("/")[0][7:]
                original_output_names += [
                    output_name
                    for output_name in layer_output_names
                    if output_name.startswith(original_output_name_prefix)
                ]
            else:
                original_output_names.append(name)
        output_node_names = original_output_names

        # Add output node names for the recurrent layers,
        # to handle the state external to TRT model.
        for layer in layers_with_external_state_io:
            if layer.is_stateful:
                temporal_output_node_name = get_layer_name(layer)
            else:
                temporal_output_node_name = layer.get_input_at(0).name.split(":")[0]
            if temporal_output_node_name not in output_node_names:
                output_node_names.append(temporal_output_node_name)

    # Freeze model.
    sess = keras.backend.get_session()

    # TensorFlow freeze_graph expects a comma separated string of output node names.
    output_node_names_tf = ",".join(output_node_names)

    saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)

    # Save the checkpoint file to a temporary location.
    os_handle, tmp_ckpt_file_name = tempfile.mkstemp(suffix=".ckpt")
    os.close(os_handle)
    checkpoint_path = saver.save(sess, tmp_ckpt_file_name)
    graph_io.write_graph(sess.graph, ".", output_filename)
    freeze_graph.freeze_graph(
        input_graph=output_filename,
        input_saver="",
        input_binary=False,
        input_checkpoint=checkpoint_path,
        output_node_names=output_node_names_tf,
        restore_op_name="save/restore_all",
        filename_tensor_name="save/Const:0",
        output_graph=output_filename,
        clear_devices=False,
        initializer_nodes="",
    )

    # Clean up.
    os.remove(tmp_ckpt_file_name)

    return in_name, output_node_names, in_tensor_shape


def pb_to_uff(input_filename, output_filename, out_names, text=False, quiet=True):
    """Convert a TensorFlow model to UFF.

    The input model needs to be passed as a frozen Protobuf file.
    The export UFF model may be parsed and optimized by TensorRT.

    Args:
        input_filename (str): path to protobuf file.
        output_filename (str): file to write exported model to.
        out_names (list of str): list of the names of the output nodes.
        text (boolean): whether to save .pbtxt file.
        quiet (boolean): whether to enable quiet mode.
    """
    uff.from_tensorflow_frozen_model(
        input_filename,
        out_names,
        output_filename=output_filename,
        text=text,
        quiet=quiet,
    )


def keras_to_uff(model, output_filename, output_node_names=None, custom_objects=None):
    """Export a Keras model to UFF format.

    UFF stands for Universal Framework Format and is an NVIDIA
    TensorRT file format for storing a neural network's topology and
    weights.

    NOTE: the current Keras session is cleared in this function.
    Do not use this function during training.

    Args:
        model (Model): Keras model to export.
        output_filename (str): file to write exported model to.
        output_node_names (list of str): list of model output node names as
        returned by model.layers[some_idx].get_output_at(0).name.split(':')[0].
        If not provided, then the last layer is assumed to be the output node.
        custom_objects (dict): dictionary mapping names (strings) to custom
        classes or functions to be considered during deserialization for export.
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
    # First, convert model to a temporary TensorFlow Protobuf.
    if check_for_quantized_layers(model):
        calib_json = output_filename + ".json"
        model, _ = process_quantized_layers(model, "uff", calib_json=calib_json)

    os_handle, tmp_pb_file_name = tempfile.mkstemp(suffix=".pb")
    os.close(os_handle)
    in_tensor_name, out_tensor_names, in_tensor_shapes = keras_to_pb(
        model, tmp_pb_file_name, output_node_names, custom_objects=custom_objects
    )

    # Second, convert protobuf to UFF.
    pb_to_uff(tmp_pb_file_name, output_filename, out_tensor_names)

    # Clean up.
    os.remove(tmp_pb_file_name)

    # Return a string instead of a list if there is only one output node.
    if len(out_tensor_names) == 1:
        out_tensor_names = out_tensor_names[0]

    return in_tensor_name, out_tensor_names, in_tensor_shapes
