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
"""Process and export quantized models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tempfile

import keras
from keras.layers import Conv2D, Dense, DepthwiseConv2D, TimeDistributed
from keras.utils.generic_utils import CustomObjectScope

import tensorflow as tf

from nvidia_tao_tf1.core.models.templates.qdq_layer import QDQ
from nvidia_tao_tf1.core.models.templates.quantized_conv2d import QuantizedConv2D
from nvidia_tao_tf1.core.models.templates.quantized_dense import QuantizedDense
from nvidia_tao_tf1.core.models.templates.quantized_depthwiseconv2d import QuantizedDepthwiseConv2D
from nvidia_tao_tf1.cv.common.utils import CUSTOM_OBJS

QAT_LAYERS = [
    QuantizedConv2D,
    QuantizedDepthwiseConv2D,
    QDQ,
    QuantizedDense,
]


def collapse_pad_and_conv(tensor_scale_dict):
    """ZeroPadding is fused with its following conv2d/depthwiseconv2d, collapse them."""
    padding_nodes = []
    for k in tensor_scale_dict:
        if '/Pad' in k:
            # this is a ZeroPadding node
            padding_nodes.append(k)
    for n in padding_nodes:
        tensor_scale_dict.pop(n)


def collapse_flatten_and_prev(tensor_scale_dict):
    """Flatten node is no-op in UFF, collapse with its previous layer."""
    # get flatten node
    flatten_op = tf.get_default_graph().get_operation_by_name('time_distributed_flatten/Reshape')
    if flatten_op:
        # get flatten input tensor(QDQ)
        flatten_input_tensor = flatten_op.inputs[0]
        while '_qdq' in flatten_input_tensor.name:
            # get QDQ input tensor
            flatten_input_tensor = flatten_input_tensor.op.inputs[0]
        # get previous node name
        prev_node_name = flatten_input_tensor.op.inputs[0].op.name
        if prev_node_name and (prev_node_name in tensor_scale_dict):
            tensor_scale_dict.pop(prev_node_name)
            return
        if 'crop_and_resize_' in prev_node_name:
            plugin_name = 'roi_pooling_conv_1/CropAndResize_new'
            assert plugin_name in tensor_scale_dict, (
                "Expect plugin node: {} in tensor_scale_dict, but not found.".format(plugin_name)
            )
            tensor_scale_dict.pop(plugin_name)
            return


def process_flatten_name(tensor_name):
    """Strip Flatten TD reshape."""
    if re.match(r'time_distributed_flatten/Reshape_2', tensor_name):
        return tensor_name.replace('Reshape_2', 'Reshape_1', 1)
    return tensor_name


def process_plugins_name(tensor_name):
    """replace the node name with the corresponding plugins name."""
    if re.match(r'crop_and_resize_1/Reshape_1', tensor_name):
        plugin_name = 'roi_pooling_conv_1/CropAndResize_new'
        return plugin_name
    return tensor_name


def process_td_output_name(tensor_name, layer, up=False):
    """replace the output name of TD layer with its inner layer output name."""
    # if the input comes from a TD layer, we should use the name
    # of the inner layer of TD layer to align with pb
    if re.match(r'time_distributed_[0-9]+/Reshape_[0-9]+', tensor_name):
        if up:
            prev_layer = layer._inbound_nodes[0].inbound_layers[0]
        else:
            prev_layer = layer
        assert type(prev_layer) == TimeDistributed, type(prev_layer)
        # the TD inner layer .output attr is not set
        # so we have to find it out with TF APIs
        # get the node of the TD Reshape_1 op
        td_reshape_1 = tf.get_default_graph().get_operation_by_name(tensor_name.split(":")[0])
        # get this op's input tensor
        tensor_name_inner = td_reshape_1.inputs[0].name
        if re.match(r'time_distributed_[0-9]+/Reshape:.*$', tensor_name_inner):
            # this is a TD Dropout layer, get its input again
            tensor_name_inner = td_reshape_1.inputs[0].op.inputs[0].name
        tensor = td_reshape_1.inputs[0].op.inputs[0]
        # probably there are some QDQ layers in between
        while '_qdq' in tensor_name_inner:
            tensor = tensor.op.inputs[0]
            tensor_name_inner = tensor.name
        # can still get a TD layer, strip it
        if re.match(r'time_distributed_[0-9]+/Reshape_[0-9]+', tensor_name_inner):
            tensor = tensor.op.inputs[0]
            tensor_name_inner = tensor.name
        return tensor_name_inner
    return tensor_name


def check_for_quantized_layers(model):
    """Check Keras model for quantization layers."""
    for layer in model.layers:
        if type(layer) in QAT_LAYERS:
            return True
        if type(layer) == TimeDistributed:
            if type(layer.layer) in QAT_LAYERS:
                return True
    return False


def process_quantized_layers(model,
                             output_format,
                             create_session=False,
                             learning_phase=0):
    """Remove QDQ, replace the QuantizedConv2D with Conv2D and extract calibration cache."""
    network_dict = {"input_layers_of": {}, "new_output_tensor_of": {}}

    # Set the input layers of each layer.
    for layer in model.layers:
        if len(layer._inbound_nodes) > 1:
            raise AttributeError(
                "Layers with multiple inbound nodes are not supported."
            )
        inbound_node = layer._inbound_nodes[0]
        inbound_layers = [in_layer.name for in_layer in inbound_node.inbound_layers]
        if len(inbound_layers) > 0:
            network_dict["input_layers_of"].update({layer.name: inbound_layers})

    input_layers = [
        l for l in model.layers if len(l._inbound_nodes[0].inbound_layers) == 0
    ]
    assert len(input_layers) > 0, "No input layer was found."
    assert len(input_layers) == len(
        model.inputs
    ), "Number of input layers does not match number of input tensors."
    for layer in input_layers:
        input_tensor = layer._inbound_nodes[0].input_tensors[0]
        assert input_tensor in model.inputs, "Input tensor not found in model inputs."
        network_dict["new_output_tensor_of"].update({layer.name: input_tensor})

    qdq_scale_dict = {}
    for layer in model.layers:
        if type(layer) == QDQ:
            scaling_factor = layer.get_weights()
            scaling_factor = scaling_factor[0]
            prev_layer_name = network_dict["input_layers_of"][layer.name]
            assert (
                len(prev_layer_name) == 1
            ), "QDQ layer is expected to have only one input layer."
            qdq_scale_dict[prev_layer_name[0]] = scaling_factor
            for node in layer._outbound_nodes:
                layer_name = node.outbound_layer.name
                if type(node.outbound_layer) == QDQ:
                    raise AttributeError("Cascaded QDQ layers are not supported.")
                idx = network_dict["input_layers_of"][layer_name].index(layer.name)
                network_dict["input_layers_of"][layer_name][idx] = prev_layer_name[0]

    output_tensors = []
    tensor_scale_dict = {}

    for layer in model.layers:
        if layer.name not in network_dict["input_layers_of"]:
            # It's an input layer.
            if layer.name in qdq_scale_dict:
                tensor_name = layer.output.name
                # UFF exporter freezes the graph into a .pb file before exporting to UFF.
                # As a result, the ":0", ":1", ... which indicates the output index of
                # a TensorFlow OP in the output tensor name will be removed from the name
                # of the tensors. The ONNX exporter does not seem to be starting from
                # a frozen graph.
                if output_format != "onnx":
                    tensor_name = tensor_name.split(":")[0]
                tensor_scale_dict[tensor_name] = qdq_scale_dict[layer.name]
            continue
        if type(layer) == QDQ:
            continue

        # Determine input tensors.
        layer_input = [
            network_dict["new_output_tensor_of"][layer_aux]
            for layer_aux in network_dict["input_layers_of"][layer.name]
        ]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        if type(layer) == QuantizedConv2D:
            x = layer_input
            layer_config = layer.get_config()
            layer_config.pop("bitwidth")
            quantize_input = layer_config.pop("quantize")
            new_layer = Conv2D.from_config(layer_config)
            if quantize_input:
                if layer.use_bias:
                    kernels, biases, scaling_factor = layer.get_weights()
                else:
                    kernels, scaling_factor = layer.get_weights()
                assert (
                    scaling_factor.shape == ()
                ), "Unexpected shape for scaling factor parameter."
            else:
                if layer.use_bias:
                    kernels, biases = layer.get_weights()
                else:
                    kernels = layer.get_weights()[0]
            x = new_layer(x)
            if layer.use_bias:
                new_layer.set_weights([kernels, biases])
            else:
                new_layer.set_weights([kernels])
            if (
                quantize_input
                and type(layer._inbound_nodes[0].inbound_layers[0]) != QDQ
            ):
                tensor_name = process_td_output_name(layer.input.name, layer, up=True)
                if output_format != "onnx":
                    tensor_name = process_plugins_name(tensor_name)
                tensor_name = process_flatten_name(tensor_name)
                if output_format != "onnx":
                    tensor_name = tensor_name.split(":")[0]
                if tensor_name in tensor_scale_dict:
                    tensor_scale_dict[tensor_name] = max(
                        tensor_scale_dict[tensor_name], scaling_factor
                    )
                else:
                    tensor_scale_dict[tensor_name] = scaling_factor
        elif type(layer) == TimeDistributed and type(layer.layer) == QuantizedConv2D:
            x = layer_input
            layer_config = layer.layer.get_config()
            layer_config.pop("bitwidth")
            quantize_input = layer_config.pop("quantize")
            new_layer = Conv2D.from_config(layer_config)
            if quantize_input:
                if layer.layer.use_bias:
                    kernels, biases, scaling_factor = layer.layer.get_weights()
                else:
                    kernels, scaling_factor = layer.layer.get_weights()
                assert (
                    scaling_factor.shape == ()
                ), "Unexpected shape for scaling factor parameter."
            else:
                if layer.layer.use_bias:
                    kernels, biases = layer.layer.get_weights()
                else:
                    kernels = layer.layer.get_weights()[0]
            x = TimeDistributed(new_layer, name=layer.name)(x)
            if layer.layer.use_bias:
                new_layer.set_weights([kernels, biases])
            else:
                new_layer.set_weights([kernels])
            if (
                quantize_input
                and type(layer._inbound_nodes[0].inbound_layers[0]) != QDQ
            ):
                tensor_name = process_td_output_name(layer.input.name, layer, up=True)
                if output_format != "onnx":
                    tensor_name = process_plugins_name(tensor_name)
                tensor_name = process_flatten_name(tensor_name)
                if output_format != "onnx":
                    tensor_name = tensor_name.split(":")[0]
                if tensor_name in tensor_scale_dict:
                    tensor_scale_dict[tensor_name] = max(
                        tensor_scale_dict[tensor_name], scaling_factor
                    )
                else:
                    tensor_scale_dict[tensor_name] = scaling_factor
        elif type(layer) == QuantizedDepthwiseConv2D:
            x = layer_input
            layer_config = layer.get_config()
            layer_config.pop("bitwidth")
            quantize_input = layer_config.pop("quantize")
            new_layer = DepthwiseConv2D.from_config(layer_config)
            if quantize_input:
                if layer.use_bias:
                    kernels, biases, scaling_factor = layer.get_weights()
                else:
                    kernels, scaling_factor = layer.get_weights()
                assert (
                    scaling_factor.shape == ()
                ), "Unexpected shape for scaling factor parameter."
            else:
                if layer.use_bias:
                    kernels, biases = layer.get_weights()
                else:
                    kernels = layer.get_weights()[0]
            x = new_layer(x)
            if layer.use_bias:
                new_layer.set_weights([kernels, biases])
            else:
                new_layer.set_weights([kernels])
            if (
                quantize_input
                and type(layer._inbound_nodes[0].inbound_layers[0]) != QDQ
            ):
                tensor_name = process_td_output_name(layer.input.name, layer, up=True)
                if output_format != "onnx":
                    tensor_name = process_plugins_name(tensor_name)
                tensor_name = process_flatten_name(tensor_name)
                if output_format != "onnx":
                    tensor_name = tensor_name.split(":")[0]
                if tensor_name in tensor_scale_dict:
                    tensor_scale_dict[tensor_name] = max(
                        tensor_scale_dict[tensor_name], scaling_factor
                    )
                else:
                    tensor_scale_dict[tensor_name] = scaling_factor
        elif type(layer) == TimeDistributed and type(layer.layer) == QuantizedDepthwiseConv2D:
            x = layer_input
            layer_config = layer.layer.get_config()
            layer_config.pop("bitwidth")
            quantize_input = layer_config.pop("quantize")
            new_layer = DepthwiseConv2D.from_config(layer_config)
            if quantize_input:
                if layer.layer.use_bias:
                    kernels, biases, scaling_factor = layer.layer.get_weights()
                else:
                    kernels, scaling_factor = layer.layer.get_weights()
                assert (
                    scaling_factor.shape == ()
                ), "Unexpected shape for scaling factor parameter."
            else:
                if layer.layer.use_bias:
                    kernels, biases = layer.layer.get_weights()
                else:
                    kernels = layer.layer.get_weights()[0]
            x = TimeDistributed(new_layer, name=layer.name)(x)
            if layer.layer.use_bias:
                new_layer.set_weights([kernels, biases])
            else:
                new_layer.set_weights([kernels])
            if (
                quantize_input
                and type(layer._inbound_nodes[0].inbound_layers[0]) != QDQ
            ):
                tensor_name = process_td_output_name(layer.input.name, layer, up=True)
                if output_format != "onnx":
                    tensor_name = process_plugins_name(tensor_name)
                tensor_name = process_flatten_name(tensor_name)
                if output_format != "onnx":
                    tensor_name = tensor_name.split(":")[0]
                if tensor_name in tensor_scale_dict:
                    tensor_scale_dict[tensor_name] = max(
                        tensor_scale_dict[tensor_name], scaling_factor
                    )
                else:
                    tensor_scale_dict[tensor_name] = scaling_factor
        elif type(layer) == QuantizedDense:
            x = layer_input
            layer_config = layer.get_config()
            layer_config.pop("bitwidth")
            quantize_input = layer_config.pop("quantize")
            new_layer = Dense.from_config(layer_config)
            if quantize_input:
                if layer.use_bias:
                    kernels, biases, scaling_factor = layer.get_weights()
                else:
                    kernels, scaling_factor = layer.get_weights()
                assert (
                    scaling_factor.shape == ()
                ), "Unexpected shape for scaling factor parameter."
            else:
                if layer.use_bias:
                    kernels, biases = layer.get_weights()
                else:
                    kernels = layer.get_weights()[0]
            x = new_layer(x)
            if layer.use_bias:
                new_layer.set_weights([kernels, biases])
            else:
                new_layer.set_weights([kernels])
            if (
                quantize_input
                and type(layer._inbound_nodes[0].inbound_layers[0]) != QDQ
            ):
                tensor_name = process_td_output_name(layer.input.name, layer, up=True)
                if output_format != "onnx":
                    tensor_name = process_plugins_name(tensor_name)
                tensor_name = process_flatten_name(tensor_name)
                if output_format != "onnx":
                    tensor_name = tensor_name.split(":")[0]
                if tensor_name in tensor_scale_dict:
                    tensor_scale_dict[tensor_name] = max(
                        tensor_scale_dict[tensor_name], scaling_factor
                    )
                else:
                    tensor_scale_dict[tensor_name] = scaling_factor
        elif type(layer) == TimeDistributed and type(layer.layer) == QuantizedDense:
            x = layer_input
            layer_config = layer.layer.get_config()
            layer_config.pop("bitwidth")
            quantize_input = layer_config.pop("quantize")
            new_layer = Dense.from_config(layer_config)
            if quantize_input:
                if layer.layer.use_bias:
                    kernels, biases, scaling_factor = layer.layer.get_weights()
                else:
                    kernels, scaling_factor = layer.layer.get_weights()
                assert (
                    scaling_factor.shape == ()
                ), "Unexpected shape for scaling factor parameter."
            else:
                if layer.layer.use_bias:
                    kernels, biases = layer.layer.get_weights()
                else:
                    kernels = layer.layer.get_weights()[0]
            x = TimeDistributed(new_layer, name=layer.name)(x)
            if layer.layer.use_bias:
                new_layer.set_weights([kernels, biases])
            else:
                new_layer.set_weights([kernels])
            if (
                quantize_input
                and type(layer._inbound_nodes[0].inbound_layers[0]) != QDQ
            ):
                tensor_name = process_td_output_name(layer.input.name, layer, up=True)
                if output_format != "onnx":
                    tensor_name = process_plugins_name(tensor_name)
                tensor_name = process_flatten_name(tensor_name)
                if output_format != "onnx":
                    tensor_name = tensor_name.split(":")[0]
                if tensor_name in tensor_scale_dict:
                    tensor_scale_dict[tensor_name] = max(
                        tensor_scale_dict[tensor_name], scaling_factor
                    )
                else:
                    tensor_scale_dict[tensor_name] = scaling_factor
        else:
            weights = layer.get_weights()
            layer_config = layer.get_config()
            new_layer = type(layer).from_config(layer_config)
            x = new_layer(layer_input)
            new_layer.set_weights(weights)

        if layer.name in qdq_scale_dict:
            tensor_name = process_td_output_name(layer.output.name, layer)
            if output_format != "onnx":
                tensor_name = process_plugins_name(tensor_name)
            tensor_name = process_flatten_name(tensor_name)
            if output_format != "onnx":
                tensor_name = tensor_name.split(":")[0]
            tensor_scale_dict[tensor_name] = qdq_scale_dict[layer.name]

        if len(layer._outbound_nodes) == 0:
            if isinstance(x, list):
                output_tensors.extend(x)
            else:
                output_tensors.append(x)

        for node in layer._outbound_nodes:
            outbound_layer = node.outbound_layer
            if type(outbound_layer) == QDQ:
                if len(outbound_layer._outbound_nodes) == 0:
                    if isinstance(x, list):
                        output_tensors.extend(x)
                    else:
                        output_tensors.append(x)

        network_dict["new_output_tensor_of"].update({layer.name: x})

    model = keras.models.Model(inputs=model.inputs, outputs=output_tensors)
    # collapse flatten node and its previous node
    if output_format != "onnx":
        collapse_flatten_and_prev(tensor_scale_dict)
    # collapse padding and conv2d/depthwiseconv2d
    collapse_pad_and_conv(tensor_scale_dict)
    # convert input_image:0 to input_image for onnx
    # since it seems there is no :0 for input in onnx model
    if output_format == "onnx":
        if "input_image:0" in tensor_scale_dict:
            tensor_scale_dict.update(
                {"input_image": tensor_scale_dict["input_image:0"]}
            )
            tensor_scale_dict.pop("input_image:0")
    # save model to file, reset the tf graph and load it to make sure the tf op names
    # not appended with _n
    os_handle, temp_file_name = tempfile.mkstemp()
    os.close(os_handle)
    with CustomObjectScope(CUSTOM_OBJS):
        model.save(temp_file_name)
    # clear old tf graph and session
    keras.backend.clear_session()
    if create_session:
        # create a new tf session and use it as Keras session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        keras.backend.set_session(tf.Session(config=config))
    assert learning_phase in [0, 1], "Keras learning phase should be 0 or 1, got {}".format(
        learning_phase
    )
    keras.backend.set_learning_phase(learning_phase)
    with CustomObjectScope(CUSTOM_OBJS):
        new_model = keras.models.load_model(temp_file_name, compile=False)
    os.remove(temp_file_name)
    return new_model, tensor_scale_dict
