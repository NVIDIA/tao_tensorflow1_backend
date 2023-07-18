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

import json
from struct import pack, unpack

import keras
from keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D
from keras.utils import CustomObjectScope

from nvidia_tao_tf1.core.models.templates.qdq_layer import QDQ
from nvidia_tao_tf1.core.models.templates.quantized_conv2d import QuantizedConv2D
from nvidia_tao_tf1.core.models.templates.quantized_conv2dtranspose import QuantizedConv2DTranspose
from nvidia_tao_tf1.core.models.templates.quantized_depthwiseconv2d import QuantizedDepthwiseConv2D
from nvidia_tao_tf1.core.utils.path_utils import expand_path
from nvidia_tao_tf1.cv.retinanet.initializers.prior_prob import PriorProbability


QAT_LAYER_MAPPING = {
    QuantizedConv2D: Conv2D,
    QuantizedConv2DTranspose: Conv2DTranspose,
    QuantizedDepthwiseConv2D: DepthwiseConv2D
}


def check_for_quantized_layers(model):
    """Check Keras model for quantization layers."""

    # syntax valid only under Python 3
    qat_layers = [*QAT_LAYER_MAPPING.keys(), QDQ]
    for layer in model.layers:
        if type(layer) in qat_layers:
            return True
    return False


def process_quantized_layers(model, output_format, calib_cache=None, calib_json=None):
    """Remove QDQ, replace the quantized layer with non-QAT layer and extract calibration cache."""
    network_dict = {"input_layers_of": {}, "new_output_tensor_of": {}}

    # Set the input layers of each layer.
    for layer in model.layers:
        if len(layer._inbound_nodes) > 1:
            inbound_layers_list = []
            for i in range(len(layer._inbound_nodes)):
                inbound_node = layer._inbound_nodes[i]
                inbound_layers = [in_layer.name for in_layer in inbound_node.inbound_layers]
                if len(inbound_layers) > 0:
                    inbound_layers_list += inbound_layers

            network_dict["input_layers_of"].update({layer.name: sorted(inbound_layers_list)})
        else:
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
    layer_count = {}
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
        if isinstance(layer_input[0], list):
            layer_input = layer_input[0]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        if type(layer) in QAT_LAYER_MAPPING:
            x = layer_input
            layer_config = layer.get_config()
            layer_config.pop("bitwidth")
            quantize_input = layer_config.pop("quantize")
            new_layer = QAT_LAYER_MAPPING[type(layer)].from_config(layer_config)
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
                tensor_name = layer.input.name
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
            with CustomObjectScope({'PriorProbability': PriorProbability}):
                new_layer = type(layer).from_config(layer_config)
            if not isinstance(layer_input, list) or type(layer) in [
                keras.layers.Add, keras.layers.Multiply, keras.layers.Concatenate
            ]:
                x = new_layer(layer_input)
                new_layer.set_weights(weights)
            else:
                if len(layer_input) > 1:
                    if len(network_dict["input_layers_of"][layer.name]) > 1:
                        x_list = []
                        for i in range(len(layer_input)):
                            x = new_layer(layer_input[i])
                            new_layer.set_weights(weights)
                            x_list.append(x)
                        x = x_list
                    else:
                        # To support RetinaNet subnets, AnchorBox and Permute layers
                        if network_dict["input_layers_of"][layer.name][0] in layer_count:
                            layer_count[network_dict["input_layers_of"][layer.name][0]] += 1
                        else:
                            layer_count[network_dict["input_layers_of"][layer.name][0]] = 0
                        layer_count[network_dict["input_layers_of"][layer.name][0]] %= 5
                        x = new_layer(
                            layer_input[
                                layer_count[network_dict["input_layers_of"][layer.name][0]]])
                        new_layer.set_weights(weights)
                else:
                    raise ValueError("Model not supported!")

        if layer.name in qdq_scale_dict:
            tensor_name = layer.output.name
            if output_format != "onnx":
                tensor_name = tensor_name.split(":")[0]
            tensor_scale_dict[tensor_name] = qdq_scale_dict[layer.name]

        if len(layer._outbound_nodes) == 0:
            output_tensors.append(x)

        for node in layer._outbound_nodes:
            outbound_layer = node.outbound_layer
            if type(outbound_layer) == QDQ:
                if len(outbound_layer._outbound_nodes) == 0:
                    output_tensors.append(x)

        network_dict["new_output_tensor_of"].update({layer.name: x})
    model = keras.models.Model(inputs=model.inputs, outputs=output_tensors)

    if calib_cache is not None:
        cal_cache_str = "1\n"
        for tensor in tensor_scale_dict:
            scaling_factor = tensor_scale_dict[tensor] / 127.0
            cal_scale = hex(unpack("i", pack("f", scaling_factor))[0])
            assert cal_scale.startswith("0x"), "Hex number expected to start with 0x."
            cal_scale = cal_scale[2:]
            cal_cache_str += tensor + ": " + cal_scale + "\n"
        with open(expand_path(calib_cache), "w") as f:
            f.write(cal_cache_str)

    if calib_json is not None:
        calib_json_data = {"tensor_scales": {}}
        for tensor in tensor_scale_dict:
            calib_json_data["tensor_scales"][tensor] = float(tensor_scale_dict[tensor])
        with open(expand_path(calib_json), "w") as outfile:
            json.dump(calib_json_data, outfile, indent=4)

    return model, tensor_scale_dict
