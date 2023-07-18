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
"""Create a Keras model for Quantization-Aware Training (QAT)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.layers import (
    Add,
    Average,
    Concatenate,
    Maximum,
    Minimum,
    Multiply,
    Permute,
    Subtract,
)
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, DepthwiseConv2D, Softmax
from keras.layers import ELU, LeakyReLU, PReLU, ReLU
from keras.layers.core import Activation
from nvidia_tao_tf1.core.models.templates.qdq_layer import QDQ
from nvidia_tao_tf1.core.models.templates.quantized_conv2d import QuantizedConv2D
from nvidia_tao_tf1.core.models.templates.quantized_conv2dtranspose import QuantizedConv2DTranspose
from nvidia_tao_tf1.core.models.templates.quantized_depthwiseconv2d import QuantizedDepthwiseConv2D

output_types = [
    Activation,
    ReLU,
    Softmax,
    Add,
    Subtract,
    Multiply,
    Average,
    Maximum,
    Minimum,
    Concatenate,
    Permute,
]


def create_layer_from_config(layer, input_tensor):
    """Re-create a Keras layer from config."""
    layer_config = layer.get_config()
    weights = layer.get_weights()
    new_layer = type(layer).from_config(layer_config)
    x = new_layer(input_tensor)
    new_layer.set_weights(weights)
    return x


def _add_outputs(layer, output_layers):
    """Recursively find the output layers."""
    for prev_layer in layer._inbound_nodes[0].inbound_layers:
        if prev_layer.name not in output_layers:
            output_layers.append(prev_layer.name)
        if type(prev_layer) in output_types:
            output_layers = _add_outputs(prev_layer, output_layers)
    return output_layers


def create_quantized_keras_model(model):
    """Quantize a Keras model.

    This function replaces the Conv2D with QuantizedConv2D, ReLU with ReLU6 and adds QDQ layers
    as needed in the graph. It also uses the weights from original Keras model.

    Args:
        model (Keras model): The input Keras model.

    Returns:
        model (Keras model): A keras model prepared for Quantization-Aware Training.
    """
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
        input_tensor = QDQ(name=layer.name + "_qdq")(input_tensor)
        network_dict["new_output_tensor_of"].update({layer.name: input_tensor})

    output_layers = []

    for layer in model.layers:
        if len(layer._outbound_nodes) == 0:
            output_layers.append(layer.name)
            if type(layer) in output_types:
                output_layers = _add_outputs(layer, output_layers)

    output_tensors = []

    for layer in model.layers:
        if layer.name not in network_dict["input_layers_of"]:
            # It's an input layer.
            continue

        # Determine input tensors.
        layer_input = [
            network_dict["new_output_tensor_of"][layer_aux]
            for layer_aux in network_dict["input_layers_of"][layer.name]
        ]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        is_output = layer.name in output_layers
        if is_output:
            x = layer_input
            x = create_layer_from_config(layer, x)
        else:
            if type(layer) in [Conv2D, Conv2DTranspose, DepthwiseConv2D]:
                x = layer_input
                layer_config = layer.get_config()
                layer_config["quantize"] = False
                layer_config["bitwidth"] = 8
                conv_act = layer_config["activation"]
                if conv_act != "linear":
                    layer_config["activation"] = "linear"
                if type(layer) == Conv2D:
                    new_layer = QuantizedConv2D.from_config(layer_config)
                elif type(layer) == Conv2DTranspose:
                    new_layer = QuantizedConv2DTranspose.from_config(layer_config)
                elif type(layer) == DepthwiseConv2D:
                    new_layer = QuantizedDepthwiseConv2D.from_config(layer_config)
                else:
                    raise NotImplementedError(
                        "Quantization for "+layer.__class__.__name__+" is not implemented"
                        )
                if layer.use_bias:
                    kernels, biases = layer.get_weights()
                    x = new_layer(x)
                    new_layer.set_weights([kernels, biases])
                else:
                    kernels = layer.get_weights()[0]
                    x = new_layer(x)
                    new_layer.set_weights([kernels])
                if conv_act == "linear":
                    # TensorRT folds the BN into previous Conv. layer.
                    # So if the output of this Conv. layer goes to only
                    # a BN layer, then don't add a QDQ layer.
                    if (
                        len(layer._outbound_nodes) != 1
                        or type(layer._outbound_nodes[0].outbound_layer)
                        != BatchNormalization
                    ):
                        x = QDQ(name=layer.name + "_qdq")(x)
                else:
                    # TensorRT fuses ReLU back into the Conv. layer.
                    # Other activations are implemented as separate layers.
                    # So we need to add QDQ layer unless the activation is ReLU
                    if conv_act == "relu":
                        x = ReLU(max_value=6.0)(x)
                    else:
                        x = QDQ(name=layer.name + "_qdq")(x)
                        x = Activation(conv_act)(x)
                    x = QDQ(name=layer.name + "_act_qdq")(x)

            elif type(layer) == Activation:
                # Need QDQ layer after every activation layers (except output layers.)
                x = layer_input
                layer_config = layer.get_config()
                if layer_config["activation"] == "relu":
                    x = ReLU(max_value=6.0, name=layer.name)(x)
                else:
                    x = create_layer_from_config(layer, x)
                x = QDQ(name=layer.name + "_qdq")(x)

            elif type(layer) in [ReLU, PReLU, ELU, LeakyReLU]:
                x = layer_input
                layer_config = layer.get_config()
                x = ReLU(max_value=6.0, name=layer.name)(x)
                x = QDQ(name=layer.name + "_qdq")(x)

            elif type(layer) == BatchNormalization:
                # TensorRT fuses Conv + BN + ReLU together.
                # So if previous layer is Conv. and next layer is
                # ReLU we should not add QDQ layers.
                x = layer_input
                x = create_layer_from_config(layer, x)
                next_layer_is_relu = False
                if len(layer._outbound_nodes) == 1:
                    next_layer = layer._outbound_nodes[0].outbound_layer
                    if type(next_layer) in [ReLU, PReLU, ELU, LeakyReLU]:
                        next_layer_is_relu = True
                    elif type(next_layer) == Activation:
                        next_layer_cfg = next_layer.get_config()
                        if next_layer_cfg["activation"] == "relu":
                            next_layer_is_relu = True
                prev_layer_is_conv = (
                    len(layer._inbound_nodes[0].inbound_layers) == 1
                    and type(layer._inbound_nodes[0].inbound_layers[0]) in
                    [Conv2D, Conv2DTranspose, DepthwiseConv2D]
                )
                if not (next_layer_is_relu and prev_layer_is_conv):
                    x = QDQ(name=layer.name + "_qdq")(x)

            else:
                x = layer_input
                x = create_layer_from_config(layer, x)
                x = QDQ(name=layer.name + "_qdq")(x)

        if len(layer._outbound_nodes) == 0:
            output_tensors.append(x)

        network_dict["new_output_tensor_of"].update({layer.name: x})

    model = keras.models.Model(inputs=model.inputs, outputs=output_tensors)

    return model
