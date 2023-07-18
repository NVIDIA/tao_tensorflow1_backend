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
from keras.layers import (
    BatchNormalization, Conv2D, Dense, DepthwiseConv2D,
    ReLU, Softmax, TimeDistributed
)
from keras.layers.core import Activation

from nvidia_tao_tf1.core.models.templates.qdq_layer import QDQ
from nvidia_tao_tf1.core.models.templates.quantized_conv2d import QuantizedConv2D
from nvidia_tao_tf1.core.models.templates.quantized_dense import QuantizedDense
from nvidia_tao_tf1.core.models.templates.quantized_depthwiseconv2d import QuantizedDepthwiseConv2D
from nvidia_tao_tf1.cv.faster_rcnn.layers.custom_layers import CropAndResize, Proposal, ProposalTarget

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


def create_layer_from_config(layer, input_tensor, freeze_bn=False):
    """Re-create a Keras layer from config."""
    layer_config = layer.get_config()
    weights = layer.get_weights()
    new_layer = type(layer).from_config(layer_config)
    if (type(new_layer) == BatchNormalization or
        (type(new_layer) == TimeDistributed and
         type(new_layer.layer) == BatchNormalization)):
        if freeze_bn:
            x = new_layer(input_tensor, training=False)
        else:
            x = new_layer(input_tensor)
    else:
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


def create_quantized_keras_model(model, freeze_bn=False, training=False):
    """Quantize a Keras model.

    This function replaces the Conv2D with QuantizedConv2D, ReLU with ReLU6 and adds QDQ layers
    as needed in the graph. It also uses the weights from original Keras model.

    Args:
        model (Keras model): The input Keras model.
        freeze_bn(bool): Freeze BN layers or not.
        training(bool): Flag for training or validation mode.

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
        # only input image need to be quantized
        if "input_image" in layer.name:
            input_tensor = QDQ(name=layer.name + "_qdq")(input_tensor)
        network_dict["new_output_tensor_of"].update({layer.name: input_tensor})

    output_layers = []

    for layer in model.layers:
        if len(layer._outbound_nodes) == 0:
            output_layers.append(layer.name)
            if type(layer) in output_types:
                output_layers = _add_outputs(layer, output_layers)
    output_tensors = []

    record_cr_rois = None
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
        if is_output or type(layer) in [Proposal, ProposalTarget]:
            x = layer_input
            x = create_layer_from_config(layer, x)
        else:
            if type(layer) == Conv2D or (
               type(layer) == TimeDistributed and
               type(layer.layer) == Conv2D):
                x = layer_input
                if type(layer) == Conv2D:
                    layer_config = layer.get_config()
                else:
                    layer_config = layer.layer.get_config()
                layer_config["quantize"] = False
                layer_config["bitwidth"] = 8
                conv_act = layer_config["activation"]
                if conv_act != "linear":
                    layer_config["activation"] = "linear"
                new_layer = QuantizedConv2D.from_config(layer_config)
                if type(layer) == Conv2D:
                    if layer.use_bias:
                        kernels, biases = layer.get_weights()
                        x = new_layer(x)
                        new_layer.set_weights([kernels, biases])
                    else:
                        kernels = layer.get_weights()[0]
                        x = new_layer(x)
                        new_layer.set_weights([kernels])
                else:
                    if layer.layer.use_bias:
                        kernels, biases = layer.layer.get_weights()
                        x = TimeDistributed(new_layer, name=layer.name)(x)
                        new_layer.set_weights([kernels, biases])
                    else:
                        kernels = layer.layer.get_weights()[0]
                        x = TimeDistributed(new_layer, name=layer.name)(x)
                        new_layer.set_weights([kernels])
                if conv_act == "linear":
                    # TensorRT folds the BN into previous Conv. layer.
                    # So if the output of this Conv. layer goes to only
                    # a BN layer, then don't add a QDQ layer.
                    next_layer_is_relu = False
                    next_layer = layer._outbound_nodes[0].outbound_layer
                    if len(layer._outbound_nodes) == 1:
                        if type(next_layer) == Activation:
                            next_layer_act = next_layer.get_config()['activation']
                            if next_layer_act == "relu":
                                next_layer_is_relu = True
                        elif type(next_layer) == ReLU:
                            next_layer_is_relu = True
                    next_layer_is_bn = False
                    if type(layer) == Conv2D:
                        if (len(layer._outbound_nodes) == 1 and
                           type(layer._outbound_nodes[0].outbound_layer) == BatchNormalization):
                            next_layer_is_bn = True
                    else:
                        next_layer = layer._outbound_nodes[0].outbound_layer
                        if (len(layer._outbound_nodes) == 1 and
                           type(next_layer) == TimeDistributed and
                           type(next_layer.layer) == BatchNormalization):
                            next_layer_is_bn = True
                    if (not next_layer_is_relu) and (not next_layer_is_bn):
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

            elif type(layer) == DepthwiseConv2D or (
                 type(layer) == TimeDistributed and
                 type(layer.layer) == DepthwiseConv2D):
                x = layer_input
                if type(layer) == DepthwiseConv2D:
                    layer_config = layer.get_config()
                else:
                    layer_config = layer.layer.get_config()
                layer_config["quantize"] = False
                layer_config["bitwidth"] = 8
                conv_act = layer_config["activation"]
                if conv_act != "linear":
                    layer_config["activation"] = "linear"
                new_layer = QuantizedDepthwiseConv2D.from_config(layer_config)
                if type(layer) == DepthwiseConv2D:
                    if layer.use_bias:
                        kernels, biases = layer.get_weights()
                        x = new_layer(x)
                        new_layer.set_weights([kernels, biases])
                    else:
                        kernels = layer.get_weights()[0]
                        x = new_layer(x)
                        new_layer.set_weights([kernels])
                else:
                    if layer.layer.use_bias:
                        kernels, biases = layer.layer.get_weights()
                        x = TimeDistributed(new_layer, name=layer.name)(x)
                        new_layer.set_weights([kernels, biases])
                    else:
                        kernels = layer.layer.get_weights()[0]
                        x = TimeDistributed(new_layer, name=layer.name)(x)
                        new_layer.set_weights([kernels])
                if conv_act == "linear":
                    # TensorRT folds the BN into previous Conv. layer.
                    # So if the output of this Conv. layer goes to only
                    # a BN layer, then don't add a QDQ layer.
                    next_layer_is_relu = False
                    next_layer = layer._outbound_nodes[0].outbound_layer
                    if len(layer._outbound_nodes) == 1:
                        if type(next_layer) == Activation:
                            next_layer_act = next_layer.get_config()['activation']
                            if next_layer_act == "relu":
                                next_layer_is_relu = True
                        elif type(next_layer) == ReLU:
                            next_layer_is_relu = True
                    next_layer_is_bn = False
                    if type(layer) == DepthwiseConv2D:
                        if (len(layer._outbound_nodes) == 1 and
                           type(layer._outbound_nodes[0].outbound_layer) == BatchNormalization):
                            next_layer_is_bn = True
                    else:
                        next_layer = layer._outbound_nodes[0].outbound_layer
                        if (len(layer._outbound_nodes) == 1 and
                           type(next_layer) == TimeDistributed and
                           type(next_layer.layer) == BatchNormalization):
                            next_layer_is_bn = True
                    if (not next_layer_is_relu) and (not next_layer_is_bn):
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

            elif type(layer) == Dense or (
               type(layer) == TimeDistributed and
               type(layer.layer) == Dense):
                x = layer_input
                if type(layer) == Dense:
                    layer_config = layer.get_config()
                else:
                    layer_config = layer.layer.get_config()
                layer_config["quantize"] = False
                layer_config["bitwidth"] = 8
                conv_act = layer_config["activation"]
                if conv_act != "linear":
                    layer_config["activation"] = "linear"
                new_layer = QuantizedDense.from_config(layer_config)
                if type(layer) == Dense:
                    if layer.use_bias:
                        kernels, biases = layer.get_weights()
                        x = new_layer(x)
                        new_layer.set_weights([kernels, biases])
                    else:
                        kernels = layer.get_weights()[0]
                        x = new_layer(x)
                        new_layer.set_weights([kernels])
                else:
                    if layer.layer.use_bias:
                        kernels, biases = layer.layer.get_weights()
                        x = TimeDistributed(new_layer, name=layer.name)(x)
                        new_layer.set_weights([kernels, biases])
                    else:
                        kernels = layer.layer.get_weights()[0]
                        x = TimeDistributed(new_layer, name=layer.name)(x)
                        new_layer.set_weights([kernels])
                # TensorRT does not fuse FC and Relu6, so always insert QDQ after FC
                x = QDQ(name=layer.name + "_qdq")(x)
                if conv_act == "relu":
                    x = ReLU(max_value=6.0)(x)
                else:
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

            elif type(layer) == ReLU:
                x = layer_input
                x = ReLU(max_value=6.0, name=layer.name)(x)
                x = QDQ(name=layer.name + "_qdq")(x)

            elif type(layer) == BatchNormalization or (
                type(layer) == TimeDistributed and type(layer.layer) == BatchNormalization
            ):
                # TensorRT fuses Conv + BN + ReLU together.
                # So if previous layer is Conv. and next layer is
                # ReLU we should not add QDQ layers.
                x = layer_input
                # BN layers training=False it not serialized in config
                # so we have to pin it in the call method
                x = create_layer_from_config(layer, x, freeze_bn=freeze_bn)
                next_layer_is_relu = False
                if len(layer._outbound_nodes) == 1:
                    next_layer = layer._outbound_nodes[0].outbound_layer
                    if type(next_layer) == ReLU:
                        next_layer_is_relu = True
                    elif type(next_layer) == Activation:
                        next_layer_cfg = next_layer.get_config()
                        if next_layer_cfg["activation"] == "relu":
                            next_layer_is_relu = True
                prev_layer_is_conv = False
                if len(layer._inbound_nodes[0].inbound_layers) == 1:
                    prev_layer = layer._inbound_nodes[0].inbound_layers[0]
                    if (type(layer) == BatchNormalization and
                        type(prev_layer) in [Conv2D, DepthwiseConv2D]):
                        prev_layer_is_conv = True
                    elif (type(layer) == TimeDistributed and
                          type(prev_layer) == TimeDistributed and
                          type(prev_layer.layer) in [Conv2D, DepthwiseConv2D]):
                        prev_layer_is_conv = True
                if not (next_layer_is_relu and prev_layer_is_conv):
                    x = QDQ(name=layer.name + "_qdq")(x)

            else:
                x = layer_input
                # CropAndResize only need the first output from ProposalTarget as
                # the 2nd input for training model.
                if type(layer) == CropAndResize:
                    if training:
                        x = [x[0], x[1][0], x[2]]
                    record_cr_rois = x[1]
                x = create_layer_from_config(layer, x)
                x = QDQ(name=layer.name + "_qdq")(x)

        if len(layer._outbound_nodes) == 0 or (training and 'rpn_out' in layer.name):
            output_tensors.append(x)
        network_dict["new_output_tensor_of"].update({layer.name: x})
    if (not training) and (record_cr_rois is not None):
        output_tensors.insert(0, record_cr_rois)
    model = keras.models.Model(inputs=model.inputs, outputs=output_tensors)

    return model
