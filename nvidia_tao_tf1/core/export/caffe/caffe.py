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

import copy
import logging

import google.protobuf.text_format as text_format
import keras
import numpy as np

logger = logging.getLogger(__name__)  # noqa

from nvidia_tao_tf1.core.export.caffe import caffe_pb2
from nvidia_tao_tf1.core.export.caffe.net_spec import layers as L
from nvidia_tao_tf1.core.export.caffe.net_spec import NetSpec


class CaffeExporter(object):
    """A class to handle exporting a Keras model to Caffe."""

    def __init__(self):
        """Initialization routine."""
        self._exported_layers = {}
        self._special_paddings = {}

    def _add_activation_layer(
        self, name, activation_type, previous_layer, parameters=None
    ):
        """Add an activation layer to the model.

        Args:
            name (str): name of the layer to add activation to.
            activation_type (str): Keras string identifier of activation to return.
            previous_layer (layer): layer to append activation to.
            parameters (list): optional list of parameters needed for a given activation layer.
        """
        if activation_type == "linear":
            layer = None
        elif activation_type == "sigmoid":
            layer = L.Sigmoid(previous_layer)
        elif activation_type == "softmax":
            layer = L.Softmax(previous_layer)
        elif activation_type == "softmax_with_axis":
            # Requires one parameter: the axis over which to apply the softmax.
            axis = parameters[0]
            assert (
                axis == 1
            ), "Only softmax over axis = 1 is supported in TensorRT at the moment."
            layer = L.Softmax(previous_layer, axis=axis)
        elif activation_type == "relu":
            layer = L.ReLU(previous_layer)
        elif activation_type == "tanh":
            layer = L.TanH(previous_layer)
        else:
            raise ValueError("Unhandled activation type: %s" % activation_type)
        if layer is not None:
            name = self._get_activation_layer_name(name, activation_type)
            self._add_layer(name, layer)

    def _add_batchnorm_layer(self, keras_layer):
        """Add a batchnorm layer to the model.

        Args:
            keras_layer: the Keras layer to convert.
        """
        assert keras_layer.scale, "Expect scaling to be enabled for batch norm."
        assert keras_layer.center, "Expect centering to be enabled for batch norm."
        caffe_layer = L.Scale(
            self._get_previous_layer(keras_layer),
            axis=keras_layer.axis,
            bias_term=keras_layer.center,
        )
        self._add_layer(keras_layer.name, caffe_layer)

        # Save weights.
        weights = keras_layer.get_weights()
        gamma, beta, moving_mean, moving_var = weights
        # We need to divide the scaling factor (gamma) by the standard deviation.
        denom = np.sqrt(moving_var + keras_layer.epsilon)
        scale = gamma / denom
        bias = beta - (gamma * moving_mean) / denom
        self._params[keras_layer.name] = [scale, bias]

    def _add_concatenate_layer(self, keras_layer):
        """Add a concatenation layer to the model.

        Args:
            keras_layer: the Keras layer to convert.
        """
        previous_layers = keras_layer._inbound_nodes[-1].inbound_layers
        logger.debug(
            "Concatenate layers %s along axis=%d",
            repr([l.name for l in previous_layers]),
            keras_layer.axis,
        )
        bottom_layers = self._get_bottom_layers(previous_layers)
        caffe_layer = L.Concat(*bottom_layers, axis=keras_layer.axis)
        self._add_layer(keras_layer.name, caffe_layer)

    def _add_conv_layer(self, keras_layer, pad_h=None, pad_w=None):
        """Add a conv layer to the model.

        This applies to both regular and transpose convolutions.

        Args:
            keras_layer: the Keras layer to convert.
        """
        kernel_h, kernel_w = keras_layer.kernel_size
        stride_h, stride_w = keras_layer.strides

        # Set padding according to border mode.
        if pad_h is None or pad_w is None:
            if keras_layer.padding == "valid":
                pad_w, pad_h = 0, 0
            elif keras_layer.padding == "same":
                if type(keras_layer) == keras.layers.convolutional.Conv2D:
                    dilation_h, dilation_w = keras_layer.dilation_rate
                    # In the case of no dilation, i.e. dilation == 1, pad = kernel // 2.
                    pad_h = ((kernel_h - 1) * dilation_h + 1) // 2
                    pad_w = ((kernel_w - 1) * dilation_w + 1) // 2
                else:
                    dilation_h, dilation_w = keras_layer.dilation_rate
                    # In the case of no dilation, i.e. dilation == 1, pad = kernel // 2.
                    pad_h = ((kernel_h - 1) * dilation_h + 1 - stride_h) // 2
                    pad_w = ((kernel_w - 1) * dilation_w + 1 - stride_w) // 2
            else:
                raise ValueError("Unknown padding type: %s" % keras_layer.padding)
        else:
            if keras_layer.padding == "valid":
                pass
            elif keras_layer.padding == "same":
                dilation_h, dilation_w = keras_layer.dilation_rate
                # In the case of no dilation, i.e. dilation == 1, pad = kernel // 2.
                pad_h = pad_h + ((kernel_h - 1) * dilation_h + 1) // 2
                pad_w = pad_w + ((kernel_w - 1) * dilation_w + 1) // 2
            else:
                raise ValueError("Unknown padding type: %s" % keras_layer.padding)

        if type(keras_layer) == keras.layers.convolutional.Conv2D:
            layer_func = L.Convolution
        else:
            layer_func = L.Deconvolution

        caffe_layer = layer_func(
            self._get_previous_layer(keras_layer),
            convolution_param=dict(
                num_output=keras_layer.filters,
                kernel_h=kernel_h,
                kernel_w=kernel_w,
                pad_h=pad_h,
                pad_w=pad_w,
                stride_h=stride_h,
                stride_w=stride_w,
                dilation=list(keras_layer.dilation_rate),
            ),
        )

        self._add_layer(keras_layer.name, caffe_layer)

        self._add_activation_layer(
            keras_layer.name,
            keras_layer.activation.__name__,
            self._exported_layers[keras_layer.name],
        )

        # Save weights.
        weights = keras_layer.get_weights()
        kernels = weights[0]
        biases = (
            np.zeros((kernels.shape[-1]), dtype=kernels.dtype)
            if len(weights) == 1
            else weights[1]
        )
        # Convert kernel shape from Keras to Caffe ordering.
        # For convolutions:
        #    Keras (h, w, n_in, n_out) to Caffe (n_out, n_in, h, w).
        # For transpose convolutions:
        #    Keras (h, w, n_out, n_in) to Caffe (n_in, n_out, h, w).
        # The same transpose operation works in either case.
        kernels = kernels.transpose(3, 2, 0, 1)
        self._params[keras_layer.name] = kernels, biases

    def _add_dense_layer(self, keras_layer):
        """Add a dense layer to the model.

        Args:
            keras_layer: the Keras layer to convert.
        """
        caffe_layer = L.InnerProduct(
            self._get_previous_layer(keras_layer),
            inner_product_param=dict(num_output=keras_layer.units),
        )

        self._add_layer(keras_layer.name, caffe_layer)

        self._add_activation_layer(
            keras_layer.name,
            keras_layer.activation.__name__,
            self._exported_layers[keras_layer.name],
        )

        # Save weights.
        weights, biases = keras_layer.get_weights()
        weights = weights.transpose(1, 0)
        self._params[keras_layer.name] = weights, biases

    def _add_eltwise_layer(self, keras_layer, operation):
        previous_layers = keras_layer._inbound_nodes[-1].inbound_layers
        bottom_layers = self._get_bottom_layers(previous_layers)
        if operation == "add":
            operation = caffe_pb2.EltwiseParameter.SUM
        elif operation == "subtract":
            caffe_layer = L.Power(bottom_layers[1], power_param=dict(scale=-1))
            self._add_layer(previous_layers[1].name + "_negate", caffe_layer)
            bottom_layers[1] = caffe_layer
            operation = caffe_pb2.EltwiseParameter.SUM
        elif operation == "multiply":
            operation = caffe_pb2.EltwiseParameter.PROD
        else:
            raise ValueError("Unsupported operation: %s" % operation)

        caffe_layer = L.Eltwise(*bottom_layers, eltwise_param=dict(operation=operation))
        self._add_layer(keras_layer.name, caffe_layer)

    def _add_flatten_layer(self, keras_layer):
        """Add a flatten layer to the model.

        Args:
            keras_layer: the Keras layer to convert.
        """
        caffe_layer = L.Flatten(self._get_previous_layer(keras_layer))
        self._add_layer(keras_layer.name, caffe_layer)

    def _add_input_layer(self, keras_layer, in_name):
        """Add an input layer.

        Args:
            keras_layer: the Keras layer to convert.
            in_name: name to give to input layer.
        """
        input_dim = keras_layer.batch_input_shape[1:]
        # To ensure caffe-to-TensorRT export compatibility:
        # TensorRT assumes all input shape to be 4-dimensional: [B, H, W, C].
        # If the input is a vector, dim should be [1, 1, 1, C].
        # If the input is an image, dim should be [1, H, W, C].
        dim = ((1,) * (4 - len(input_dim))) + input_dim
        caffe_layer = L.Input(shape=[dict(dim=list(dim))])
        self._add_layer(in_name, caffe_layer)
        logger.debug("Input shape: %s" % str(input_dim))

    def _add_layer(self, name, layer):
        """Add a layer to the Caffe model."""
        self._net_spec.tops[name] = layer  # Replacing setattr(self._net_spec, name, layer) to avoid object access violation
        self._exported_layers[name] = layer

    def _add_pool2D_layer(self, keras_layer, pool_type):
        """Add a 2D pooling layer to the model.

        Args:
            keras_layer: the Keras layer to convert.
        """
        kernel_h, kernel_w = keras_layer.pool_size
        stride_h, stride_w = keras_layer.strides
        # Set padding according to border mode.
        if keras_layer.padding == "valid":
            pad_w, pad_h = 0, 0
        elif keras_layer.padding == "same":
            pad_w, pad_h = (kernel_h - 1) // 2, (kernel_w - 1) // 2
        else:
            raise ValueError("Unknown padding type: %s" % keras_layer.padding)

        pool_num = caffe_pb2.PoolingParameter.PoolMethod.Value(pool_type)
        caffe_layer = L.Pooling(
            self._get_previous_layer(keras_layer),
            pooling_param=dict(
                pool=pool_num,
                kernel_h=kernel_h,
                kernel_w=kernel_w,
                pad_h=pad_h,
                pad_w=pad_w,
                stride_h=stride_h,
                stride_w=stride_w,
            ),
        )

        self._add_layer(keras_layer.name, caffe_layer)

    def _add_reshape_layer(self, keras_layer):
        """Add a reshape layer to the model.

        Args:
            keras_layer: the Keras layer to convert.
        """
        shape = keras_layer.target_shape
        # Prepend a "0" to the shape to denote the fact that
        # we want to keep the batch dimension unchanged.
        caffe_shape = list((0,) + shape)
        caffe_layer = L.Reshape(
            self._get_previous_layer(keras_layer),
            reshape_param=dict(shape=dict(dim=caffe_shape)),
        )
        self._add_layer(keras_layer.name, caffe_layer)

    def _create_net_spec(self):
        """Create an empty Caffe ``NetSpec``."""
        self._net_spec = NetSpec()
        self._params = {}

    @staticmethod
    def _get_activation_layer_name(name, activation_type):
        """Return Caffe "top" name to assign to an activation layer.

        Args:
            activation_type (str): activation type.
        """
        if activation_type == "softmax_with_axis":
            layer_name = "%s" % (name)
        else:
            layer_name = "%s/%s" % (name, activation_type.capitalize())
        return layer_name

    def _get_bottom_layers(self, previous_layers):
        return [
            self._exported_layers[self._get_caffe_top_name(l)] for l in previous_layers
        ]

    @classmethod
    def _get_caffe_top_name(cls, keras_layer):
        """Get the Caffe "top" assigned with a layer.

        This handles the case where an activation layer is implicitly added
        during Caffe conversion.

        For example if we have a Keras layer like:
            keras.layers.Conv2D(name='conv2d', activation='relu')

        Then we will get two Caffe layers:

            - ``Convolution(bottom='...', top='conv2d')``
            - ``ReLU(bottom='conv2d', top='conv2d/Relu')``

        In that case this function will return ``conv2d/Relu``.

        Args:
            keras_layer: Keras layer to get corresponding Caffe top of.
        """
        name = keras_layer.name
        if (
            hasattr(keras_layer, "activation")
            and keras_layer.activation.__name__ != "linear"
        ):
            name = cls._get_activation_layer_name(name, keras_layer.activation.__name__)
        return name

    def _get_previous_layer(self, layer):
        """Return the preceding layer.

        Raises an error if the specified layer has multiple inbound layers.

        Args:
            layer: the layer to get preceding layer of.
        """
        inbound_layers = layer._inbound_nodes[-1].inbound_layers
        if len(inbound_layers) > 1:
            raise RuntimeError(
                "This function does not support multiple "
                "inbound nodes. Got %s" % len(inbound_layers)
            )
        inbound_layer = inbound_layers[0]
        name = self._get_caffe_top_name(inbound_layer)
        return self._exported_layers[name]

    def export(self, model, prototxt_filename, model_filename, output_node_names):
        """Export Keras model to Caffe.

        This creates two files:

        - A "proto" file that defines the topology of the exported model.
        - A "caffemodel" file that includes the weights of the exported model.

        Args:
            model (Model): Keras model to export.
            prototxt_filename (str): file to write exported proto to.
            model_filename (str): file to write exported model weights to.
            output_node_names (list of str): list of model output node names as
            as Caffe layer names. If not provided, the model output layers are used.
        Returns:
            The names of the input and output nodes. These must be
            passed to the TensorRT optimization tool to identify
            input and output blobs.
        """
        # Get names of output nodes.
        # If output node names are not given, use the model output layers.
        if output_node_names is None:
            out_names = [
                self._get_caffe_top_name(layer) for layer in model._output_layers
            ]
        else:
            out_names = output_node_names

        # Create list to save input node names.
        in_names = []

        # Explore the graph in Breadth First Search fashion, starting from the input layers.
        layers_to_explore = copy.copy(model._input_layers)

        self._create_net_spec()

        # Loop until we have explored all layers.
        while layers_to_explore:

            logger.debug(
                "Layers to explore: %s",
                repr([layer.name for layer in layers_to_explore]),
            )

            # Pick a layer to explore from the list.
            layer = layers_to_explore.pop(0)

            inbound_layers = layer._inbound_nodes[-1].inbound_layers
            predecessor_names = [self._get_caffe_top_name(l) for l in inbound_layers]

            if not all([l in self._exported_layers for l in predecessor_names]):
                # Some of the inbound layers have not been explored yet.
                # Skip this layer for now, it will come back to the list
                # of layers to explore as the outbound layer of one of the
                # yet unexplored layers.
                continue

            logger.debug("Processing layer %s: type=%s" % (layer.name, type(layer)))

            # Layer-specific handling.
            if type(layer) == keras.layers.InputLayer:
                in_name = self._get_caffe_top_name(layer)
                self._add_input_layer(layer, in_name)
                in_names.append(in_name)
            elif type(layer) in [
                keras.layers.convolutional.Conv2D,
                keras.layers.convolutional.Conv2DTranspose,
            ]:
                # Export Conv2D, and to handle ZeroPadding2D for Conv2D layer
                conv_outbound_nodes = layer._outbound_nodes
                layers_after_conv = [
                    node.outbound_layer for node in conv_outbound_nodes
                ]
                if (
                    len(layers_after_conv) == 1
                    and type(layers_after_conv[0])
                    == keras.layers.convolutional.ZeroPadding2D
                ):
                    padding = layers_after_conv[0].padding
                    if (
                        padding[0][0] == padding[0][1]
                        and padding[1][0] == padding[1][1]
                    ):
                        pad_h = padding[0][0]
                        pad_w = padding[1][0]
                        layer._outbound_nodes = layers_after_conv[0]._outbound_nodes
                        layer._outbound_nodes[0].inbound_layers = [layer]
                        self._special_paddings[
                            layer._outbound_nodes[0].outbound_layer.name
                        ] = (pad_h, pad_w)
                    else:
                        raise ValueError("Asymmetric padding is not supported!")

                if layer.name in self._special_paddings.keys():
                    self._add_conv_layer(
                        layer,
                        pad_h=self._special_paddings[layer.name][0],
                        pad_w=self._special_paddings[layer.name][1],
                    )
                else:
                    self._add_conv_layer(layer)
            elif type(layer) == keras.layers.normalization.BatchNormalization:
                self._add_batchnorm_layer(layer)
            elif type(layer) == keras.layers.core.Reshape:
                self._add_reshape_layer(layer)
            elif type(layer) in [
                keras.layers.core.Dropout,
                keras.layers.core.SpatialDropout2D,
            ]:
                # Dropout is a pass-through during inference, just pretend we
                # have exported this layer by pointing to the previous layer
                # in the graph.
                self._exported_layers[layer.name] = self._get_previous_layer(layer)
            elif type(layer) == keras.layers.core.Activation:
                self._add_activation_layer(
                    layer.name,
                    layer.activation.__name__,
                    self._get_previous_layer(layer),
                )
            elif type(layer) == keras.layers.core.Dense:
                self._add_dense_layer(layer)
            elif type(layer) == keras.layers.core.Flatten:
                self._add_flatten_layer(layer)
            elif type(layer) == keras.layers.pooling.MaxPooling2D:
                self._add_pool2D_layer(layer, pool_type="MAX")
            elif type(layer) == keras.layers.pooling.AveragePooling2D:
                self._add_pool2D_layer(layer, pool_type="AVE")
            elif type(layer) == keras.layers.Concatenate:
                self._add_concatenate_layer(layer)
            elif type(layer) == keras.engine.training.Model:
                # This is a model-in-model type of architecture and this layer
                # is a container for other layers. Look into the first
                # layer and keep following outbound nodes.
                layer = layer.layers[0]
            elif type(layer) == keras.layers.Softmax:
                self._add_activation_layer(
                    layer.name,
                    "softmax_with_axis",
                    self._get_previous_layer(layer),
                    parameters=[layer.axis],
                )
            elif type(layer) == keras.layers.Add:
                self._add_eltwise_layer(layer, operation="add")
            elif type(layer) == keras.layers.Subtract:
                self._add_eltwise_layer(layer, operation="subtract")
            elif type(layer) == keras.layers.Multiply:
                self._add_eltwise_layer(layer, operation="multiply")
            else:
                raise ValueError("Unhandled layer type: %s" % type(layer))

            outbound_nodes = layer._outbound_nodes
            layers_to_explore.extend([node.outbound_layer for node in outbound_nodes])

            if hasattr(layer, "data_format"):
                if layer.data_format != "channels_first":
                    raise ValueError("Only 'channels_first' is supported.")

            logger.debug("Explored layers: %s", repr(self._exported_layers.keys()))

        # If output node names are provided, then remove layers after them. Start from the output
        # nodes, move towards the input, and mark visited layers. Unmarked layers are removed.
        if output_node_names is not None:
            self._remove_layers_after_outputs(output_node_names)

        self._save_protobuf(prototxt_filename)
        self._save_weights(prototxt_filename, model_filename)

        return in_names, out_names

    def _remove_layers_after_outputs(self, output_node_names):
        """Remove unnecessary layers after the given output node names."""
        self._marked_layers = set()

        for output_node_name in output_node_names:
            layer = self._exported_layers.get(output_node_name)

            if layer is None:
                raise KeyError(
                    "Output node %s does not exist in the Caffe model."
                    % output_node_name
                )

            # Mark the output layer and its inputs recursively.
            self._mark_layers(layer)

        # Find layers that were not marked.
        exported_layers_set = set(self._exported_layers.values())

        unmarked_layers = exported_layers_set.difference(self._marked_layers)

        # Remove the unmarked layers from the Caffe NetSpec and dictionary of parameters.
        if unmarked_layers:
            # Get a mapping from the layer objects to layer names.
            layer_to_name = {v: k for k, v in self._exported_layers.items()}

            for unmarked_layer in unmarked_layers:
                layer_name = layer_to_name[unmarked_layer]
                self._net_spec.tops.pop(layer_name)
                self._params.pop(
                    layer_name, None
                )  # Some layers do not have any parameters.

    def _mark_layers(self, layer):
        """Mark layers to be exported by adding them to a set of marked layers."""
        # Check if the path to the inputs is already traversed.
        if layer in self._marked_layers:
            return

        self._marked_layers.add(layer)

        # Mark recursively layers before the current layer.
        for input_layer in layer.fn.inputs:
            self._mark_layers(input_layer)

    def _save_protobuf(self, prototxt_filename):
        """Write protobuf out."""
        with open(prototxt_filename, "w") as f:
            f.write(str(self._net_spec.to_proto()))

    def _save_weights(self, prototxt_filename, model_filename):
        """Write weights out."""
        net = caffe_pb2.NetParameter()
        text_format.Merge(open(prototxt_filename, "r").read(), net)
        for layer in net.layer:
            layer.phase = caffe_pb2.TEST
            name = layer.name
            if name in self._params:
                for source_param in self._params[name]:
                    blob = layer.blobs.add()
                    # Add dims.
                    for dim in source_param.shape:
                        blob.shape.dim.append(dim)
                    # Add blob.
                    blob.data.extend(source_param.flat)
        with open(model_filename, "wb") as f:
            f.write(net.SerializeToString())


def keras_to_caffe(model, prototxt_filename, model_filename, output_node_names=None):
    """Export a Keras model to Caffe.

    This creates two files:

    - A "proto" file that defines the topology of the exported model.
    - A "caffemodel" file that includes the weights of the exported model.

    Args:
        model (Model): Keras model to export.
        proto_filename (str): file to write exported proto to.
        output_node_names (list of str): list of model output node names as
        as caffe layer names. if not provided, then the last layer is assumed
        to be the output node.
    Returns:
        The names of the input and output nodes. These must be
        passed to the TensorRT optimization tool to identify
        input and output blobs.
    """
    exporter = CaffeExporter()
    in_names, out_names = exporter.export(
        model, prototxt_filename, model_filename, output_node_names
    )

    # Return a string instead of a list if there is only one input node.
    if len(in_names) == 1:
        in_names = in_names[0]

    # Return a string instead of a list if there is only one output node.
    if len(out_names) == 1:
        out_names = out_names[0]

    return in_names, out_names
