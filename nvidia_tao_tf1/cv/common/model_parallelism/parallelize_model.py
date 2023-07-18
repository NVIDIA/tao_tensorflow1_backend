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
"""Utility to parallelize a Keras model for model parallelism in training."""

import keras
import numpy as np
import tensorflow as tf


def find_segment_idx(layer_idx, layer_splits):
    """Find the segment index for a given layer index."""
    idx = 0
    while not (layer_splits[idx] <= layer_idx < layer_splits[idx+1]):
        idx += 1
    assert idx < len(layer_splits) - 1, (
        "Segment index should be less than {}".format(len(layer_splits) - 1)
    )
    return idx


def model_parallelism(
    model,
    parallelism,
    freeze_bn=False
):
    """Split the model into several parts on multiple GPUs for model parallelism."""
    # set training=False for BN layers if freeze_bn=True
    # otherwise the freeze_bn flag in model builder will be ineffective
    def compose_call(prev_call_method):
        def call(self, inputs, training=False):
            return prev_call_method(self, inputs, training)

        return call

    prev_batchnorm_call = keras.layers.normalization.BatchNormalization.call
    if freeze_bn:
        keras.layers.normalization.BatchNormalization.call = compose_call(
            prev_batchnorm_call
        )
    world_size = len(parallelism)
    # in case that model parallelism is not enabled at all...
    if world_size == 0:
        world_size = 1
        parallelism = (1.0,)
    p_arr = np.array((0.0,) + parallelism, dtype=np.float32)
    cum_p_arr = np.cumsum(p_arr)
    # splitting points for each segment of the model
    splits = cum_p_arr / cum_p_arr[-1]
    layer_splits = np.round(splits * len(model.layers))
    layer_idx = 0
    _explored_layers = dict()
    for l in model.layers:
        _explored_layers[l.name] = [False, None]
    input_layer = [l for l in model.layers if (type(l) == keras.layers.InputLayer)]
    layers_to_explore = input_layer
    model_outputs = {}
    # Loop until we reach the last layer.
    while layers_to_explore:
        layer = layers_to_explore.pop(0)
        # Skip layers that may be revisited in the graph to prevent duplicates.
        if not _explored_layers[layer.name][0]:
            # Check if all inbound layers explored for given layer.
            if not all([
                    _explored_layers[l.name][0]
                    for n in layer._inbound_nodes
                    for l in n.inbound_layers
                    ]):
                continue
            outputs = None
            # Visit input layer.
            if type(layer) == keras.layers.InputLayer:
                # Re-use the existing InputLayer.
                outputs = layer.output
                new_layer = layer
            else:
                gpu_idx = find_segment_idx(layer_idx, layer_splits)
                layer_idx += 1
                # pin this layer on a certain GPU
                with tf.device("/gpu:{}".format(gpu_idx)):
                    # Create new layer.
                    layer_config = layer.get_config()
                    new_layer = type(layer).from_config(layer_config)
                    # Add to model.
                    outputs = []
                    for node in layer._inbound_nodes:
                        prev_outputs = []
                        for idx, l in enumerate(node.inbound_layers):
                            keras_layer = _explored_layers[l.name][1]
                            prev_outputs.append(keras_layer.get_output_at(node.node_indices[idx]))
                        assert prev_outputs, "Expected non-input layer to have inputs."
                        if len(prev_outputs) == 1:
                            prev_outputs = prev_outputs[0]
                        outputs.append(new_layer(prev_outputs))
                    if len(outputs) == 1:
                        outputs = outputs[0]
                    weights = layer.get_weights()
                    if weights is not None:
                        new_layer.set_weights(weights)
            outbound_nodes = layer._outbound_nodes
            if not outbound_nodes:
                model_outputs[layer.output.name] = outputs
            layers_to_explore.extend([node.outbound_layer for node in outbound_nodes])
            # Mark current layer as visited and assign output nodes to the layer.
            _explored_layers[layer.name][0] = True
            _explored_layers[layer.name][1] = new_layer
        else:
            continue
    output_tensors = [model_outputs[l.name] for l in model.outputs if l.name in model_outputs]
    new_model = keras.models.Model(inputs=model.inputs,
                                   outputs=output_tensors,
                                   name=model.name)
    # restore the BN call method before return
    if freeze_bn:
        keras.layers.normalization.BatchNormalization.call = prev_batchnorm_call
    return new_model
