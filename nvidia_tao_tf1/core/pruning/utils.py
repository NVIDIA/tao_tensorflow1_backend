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
"""Modulus pruning utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from nvidia_tao_tf1.core.models.import_keras import keras as keras_fn
from nvidia_tao_tf1.core.models.templates.qdq_layer import QDQ
from nvidia_tao_tf1.core.models.templates.quantized_conv2d import QuantizedConv2D
from nvidia_tao_tf1.core.models.templates.quantized_dense import QuantizedDense
from nvidia_tao_tf1.core.models.templates.quantized_depthwiseconv2d import QuantizedDepthwiseConv2D

from nvidia_tao_tf1.cv.efficientdet.layers.image_resize_layer import ImageResizeLayer
from nvidia_tao_tf1.cv.efficientdet.layers.weighted_fusion_layer import WeightedFusion
from nvidia_tao_tf1.cv.efficientdet.utils.utils import PatchedBatchNormalization
from nvidia_tao_tf1.cv.faster_rcnn.layers.custom_layers import CropAndResize, ProposalTarget
from nvidia_tao_tf1.cv.mask_rcnn.layers.box_input_layer import BoxInput
from nvidia_tao_tf1.cv.mask_rcnn.layers.class_input_layer import ClassInput
from nvidia_tao_tf1.cv.mask_rcnn.layers.image_input_layer import ImageInput
from nvidia_tao_tf1.cv.mask_rcnn.layers.info_input_layer import InfoInput
from nvidia_tao_tf1.cv.mask_rcnn.layers.mask_input_layer import MaskInput

keras = keras_fn()

TRAVERSABLE_LAYERS = [
    keras.layers.BatchNormalization,
    keras.layers.Activation,
    keras.layers.Dropout,
    keras.layers.Softmax,
    keras.layers.MaxPooling2D,
    keras.layers.AveragePooling2D,
    keras.layers.Add,
    keras.layers.Subtract,
    keras.layers.Multiply,
    keras.layers.Average,
    keras.layers.Maximum,
    keras.layers.DepthwiseConv2D,
    QuantizedDepthwiseConv2D,
    keras.layers.ZeroPadding2D,
    keras.layers.ReLU, CropAndResize,
    keras.layers.TimeDistributed,
    keras.layers.LeakyReLU,
    keras.layers.UpSampling2D,
    keras.layers.Conv2D,
    QDQ, ImageResizeLayer, WeightedFusion,
    keras.layers.SeparableConv2D,
    PatchedBatchNormalization
]


def normalize_stat(stat, normalizer):
    """Normalize pruning statistics.

    Args:
        stat (Array): array of statistics to normalize
        normalizer (str): either 'L2' (normalize by dividing by L2 norm) or
            'max' (normalize by dividing by max)
    Returns:
        The normalized array.
    """
    if normalizer == "L2":
        stat = stat / np.sqrt(np.sum(stat ** 2)) * len(stat)
    elif normalizer == "max":
        stat = stat / np.max(stat)
    elif normalizer != "off":
        raise NotImplementedError("Invalid pruning normalizer: %s" % normalizer)
    return stat


def get_L2_norm(kernels, layer):
    """Get the L2 norms of the filters for pruning.

    Args:
        kernels (Array): array of kernels to compute norms of, where the last
            dimension indexes individual kernels.
        layer(keras Layer): the layer whose filters we are going to make statistics.
            Special treatment to the DepthwiseConv2D.
    Returns:
        A vector of L2 norms, one for each kernel.
    """
    if type(layer) in [keras.layers.DepthwiseConv2D, QuantizedDepthwiseConv2D] or \
        (type(layer) == keras.layers.TimeDistributed and
         type(layer.layer) in [keras.layers.DepthwiseConv2D, QuantizedDepthwiseConv2D]):
        # For DepthwiseConv2D, currently we only support depthwise_multiplier = 1.
        # I.e., axis 3 is always of size 1, kernel shape = (K_h, K_w, C_in, 1)
        norm = np.sqrt(np.sum(kernels**2, axis=(0, 1, 3)))
    else:
        norm = np.sqrt(np.sum(kernels**2, axis=tuple(range(kernels.ndim-1))))

    return norm


def find_prunable_parent(prunable_parents,
                         layer,
                         skip_root=False,
                         visited=None):
    """Recursive function to find the first prunable parent in the current branch.

    Args:
        prunable_parents (list): A list of prunable parents accumulated till before the current
            layer was explored.
        layer (keras layer object): Current layer being explored.
        skip_root(bool): Whether or not to skip the root layer(the root of the recursion tree).
            This is useful for depthwise conv case, because the current layer is prunable,
            but we want to find its parent that is prunable rather than returning itself.
    Return:
        A list of keras layers which are prunable inputs to the given layer.
    """
    visited = visited or {}

    # exit if you have encountered a prunable parent.
    if (type(layer) in [keras.layers.Conv2D,
                        keras.layers.Dense,
                        keras.layers.DepthwiseConv2D,
                        QuantizedConv2D,
                        QuantizedDepthwiseConv2D,
                        QuantizedDense]
        and len(layer._inbound_nodes) == 1) or \
        (type(layer) == keras.layers.TimeDistributed and
         type(layer.layer) in [keras.layers.Conv2D,
                               keras.layers.Dense,
                               keras.layers.DepthwiseConv2D,
                               QuantizedConv2D,
                               QuantizedDepthwiseConv2D,
                               QuantizedDense]):
        if not skip_root:
            prunable_parents.extend([layer])
            return list(set(prunable_parents))
    # If you hit a shape manipulation layer, drop an exception.
    if type(layer) not in TRAVERSABLE_LAYERS:
        raise NotImplementedError(
            "Pruning is not possible with {} layer " "in the way".format(layer.name)
        )
    # Recurse across all branches to return prunable parents.
    previous_layers = []
    inbound_nodes = layer._inbound_nodes
    # For some reason, tf.keras does not always put things in a list.
    if not isinstance(inbound_nodes, list):
        inbound_nodes = [inbound_nodes]

    for n in inbound_nodes:
        inbound_layers = n.inbound_layers
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        for l in inbound_layers:
            previous_layers.append(l)

    for l in previous_layers:
        if visited and l in visited:
            prunable_parents.extend(visited[l])
        else:
            # Skip the Input layers if there are multiple parents.
            if type(layer) == CropAndResize:
                if type(l) != ProposalTarget:
                    visited[l] = find_prunable_parent(prunable_parents,
                                                      l,
                                                      False,
                                                      visited)
                    break
                else:
                    continue
            elif type(l) not in [keras.layers.InputLayer,
                                 BoxInput, ClassInput,
                                 MaskInput, ImageInput,
                                 InfoInput]:
                visited[l] = find_prunable_parent(prunable_parents,
                                                  l,
                                                  False,
                                                  visited)
    return list(set(prunable_parents))
