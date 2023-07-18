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

"""DetectNet V2 visualization utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import keras
import tensorflow as tf

from nvidia_tao_tf1.cv.common.visualizer.base_visualizer import BaseVisualizer, Descriptor


logger = logging.getLogger(__name__)


class TensorBoardVisualizer(BaseVisualizer):
    """Visualizer implemented as a static class."""

    backend = Descriptor("_backend")

    @classmethod
    def build(cls, enabled, num_images):
        """Build the TensorBoard Visualizer.

        Args:
            enabled (bool): Boolean value to show that the Visualizer was enabled.
            num_images (int): Number of images to be visualized.
        """
        cls._backend = "tensorboard"
        super().build(
            enabled, num_images=num_images
        )

    @classmethod
    def build_from_config(cls, visualizer_config):
        """Build visualizer from config.

        Arguments:
            visualizer_config (visualizer_config_pb2.VisualizerConfig).
        """
        enabled = visualizer_config.enabled
        num_images = visualizer_config.num_images
        cls.build(
            enabled,
            num_images
        )

    @classmethod
    def image(cls, name, value, value_range=None,
              data_format='channels_first', collections=None):
        """Add a 4D tensor to Tensorboard.

        Args:
            name (string): Image name.
            value: 4D tensor in NCHW or NHWC format.
            value_range (float, float): Black and white-point for mapping value to color.
                None means use TF automatic scaling
            data_format (string): Format of the input values. Must be either 'channels_first' (NCHW)
                or 'channels_last' (NHWC).
            collections: Optional list of ops.GraphKeys. The collections to add the summary to.
        """
        if cls.enabled and cls.num_images > 0:
            tensor_name = f"image/{name}"
            # Optionally apply scaling and offseting and cast to uint8.
            # By default rely on Tensorflow normalization.
            if value_range is not None:
                # Compute scale and offset such that black-point maps to 0.0
                # and white-point maps to 255.0
                black, white = value_range
                scale = 255.0 / (white - black)
                offset = 255.0 * black / (black - white)
                value = tf.cast(tf.clip_by_value(value * scale + offset, 0., 255.), tf.uint8)

            # Images must be in NHWC format. Convert as needed.
            if data_format == 'channels_first':
                value = tf.transpose(value, (0, 2, 3, 1))
            tf.summary.image(tensor_name, value[:, :, :, :3], cls.num_images,
                             collections=collections)

    @classmethod
    def histogram(cls, tensor_name, tensor_value, collections=None):
        """Visualize histogram for a given tensor."""
        tensor_name = f"histogram/{tensor_name}"
        tf.summary.histogram(name=tensor_name, values=tensor_value, collections=collections)

    @classmethod
    def keras_model_weight_histogram(cls, keras_model, collections=None):
        """Add model weight histogram to tensorboard summary.

        Args:
            model: Keras model.
            collections: Optional list of ops.GraphKeys. The collections to add the summary to.
        """
        for layer in keras_model.layers:
            if isinstance(layer, keras.engine.training.Model):
                cls.keras_model_weight_histogram(layer, collections)
            if isinstance(layer, keras.layers.convolutional.DepthwiseConv2D):
                # Plot histogram of conv layer weight.
                cls.histogram(
                    tensor_name=f'weight/{layer.name}',
                    tensor_value=layer.depthwise_kernel,
                    collections=collections
                )
            elif isinstance(layer, keras.layers.convolutional.Conv2D):
                # Plot histogram of conv layer weight.
                cls.histogram(
                    tensor_name=f'weight/{layer.name}',
                    tensor_value=layer.kernel,
                    collections=collections
                )
            elif isinstance(layer, keras.layers.normalization.BatchNormalization):
                # Plot histogram of gamma and beta.
                cls.histogram(
                    tensor_name=f'gamma/{layer.name}',
                    tensor_value=layer.gamma,
                    collections=collections
                )
                cls.histogram(
                    tensor_name=f'beta/{layer.name}',
                    tensor_value=layer.beta,
                    collections=collections
                )
