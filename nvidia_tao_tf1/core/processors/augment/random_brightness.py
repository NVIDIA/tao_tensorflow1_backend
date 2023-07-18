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
"""Processor for applying random brightness transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.augment import color
from nvidia_tao_tf1.core.processors.processors import Processor
from nvidia_tao_tf1.core.types import Transform


class RandomBrightness(Processor):
    """Random brightness transform."""

    @save_args
    def __init__(self, scale_max, uniform_across_channels, **kwargs):
        """Construct a RandomBrightness processor.

        Args:
            scale_max (float): The range of the brightness offsets. This value
                is half of the standard deviation, where values of twice the standard
                deviation are truncated. A value of 0 (default) will not affect the matrix.
            uniform_across_channels (bool): If true will apply the same brightness
                shift to all channels. If False, will apply a different brightness shift to each
                channel.
            kwargs (dict): keyword arguments passed to parent class.
        """
        super(RandomBrightness, self).__init__(**kwargs)
        self._scale_max = scale_max
        self._uniform_across_channels = uniform_across_channels

    def __repr__(self):
        """Return a string representation of the processor."""
        return "RandomBrightness(scale_max={}, uniform_across_channels={})".format(
            self._scale_max, self._uniform_across_channels
        )

    def call(self, transform):
        """Return a Transform whose color transformation matrix is perturbed at random.

        Args:
            transform (Transform): An input Transform instance to be processed.

        Returns:
            Transform: Final Transform instance with color transform matrix perturbed.
        """
        if not isinstance(transform, Transform):
            raise TypeError(
                "Expecting an argument of type 'Transform', "
                "given: {}.".format(type(transform).__name__)
            )
        batch_size = None
        if transform.color_transform_matrix.shape.ndims == 3:
            batch_size = tf.shape(input=transform.color_transform_matrix)[0]
        ctm_brightness = color.random_brightness_matrix(
            brightness_scale_max=self._scale_max,
            brightness_uniform_across_channels=self._uniform_across_channels,
            batch_size=batch_size,
        )
        processed_ctm = tf.matmul(ctm_brightness, transform.color_transform_matrix)

        return Transform(
            canvas_shape=transform.canvas_shape,
            color_transform_matrix=processed_ctm,
            spatial_transform_matrix=transform.spatial_transform_matrix,
        )
