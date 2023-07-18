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
"""Processor for applying random glimpse transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.augment import spatial
from nvidia_tao_tf1.core.processors.processors import Processor
from nvidia_tao_tf1.core.types import Canvas2D, Transform


class RandomGlimpse(Processor):
    """Processor for extracting random glimpses of images and labels."""

    # Always crop the center region.
    CENTER = "center"
    # Crop at random location keeping the cropped region within original image bounds.
    RANDOM = "random"
    CROP_LOCATIONS = [CENTER, RANDOM]

    @save_args
    def __init__(self, height, width, crop_location, crop_probability, **kwargs):
        """Construct a RandomGlimpse processor.

        Args:
            height (int): New height to which contents will be either cropped or scaled down to.
            width (int): New width to which contents will be either cropper or scaled down to.
            crop_location (str): Enumeration specifying how the crop location is selected.
            crop_probability (float): Probability at which a crop is performed.
            kwargs (dict): keyword arguments passed to parent class.
        """
        super(RandomGlimpse, self).__init__(**kwargs)
        if crop_location not in RandomGlimpse.CROP_LOCATIONS:
            raise ValueError(
                "RandomGlimpse.crop_location '{}' is not supported. Valid options: {}.".format(
                    crop_location, ", ".join(RandomGlimpse.CROP_LOCATIONS)
                )
            )
        if crop_probability < 0.0 or crop_probability > 1.0:
            raise ValueError(
                "RandomGlimpse.crop_probability ({}) is not within the range [0, 1].".format(
                    crop_probability
                )
            )
        if height <= 0:
            raise ValueError(
                "RandomGlimpse.height ({}) is not positive.".format(height)
            )
        if width <= 0:
            raise ValueError("RandomGlimpse.width ({}) is not positive.".format(width))

        self._height = height
        self._width = width
        self._crop_location = crop_location
        self._crop_probability = crop_probability

    def __repr__(self):
        """Return a string representation of the processor."""
        return "RandomGlimpse(height={}, width={}, crop_location={}, crop_probability={})".format(
            self._height, self._width, self._crop_location, self._crop_probability
        )

    def call(self, transform):
        """Return a Transform that either crops or scales, always producing same sized output.

        Args:
            transform (Transform): An input Transform instance to be processed.

        Returns:
            Transform: Final Transform instance with either cropping or scaling applied.
        """
        if not isinstance(transform, Transform):
            raise TypeError(
                "Expecting an argument of type 'Transform', "
                "given: {}.".format(type(transform).__name__)
            )

        input_height = transform.canvas_shape.height
        input_width = transform.canvas_shape.width

        batch_size = None
        batch_shape = []
        if transform.spatial_transform_matrix.shape.ndims == 3:
            batch_size = tf.shape(input=transform.spatial_transform_matrix)[0]
            batch_shape = [batch_size]
        crop_probability = tf.random.uniform(batch_shape, minval=0.0, maxval=1.0)
        should_crop = tf.less_equal(crop_probability, self._crop_probability)
        glimpse_stm = tf.compat.v1.where(
            should_crop,
            self._crop(
                input_height=input_height,
                input_width=input_width,
                batch_size=batch_size,
            ),
            self._scale(
                input_height=input_height,
                input_width=input_width,
                batch_size=batch_size,
            ),
        )
        processed_stm = tf.matmul(glimpse_stm, transform.spatial_transform_matrix)

        return Transform(
            canvas_shape=Canvas2D(height=self._height, width=self._width),
            color_transform_matrix=transform.color_transform_matrix,
            spatial_transform_matrix=processed_stm,
        )

    def _scale(self, input_width, input_height, batch_size):
        """Return a spatial transform matrix for scaling inputs to requested height and width."""
        horizontal_ratio = input_width / self._width
        vertical_ratio = input_height / self._height
        stm = spatial.zoom_matrix(ratio=(horizontal_ratio, vertical_ratio))
        if batch_size is not None:
            stm = tf.broadcast_to(stm, [batch_size, 3, 3])
        return stm

    def _crop(self, input_height, input_width, batch_size):
        """Return a spatial transform matrix that crops a section of desired height and width."""
        if self._crop_location == RandomGlimpse.RANDOM:
            return self._random_crop(
                input_height=input_height,
                input_width=input_width,
                batch_size=batch_size,
            )
        if self._crop_location == RandomGlimpse.CENTER:
            return self._center_crop(
                input_height=input_height,
                input_width=input_width,
                batch_size=batch_size,
            )
        raise ValueError("Unhandled crop location: '{}'.".format(self._crop_location))

    def _random_crop(self, input_height, input_width, batch_size):
        """Return a STM that crops a random location contained within the input canvas."""
        min_left_x = 0
        max_left_x = input_width - self._width
        if max_left_x < 0.0:
            raise ValueError(
                "Attempted to extract random crop ({}) wider than input width ({}).".format(
                    self._width, input_width
                )
            )

        min_top_y = 0
        max_top_y = input_height - self._height
        if max_top_y < 0.0:
            raise ValueError(
                "Attempted to extract random crop ({}) taller than input height ({}).".format(
                    self._height, input_height
                )
            )

        batch_shape = [] if batch_size is None else [batch_size]
        left_x = tf.random.uniform(batch_shape, minval=min_left_x, maxval=max_left_x)
        top_y = tf.random.uniform(batch_shape, minval=min_top_y, maxval=max_top_y)
        return spatial.translation_matrix(x=left_x, y=top_y)

    def _center_crop(self, input_height, input_width, batch_size):
        """Return a STM that crops a vertically and horizontally centered section of the canvas."""
        horizontal_space = input_width - self._width
        if horizontal_space < 0.0:
            raise ValueError(
                "Attempted to extract center crop ({}) wider than input width ({}).".format(
                    self._width, input_width
                )
            )

        vertical_space = input_height - self._height
        if vertical_space < 0.0:
            raise ValueError(
                "Attempted to extract center crop ({}) taller than input height ({}).".format(
                    self._height, input_height
                )
            )

        left_x = horizontal_space // 2
        top_y = vertical_space // 2
        stm = spatial.translation_matrix(x=left_x, y=top_y)
        if batch_size is not None:
            stm = tf.broadcast_to(stm, [batch_size, 3, 3])
        return stm
