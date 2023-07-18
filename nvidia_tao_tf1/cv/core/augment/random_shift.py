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
"""Processor for applying random shift transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.core.processors.processors import Processor

import tensorflow as tf


class RandomShift(Processor):
    """Random shift processor to shift bounding boxes."""

    def __init__(self, shift_percent_max, shift_probability, frame_shape, **kwargs):
        """Construct a random blur processor.

        Args:
            shift_percent_max (float): Maximum percent shift of bounding box
            shift_probability (float): Probability that a shift will occur.
            frame_shape (float): shape of frame (HWC).
        """
        super(RandomShift, self).__init__(**kwargs)
        if shift_percent_max < 0.0 or shift_percent_max > 1.0:
            raise ValueError(
                "RandomShift.shift_percent_max ({}) is not within the range [0, 1].".format(
                    shift_percent_max))

        if shift_probability < 0.0 or shift_probability > 1.0:
            raise ValueError(
                "RandomShift.shift_probability ({}) is not within the range [0, 1].".format(
                    shift_probability))

        self._shift_percent_max = shift_percent_max
        self._shift_probability = shift_probability
        self._frame_shape = frame_shape

    def __repr__(self):
        """Return a string representation of the processor."""
        return "RandomShift(shift_percent_max={}, shift_probability={})".format(
            self._shift_percent_max, self._shift_probability)

    def _build(self, *args, **kwargs):
        """Initialize random variables used for op.

        The build function should be used when wanting to apply a consistent random shift to
        multiple images. This means if a random shift is performed on one image, a random shift
        will occur on other images passed through this Processor. The shift amount may vary.
        """
        shift_probability = tf.random_uniform([], minval=0.0, maxval=1.0)
        should_shift = tf.less(shift_probability, self._shift_probability)
        self._percentage = tf.cond(should_shift, lambda: self._shift_percent_max, lambda: 0.0)

    def call(self, in_bbox):
        """Return a shifted bounding box.

        Args:
            in_bbox (dict): contains 'x', 'y', 'w', 'h' information for bounding box.

        Returns:
            bbox (dict): contains modified 'x', 'y', 'w', 'h' information for bounding
                                      box.
        """
        if (self._shift_percent_max == 0.0 or self._shift_probability == 0.0):
            return in_bbox

        bbox = {}
        for key in in_bbox:
            bbox[key] = tf.identity(in_bbox[key])

        # x shift is relative to width of bbox.
        bound = bbox['w'] * self._percentage
        coord_noise = tf.random_uniform([], minval=-1.0, maxval=1.0, dtype=tf.float32) * bound
        bbox['x'] += coord_noise

        # y shift is relative to width of bbox.
        bound = bbox['h'] * self._percentage
        coord_noise = tf.random_uniform([], minval=-1.0, maxval=1.0, dtype=tf.float32) * bound
        bbox['y'] += coord_noise

        # NOTE: to preserve square bbox, the same shift is applied to width and height.
        bound = bbox['w'] * self._percentage
        square_preserve = tf.random_uniform([], minval=-1.0, maxval=1.0, dtype=tf.float32) * bound
        bbox['w'] += square_preserve
        bbox['h'] += square_preserve

        bbox['x'] = tf.reshape(bbox['x'], ())
        bbox['y'] = tf.reshape(bbox['y'], ())
        bbox['w'] = tf.reshape(bbox['w'], ())
        bbox['h'] = tf.reshape(bbox['h'], ())
        return self._correct_bbox_bounds(bbox, self._frame_shape)

    @staticmethod
    def _correct_bbox_bounds(bbox, frame_shape):
        """Fix bounding box coordinates within shape of frame.

        Args:
            bbox (dict): contains 'x', 'y', 'w', 'h' information for bounding box.
            frame_shape (Tensor float32): shape of frame (HWC).

        Returns:
            bbox (dict): contains 'x', 'y', 'w', 'h' information for bounding box within
                         frame shape.
        """
        frame_h = frame_shape[0] - 1
        frame_w = frame_shape[1] - 1

        bbox['x'] = tf.clip_by_value(bbox['x'], clip_value_min=0, clip_value_max=frame_w)
        bbox['y'] = tf.clip_by_value(bbox['y'], clip_value_min=0, clip_value_max=frame_h)
        width_large = tf.greater(bbox['x'] + bbox['w'], frame_w)
        height_large = tf.greater(bbox['y'] + bbox['h'], frame_h)

        new_width = tf.cond(width_large, lambda: frame_w - bbox['x'], lambda: bbox['w'])
        new_height = tf.cond(height_large, lambda: frame_h - bbox['y'], lambda: bbox['h'])

        max_square_dim = tf.minimum(new_width, new_height)
        bbox['w'] = tf.clip_by_value(bbox['w'], clip_value_min=0, clip_value_max=max_square_dim)
        bbox['h'] = tf.clip_by_value(bbox['h'], clip_value_min=0, clip_value_max=max_square_dim)

        return bbox
