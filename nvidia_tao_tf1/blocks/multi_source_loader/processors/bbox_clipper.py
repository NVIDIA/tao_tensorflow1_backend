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

"""Processor for adjusting bounding box labels after cropping."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.data_format import CHANNELS_FIRST
from nvidia_tao_tf1.blocks.multi_source_loader.data_format import CHANNELS_LAST
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor import (
    Processor,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import LABEL_OBJECT
from nvidia_tao_tf1.blocks.multi_source_loader.types import SequenceExample
from nvidia_tao_tf1.blocks.multi_source_loader.types import TransformedExample
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.types import Example


class BboxClipper(Processor):
    """Processor for adjusting bounding box labels after cropping.

    The following changes need to be made to bounding box labels:
        1) Labels completely out of the network's input are discarded.
        2) Labels that are 'half-in, half-out' should have their coordinates clipped to the input
          crop.
        3) Labels from 2) also should have their ``truncation_type`` updated accordingly.
    """

    @save_args
    def __init__(self, crop_left=0, crop_right=0, crop_top=0, crop_bottom=0):
        """Constructor.

        If all of the provided crop coordinates are or 0, this processor will amount to a no-op.

        Args:
            crop_left (int): Left-most coordinate of the crop region.
            crop_right (int): Right-most coordinate of the crop region.
            crop_top (int): Top-most coordinate of the crop region.
            crop_bottom (int): Bottom-most coordinate of the crop region.

        Raises:
            ValueError: if crop_left > crop_right, or crop_top > crop_bottom.
        """
        super(BboxClipper, self).__init__()
        self._no_op = False
        all_crop_coords = {crop_left, crop_right, crop_top, crop_bottom}
        if all_crop_coords == {0}:
            self._no_op = True

        if not self._no_op:
            if crop_left >= crop_right or crop_top >= crop_bottom:
                raise ValueError(
                    "Provided crop coordinates result in a non-sensical crop-region."
                )

        self._crop_left = float(crop_left)
        self._crop_right = float(crop_right)
        self._crop_bottom = float(crop_bottom)
        self._crop_top = float(crop_top)

    @property
    def supported_formats(self):
        """Data formats supported by this processor.

        Returns:
            data_formats (list of 'DataFormat'): Input data formats that this processor supports.
        """
        return [CHANNELS_FIRST, CHANNELS_LAST]

    def can_compose(self, other):
        """
        Determine whether two processors can be composed into a single one.

        Args:
            other (Processor): Other processor instance.

        Returns:
            (bool): True if this processor knows how to compose the other processor.
        """
        return False

    def compose(self, other):
        """Compose two processors into a single one."""
        raise NotImplementedError("BboxClipper.compose not supported.")

    def _get_indices_inside_crop(self, coords):
        """Get indices for bounding boxes that are at least partially inside the crop region.

        Args:
            coords (tf.Tensor): Float tensor of shape (N, 4) where N is the number of bounding
                boxes. Each bbox has coordinates in the order [L, T, R, B].

        Returns:
            valid_indices (tf.Tensor): Boolean tensor of shape (N,) indicating which bounding boxes
                in the input are at least partially inside the crop region.
        """
        valid_indices = tf.ones(tf.shape(input=coords)[0], dtype=tf.bool)

        # False if left-most coordinate is to the right of the crop's region.
        valid_indices = tf.logical_and(
            valid_indices, tf.less(coords[:, 0], self._crop_right)
        )
        # False if right-most coordinate is to the left of the crop's region.
        valid_indices = tf.logical_and(
            valid_indices, tf.greater(coords[:, 2], self._crop_left)
        )
        # False if top-most coordinate is to the bottom of the crop's region.
        valid_indices = tf.logical_and(
            valid_indices, tf.less(coords[:, 1], self._crop_bottom)
        )
        # False if bottom-most coordinate is to the top of the crop's region.
        valid_indices = tf.logical_and(
            valid_indices, tf.greater(coords[:, 3], self._crop_top)
        )

        return valid_indices

    def _adjust_truncation_type(self, bbox_2d_label):
        """Adjust the truncation_type of a label if it is half-in, half-out of the crop.

        Args:
            bbox_2d_label (Bbox2DLabel): Label instance for which we will update the
                truncation_type.

        Returns:
            adjusted_label (Bbox2DLabel): Adjusted version of ``bbox_2d_label``.
        """
        if isinstance(bbox_2d_label.truncation_type, tf.SparseTensor):
            new_coords = bbox_2d_label.vertices.coordinates.values
            # Get LTRB.
            x1, y1, x2, y2 = (
                new_coords[::4],
                new_coords[1::4],
                new_coords[2::4],
                new_coords[3::4],
            )

            left_most_in = tf.logical_and(
                tf.greater_equal(x1, self._crop_left),
                tf.less_equal(x1, self._crop_right),
            )
            top_most_in = tf.logical_and(
                tf.greater_equal(y1, self._crop_top),
                tf.less_equal(y1, self._crop_bottom),
            )
            right_most_in = tf.logical_and(
                tf.greater_equal(x2, self._crop_left),
                tf.less_equal(x2, self._crop_right),
            )
            bottom_most_in = tf.logical_and(
                tf.greater_equal(y2, self._crop_top),
                tf.less_equal(y2, self._crop_bottom),
            )
            # Needs adjustment if top-left corner is inside and bottom-right corner is outside, or
            # vice versa.
            half_in_half_out = tf.math.logical_xor(
                tf.logical_and(left_most_in, top_most_in),
                tf.logical_and(right_most_in, bottom_most_in),
            )

            old_truncation_type = bbox_2d_label.truncation_type
            new_truncation_type_values = tf.cast(
                tf.logical_or(
                    tf.cast(
                        old_truncation_type.values, dtype=tf.bool
                    ),  # Why is this int32??
                    half_in_half_out,
                ),
                dtype=tf.int32,
            )

            new_truncation_type = tf.SparseTensor(
                values=new_truncation_type_values,
                indices=old_truncation_type.indices,
                dense_shape=old_truncation_type.dense_shape,
            )

            return bbox_2d_label._replace(truncation_type=new_truncation_type)

        # This corresponds to the case where the `truncation_type` field is not present.
        return bbox_2d_label

    def _clip_to_crop_region(self, bbox_2d_label):
        """Clip the coordinates to the crop region.

        Args:
            bbox_2d_label (Bbox2DLabel): Label instance to clip.

        Returns:
            clipped_label (Bbox2DLabel): Clipped version of ``bbox_2d_label``.
        """
        input_coords = bbox_2d_label.vertices.coordinates.values
        xmin, ymin, xmax, ymax = (
            input_coords[::4],
            input_coords[1::4],
            input_coords[2::4],
            input_coords[3::4],
        )

        xmin = tf.clip_by_value(xmin, self._crop_left, self._crop_right)
        ymin = tf.clip_by_value(ymin, self._crop_top, self._crop_bottom)
        xmax = tf.clip_by_value(xmax, self._crop_left, self._crop_right)
        ymax = tf.clip_by_value(ymax, self._crop_top, self._crop_bottom)

        clipped_coords = tf.stack([xmin, ymin, xmax, ymax], axis=1)
        clipped_coords = tf.reshape(clipped_coords, [-1])  # Flatten.

        new_coords = tf.SparseTensor(
            values=clipped_coords,
            indices=bbox_2d_label.vertices.coordinates.indices,
            dense_shape=bbox_2d_label.vertices.coordinates.dense_shape,
        )
        new_vertices = bbox_2d_label.vertices._replace(coordinates=new_coords)

        return bbox_2d_label._replace(vertices=new_vertices)

    def _adjust_bbox_2d_label(self, bbox_2d_label):
        """Apply adjustments due to cropping to bounding box labels.

        Args:
            bbox_2d_label (Bbox2DLabel): Label instance to apply the adjustments to.

        Returns:
            adjusted_label (Bbox2DLabel): Adjusted version of ``bbox_2d_label``.
        """
        input_coords = bbox_2d_label.vertices.coordinates.values
        # For convenience, reshape input coordinates.
        input_coords = tf.reshape(input_coords, [-1, 4])  # Order is L, T, R, B.

        # First, figure out which ones are completely outside the crop.
        valid_indices = self._get_indices_inside_crop(input_coords)

        adjusted_label = bbox_2d_label.filter(valid_indices)

        # Now, determine, which ones need to have their coordinates clipped and truncation_type
        # adjusted.
        adjusted_label = self._adjust_truncation_type(adjusted_label)
        adjusted_label = self._clip_to_crop_region(adjusted_label)

        return adjusted_label

    def process(self, example):
        """
        Process an example.

        Args:
            example (Example): Example with frames in format specified by data_format.

        Returns:
            (Example): Processed example.

        Raises:
            ValueError: Since this processor explicitly needs to be applied after transformations
                (if they are present), it does not accept TransformedExample.
        """
        if isinstance(example, TransformedExample):
            raise ValueError(
                "BboxClipper should be applied on labels that have been transformed."
            )

        if not self._no_op:
            if isinstance(example, (Example, SequenceExample)):
                if LABEL_OBJECT in example.labels:
                    example.labels[LABEL_OBJECT] = self._adjust_bbox_2d_label(
                        bbox_2d_label=example.labels[LABEL_OBJECT]
                    )

        return example
