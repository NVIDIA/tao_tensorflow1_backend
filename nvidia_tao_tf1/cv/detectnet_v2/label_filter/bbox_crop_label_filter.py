# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Crop label filter. Filters ground truth objects based on cropping area."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.cv.detectnet_v2.label_filter.base_label_filter import BaseLabelFilter


class BboxCropLabelFilter(BaseLabelFilter):
    """Filter labels based on crop coordinates and bbox coordinates."""

    def __init__(self,
                 crop_left=None,
                 crop_right=None,
                 crop_top=None,
                 crop_bottom=None):
        """Constructor.

        Args:
            crop_left/crop_right/crop_top/crop_bottom (int32): crop rectangle coordinates.
                Check if the given crop coordinates constitute a valid crop rectangle. If
                any of them is None, the filter does not remove any label. If all of them
                are 0, the filter does not remove any label.

        Raises:
            ValueError: if crop_left > crop_right, or crop_top > crop_bottom, raise error.
        """
        super(BboxCropLabelFilter, self).__init__()
        # Check if the given crop coordinates constitute a valid crop rectangle.
        if any(item is None for item in [crop_left, crop_right, crop_top, crop_bottom]):
            self._valid_crop = False
        elif all(item == 0 for item in [crop_left, crop_right, crop_top, crop_bottom]):
            self._valid_crop = False
        elif crop_left < crop_right and crop_top < crop_bottom:
            self._valid_crop = True
        else:
            raise ValueError("crop_right/crop_bottom should be greater than crop_left/crop_right.")

        self._crop_left = crop_left
        self._crop_right = crop_right
        self._crop_top = crop_top
        self._crop_bottom = crop_bottom

    def is_criterion_satisfied_dict(self, frame_ground_truth_labels):
        """Method that implements the filter criterion as TF.ops.

        Only keeps those labels whose bounding boxes have overlap with crop region.

        Args:
            frame_ground_truth_labels (dict of Tensors): contains the labels for a single frame.

        Returns:
            filtered_indices (bool tf.Tensor): follows indexing in <frame_ground_truth_labels> and,
                for each element, is True if it satisfies the criterion.
        """
        filtered_indices = \
            super(BboxCropLabelFilter, self).\
            is_criterion_satisfied_dict(frame_ground_truth_labels)
        if self._valid_crop:
            crop_left = tf.cast(self._crop_left, tf.float32)
            crop_right = tf.cast(self._crop_right, tf.float32)
            crop_top = tf.cast(self._crop_top, tf.float32)
            crop_bottom = tf.cast(self._crop_bottom, tf.float32)

            # Retrieve bbox coordinates.
            x1, y1, x2, y2 = tf.unstack(frame_ground_truth_labels['target/bbox_coordinates'],
                                        axis=1)

            filtered_indices = \
                tf.logical_and(filtered_indices, tf.less(x1, crop_right))

            filtered_indices = \
                tf.logical_and(filtered_indices, tf.greater(x2, crop_left))

            filtered_indices = \
                tf.logical_and(filtered_indices, tf.less(y1, crop_bottom))

            filtered_indices = \
                tf.logical_and(filtered_indices, tf.greater(y2, crop_top))
        return filtered_indices

    def is_criterion_satisfied_bbox_2d_label(self, ground_truth_labels):
        """Method that implements the filter criterion as TF.ops.

        Only keeps those labels whose bounding boxes have overlap with crop region.

        Args:
            ground_truth_labels (Bbox2DLabel): Contains the labels for all
                frames within a batch.

        Returns:
            filtered_indices (bool tf.Tensor): follows indexing in <ground_truth_labels> and,
                for each element, is True if it satisfies the criterion.
        """
        filtered_indices = \
            super(BboxCropLabelFilter, self).\
            is_criterion_satisfied_bbox_2d_label(ground_truth_labels)
        if self._valid_crop:
            crop_left = tf.cast(self._crop_left, tf.float32)
            crop_right = tf.cast(self._crop_right, tf.float32)
            crop_top = tf.cast(self._crop_top, tf.float32)
            crop_bottom = tf.cast(self._crop_bottom, tf.float32)

            # Retrieve bbox coordinates.
            x1 = ground_truth_labels.vertices.coordinates.values[::4]
            y1 = ground_truth_labels.vertices.coordinates.values[1::4]
            x2 = ground_truth_labels.vertices.coordinates.values[2::4]
            y2 = ground_truth_labels.vertices.coordinates.values[3::4]

            filtered_indices = \
                tf.logical_and(filtered_indices, tf.less(x1, crop_right))

            filtered_indices = \
                tf.logical_and(filtered_indices, tf.greater(x2, crop_left))

            filtered_indices = \
                tf.logical_and(filtered_indices, tf.less(y1, crop_bottom))

            filtered_indices = \
                tf.logical_and(filtered_indices, tf.greater(y2, crop_top))
        return filtered_indices
