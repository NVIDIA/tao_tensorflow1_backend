# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Height label filter. Filters ground truth objects based on their height."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.cv.detectnet_v2.label_filter.base_label_filter import BaseLabelFilter


class BboxDimensionsLabelFilter(BaseLabelFilter):
    """Filter labels based on bounding box dimension thresholds."""

    def __init__(self,
                 min_width=None,
                 min_height=None,
                 max_width=None,
                 max_height=None):
        """Constructor.

        Args:
            min/max_width/height (float): Thresholds above/below which to keep bounding box objects.
                If None, the corresponding threshold is not used.
        """
        super(BboxDimensionsLabelFilter, self).__init__()
        self.min_width = min_width
        self.min_height = min_height
        self.max_width = max_width
        self.max_height = max_height
        # Check bounds if necessary.
        if self.min_width is not None and self.max_width is not None:
            assert self.min_width < self.max_width, "max_width should be greater than min_width."
        if self.min_height is not None and self.max_height is not None:
            assert self.min_height < self.max_height, \
                "max_height should be greater than min_height."

    def is_criterion_satisfied_dict(self, frame_ground_truth_labels):
        """Method that implements the filter criterion as TF.ops.

        Only keeps those labels whose bounding boxes' width is in [self.min_width, self.max_width]
        and height is in [self.min_height, self.max_height].

        Args:
            frame_ground_truth_labels (dict of Tensors): contains the labels for a single frame.

        Returns:
            filtered_indices (bool tf.Tensor): follows indexing in <frame_ground_truth_labels> and,
                for each element, is True if it satisfies the criterion.
        """
        filtered_indices = \
            super(BboxDimensionsLabelFilter, self).\
            is_criterion_satisfied_dict(frame_ground_truth_labels)
        if {self.min_width, self.min_height, self.max_width, self.max_height} != {None}:
            # Retrieve labels' width and height.
            x1, y1, x2, y2 = tf.unstack(frame_ground_truth_labels['target/bbox_coordinates'],
                                        axis=1)
            width = x2 - x1
            height = y2 - y1
            # Chain the constraints.
            if self.min_width is not None:
                filtered_indices = \
                    tf.logical_and(filtered_indices,
                                   tf.greater_equal(width, tf.constant(self.min_width)))
            if self.max_width is not None:
                filtered_indices = tf.logical_and(filtered_indices,
                                                  tf.less_equal(width, tf.constant(self.max_width)))
            if self.min_height is not None:
                filtered_indices = tf.logical_and(filtered_indices,
                                                  tf.greater_equal(height,
                                                                   tf.constant(self.min_height)))
            if self.max_height is not None:
                filtered_indices = \
                    tf.logical_and(filtered_indices,
                                   tf.less_equal(height, tf.constant(self.max_height)))
        return filtered_indices

    def is_criterion_satisfied_bbox_2d_label(self, ground_truth_labels):
        """Method that implements the filter criterion as TF.ops.

        Only keeps those labels whose bounding boxes' width is in [self.min_width, self.max_width]
        and height is in [self.min_height, self.max_height].

        Args:
            ground_truth_labels (Bbox2DLabel): Contains the labels for all
                frames within a batch.

        Returns:
            filtered_indices (bool tf.Tensor): follows indexing in <ground_truth_labels> and,
                for each element, is True if it satisfies the criterion.
        """
        filtered_indices = \
            super(BboxDimensionsLabelFilter, self).\
            is_criterion_satisfied_bbox_2d_label(ground_truth_labels)
        if {self.min_width, self.min_height, self.max_width, self.max_height} != {None}:
            # Retrieve labels' width and height.
            x1 = ground_truth_labels.vertices.coordinates.values[::4]
            y1 = ground_truth_labels.vertices.coordinates.values[1::4]
            x2 = ground_truth_labels.vertices.coordinates.values[2::4]
            y2 = ground_truth_labels.vertices.coordinates.values[3::4]
            width = x2 - x1
            height = y2 - y1
            # Chain the constraints.
            if self.min_width is not None:
                filtered_indices = \
                    tf.logical_and(filtered_indices,
                                   tf.greater_equal(width, tf.constant(self.min_width)))
            if self.max_width is not None:
                filtered_indices = tf.logical_and(filtered_indices,
                                                  tf.less_equal(width, tf.constant(self.max_width)))
            if self.min_height is not None:
                filtered_indices = tf.logical_and(filtered_indices,
                                                  tf.greater_equal(height,
                                                                   tf.constant(self.min_height)))
            if self.max_height is not None:
                filtered_indices = \
                    tf.logical_and(filtered_indices,
                                   tf.less_equal(height, tf.constant(self.max_height)))
        return filtered_indices
