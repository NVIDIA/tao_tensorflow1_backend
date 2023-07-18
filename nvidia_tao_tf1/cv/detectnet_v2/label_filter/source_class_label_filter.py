# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Base label filter class that defines the interface for label filtering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.cv.detectnet_v2.label_filter.base_label_filter import BaseLabelFilter


class SourceClassLabelFilter(BaseLabelFilter):
    """Label filter that selects only those ground truth objects matching certain names."""

    def __init__(self, source_class_names=None):
        """Constructor.

        Args:
            source_class_names (list of str): Original class names to which this filter will be
                applied. If None, the filter is a no-op / not applied.
        """
        self.source_class_names = \
            set(source_class_names) if source_class_names is not None else None

    def is_criterion_satisfied_dict(self, frame_ground_truth_labels):
        """Method that implements the filter criterion as TF.ops.

        Selects on ground truth objects whose name is in self.source_class_names.

        Args:
            frame_ground_truth_labels (dict of Tensors): contains the labels for a single frame.

        Returns:
            filtered_indices (bool tf.Tensor): follows indexing in <frame_ground_truth_labels> and,
                for each element, is True if it satisfies the criterion.
        """
        source_classes = frame_ground_truth_labels['target/object_class']
        if self.source_class_names is None:
            # This means 'pass-through'.
            filtered_indices = \
                super(SourceClassLabelFilter, self).\
                is_criterion_satisfied_dict(frame_ground_truth_labels)
        else:
            # Initialize to all False.
            filtered_indices = tf.zeros_like(source_classes, dtype=tf.bool)
            # Now check individual classes.
            for object_class_name in self.source_class_names:
                filtered_indices = \
                    tf.logical_or(filtered_indices,
                                  tf.equal(source_classes, tf.constant(object_class_name)))

        return filtered_indices

    def is_criterion_satisfied_bbox_2d_label(self, ground_truth_labels):
        """Method that implements the filter criterion as TF.ops.

        Selects on ground truth objects whose name is in self.source_class_names.

        Args:
            ground_truth_labels (Bbox2DLabel): Contains the labels for all
                frames within a batch.

        Returns:
            filtered_indices (bool tf.Tensor): follows indexing in <ground_truth_labels> and,
                for each element, is True if it satisfies the criterion.
        """
        source_classes = ground_truth_labels.object_class
        if self.source_class_names is None:
            # This means 'pass-through'.
            filtered_indices = \
                super(SourceClassLabelFilter, self).\
                is_criterion_satisfied_bbox_2d_label(ground_truth_labels)
        else:
            # Initialize to all False.
            filtered_indices = tf.zeros_like(
                source_classes.values, dtype=tf.bool)
            # Now check individual classes.
            for object_class_name in self.source_class_names:
                filtered_indices = \
                    tf.logical_or(filtered_indices,
                                  tf.equal(source_classes.values, tf.constant(object_class_name)))

        return filtered_indices
