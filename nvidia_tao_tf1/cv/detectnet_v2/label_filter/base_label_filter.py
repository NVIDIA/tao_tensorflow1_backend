# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Base label filter class that defines the interface for label filtering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types.bbox_2d_label import Bbox2DLabel


class BaseLabelFilter(object):
    """Label filter base class defining the interface for selection / filtering."""

    def __init__(self):
        """Constructor."""

    def is_criterion_satisfied_bbox_2d_label(self, ground_truth_labels):
        """Method that implements the filter criterion as TF.ops.

        The base class's filter is a pass-through layer.

        Args:
            ground_truth_labels (Bbox2DLabel): Contains the labels for all
                frames within a batch.

        Returns:
            filtered_indices (bool tf.Tensor): follows indexing in <ground_truth_labels> and,
                for each element, is True if it satisfies the criterion.
        """
        source_classes = ground_truth_labels.object_class
        filtered_indices = tf.ones_like(source_classes.values, dtype=tf.bool)

        return filtered_indices

    def is_criterion_satisfied_dict(self, frame_ground_truth_labels):
        """Method that implements the filter criterion as TF.ops.

        The base class's filter is a pass-through layer.

        Args:
            frame_ground_truth_labels (dict of Tensors): contains the labels for a single frame.

        Returns:
            filtered_indices (bool tf.Tensor): follows indexing in <frame_ground_truth_labels> and,
                for each element, is True if it satisfies the criterion.
        """
        filtered_indices = \
            tf.ones_like(
                frame_ground_truth_labels['target/object_class'], dtype=tf.bool)

        return filtered_indices

    def is_criterion_satisfied(self, frame_ground_truth_labels):
        """Method that implements the filter criterion as TF.ops.

        The base class's filter is a pass-through layer.

        Args:
            frame_ground_truth_labels (dict of Tensors): contains the labels for a single frame.

        Returns:
            filtered_indices (bool tf.Tensor): follows indexing in <frame_ground_truth_labels> and,
                for each element, is True if it satisfies the criterion.
        """
        filtered_indices = None
        if isinstance(frame_ground_truth_labels, dict):
            filtered_indices = self.is_criterion_satisfied_dict(
                frame_ground_truth_labels)
        elif isinstance(frame_ground_truth_labels, Bbox2DLabel):
            filtered_indices = self.is_criterion_satisfied_bbox_2d_label(
                frame_ground_truth_labels)
        else:
            raise ValueError("Unsupported type.")
        return filtered_indices


def filter_labels(ground_truth_labels, indices):
    """Filter ground truth labels according to indices indicating criterion satisfaction.

    Args:
        frame_ground_truth_labels (dict of Tensors): contains the labels for a single frame.
        indices (bool tf.Tensor): follows indexing of <frame_ground_truth_labels> and indicates
            for each element whether the criterion has been met.

    Returns:
        filtered_ground_truth_labels (dict of Tensors): contains the same fields as the input,
            but keeps only those examples that satisfy the criterion.
    """
    filtered_ground_truth_labels = {}
    if isinstance(ground_truth_labels, dict):
        filtered_ground_truth_labels = filter_labels_dict(
            ground_truth_labels, indices)
    elif isinstance(ground_truth_labels, Bbox2DLabel):
        filtered_ground_truth_labels = ground_truth_labels.filter(indices)
    else:
        raise ValueError("Unsupported type.")
    return filtered_ground_truth_labels


def filter_labels_dict(frame_ground_truth_labels, indices):
    """Filter ground truth labels according to indices indicating criterion satisfaction.

    Args:
        frame_ground_truth_labels (dict of Tensors): contains the labels for a single frame.
        indices (bool tf.Tensor): follows indexing of <frame_ground_truth_labels> and indicates
            for each element whether the criterion has been met.

    Returns:
        filtered_ground_truth_labels (dict of Tensors): contains the same fields as the input,
            but keeps only those examples that satisfy the criterion.
    """
    coordinate_mask = None
    filtered_ground_truth_labels = dict()
    # Filter target features based on the valid indices. Other features are left as is.
    for feature_name, feature_tensor in six.iteritems(frame_ground_truth_labels):
        # Features which are mapped by index need a gathered mask.
        if feature_name.startswith('target/coordinates/'):
            coordinate_mask = coordinate_mask if coordinate_mask is not None else \
                tf.gather(
                    indices, frame_ground_truth_labels['target/coordinates/index'])
            if feature_name == 'target/coordinates/index':
                # Broadcast the boolean mask to the index of all polygons, then ensure that the
                #  index contains no ordinal greater than its count.
                masked_tensor = tf.unique(tf.boolean_mask(
                    feature_tensor, coordinate_mask))[1]
            else:
                # Broadcast the boolean mask to the vertices of all polygons.
                masked_tensor = tf.boolean_mask(
                    feature_tensor, coordinate_mask)
            filtered_ground_truth_labels[feature_name] = masked_tensor
        elif feature_name.startswith('target/'):
            # TODO(@williamz): when TF >= 1.5, use the 'axis' kwarg.
            filtered_ground_truth_labels[feature_name] = \
                tf.boolean_mask(feature_tensor, indices)
        else:
            filtered_ground_truth_labels[feature_name] = feature_tensor

    return filtered_ground_truth_labels


def get_chained_filters_indices(label_filters, frame_ground_truth_labels, mode=None):
    """Helper function that returns the boolean mask of the filters via logical-or or logical-and.

    Args:
        label_filters (list): Each element is an instance of any of BaseLabelFilter's child classes.
        frame_ground_truth_labels (dict of Tensors): contains the labels for a single frame.
        mode (str): How to chain all the elements in <label_filters>. Supported modes are 'or'
            and 'and'. Note that it is ignored when there is only one item in <label_filters>.

    Returns:
        filtered_indices (bool tf.Tensor): Follows indexing in <frame_ground_truth_labels> and,
            for each element, is True if it satisfies one of the criteria in <label_filters>.

    Raises:
        AssertionError: If <label_filters> is empty.
        ValueError: If <label_filters> has more than 1 filters but mode parameter is not valid.
    """
    assert label_filters, "Please provide at least one filter."
    # Get a list where each element is a tf.Tensor of bool values for the corresponding entry in
    #  in <label_filters>.
    filtered_indices_list = [
        x.is_criterion_satisfied(frame_ground_truth_labels) for x in label_filters]

    if len(label_filters) == 1:
        # No logical operation needed.
        filtered_indices = filtered_indices_list[0]
    elif mode == 'or':
        # Apply them as a logical-or.
        filtered_indices = tf.reduce_any(tf.stack(filtered_indices_list), axis=0)
    elif mode == 'and':
        # Apply them as a logical-and.
        filtered_indices = tf.reduce_all(tf.stack(filtered_indices_list), axis=0)
    else:
        # When using multiple filters, mode parameter is necessary.
        raise ValueError("Mode should be either 'or' or 'and' when filter number > 1.")

    return filtered_indices


def apply_label_filters(label_filters, ground_truth_labels, mode=None):
    """Apply multiple label filters via using user-specified mode to labels.

    Args:
        label_filters (list): Each element is an instance of any of BaseLabelFilter's child classes.
        ground_truth_labels (dict of Tensors or Bbox2DLabel):
            dict of Tensors: contains the labels for a single frame.
            Bbox2DLabel: contains bboxes for all frames in a batch.
        mode (str): How to chain all the elements in <label_filters>. Supported modes are 'or'
            and 'and'. Note that it is ignored when there is only one item in <label_filters>.

    Returns:
        filtered_ground_truth_labels (dict of Tensors): Contains the same fields as the input,
            but keeps only those examples that satisfy the criterion.
    """
    # Get filtered indices.
    filtered_indices = get_chained_filters_indices(label_filters, ground_truth_labels, mode)
    # Now apply the boolean mask.
    filtered_ground_truth_labels = filter_labels(ground_truth_labels, filtered_indices)

    return filtered_ground_truth_labels
