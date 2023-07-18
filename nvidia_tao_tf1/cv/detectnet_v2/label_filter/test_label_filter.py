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

"""Test ground truth label filters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader import types
import nvidia_tao_tf1.core
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.base_label_filter import apply_label_filters
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.base_label_filter import BaseLabelFilter
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.bbox_crop_label_filter import BboxCropLabelFilter
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.bbox_dimensions_label_filter import (
    BboxDimensionsLabelFilter
)
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.source_class_label_filter import (
    SourceClassLabelFilter
)

Canvas2D = nvidia_tao_tf1.core.types.Canvas2D

# TODO(@williamz): Similar things appear in dataloader/default_dataloader.py. If we need constants,
#  surely there is a better place / way to have them?
OBJECT_CLASS_KEY = 'target/object_class'


def get_dummy_labels(source_class_names, other_attributes=None):
    """Create a set of TF constant variables that will act as labels.

    Args:
        source_class_names (list of str): List of object class names to be used for the target
            labels.
        other_attributes (dict): Key value pairs with which to update the label examples. e.g.
            {'target/coordinates_x1': ...}

    Returns:
        frame_ground_truth_labels (dict): Contains a single key 'target/object_class' to which
            a tf.constant tensor with the input <source_class_names> is mapped.
    """
    frame_ground_truth_labels = dict()
    frame_ground_truth_labels[OBJECT_CLASS_KEY] = tf.constant(
        source_class_names)

    if other_attributes is not None:
        for attribute_name, attribute_values in other_attributes.items():
            frame_ground_truth_labels[attribute_name] = tf.constant(
                attribute_values)

    return frame_ground_truth_labels


def _get_bbox_2d_labels():
    """Bbox2DLabel for test preparation."""
    frame_indices = [0, 1, 2, 3, 3]
    object_class = tf.constant(
        ['pedestrian', 'unmapped', 'automobile', 'truck', 'truck'])
    bbox_coordinates = tf.constant(
        [7.0, 6.0, 8.0, 9.0,
         2.0, 3.0, 4.0, 5.0,
         0.0, 0.0, 3.0, 4.0,
         1.2, 3.4, 5.6, 7.8,
         4.0, 4.0, 10.0, 10.0])
    world_bbox_z = tf.constant([1.0, 2.0, 3.0, -1.0, -2.0])
    front = tf.constant([0.5, 1.0, -0.5, -1.0, 0.5])
    back = tf.constant([-1.0, 0.0, 0.0, 0.63, -1.0])

    canvas_shape = Canvas2D(height=tf.ones([1, 12]), width=tf.ones([1, 12]))

    sparse_coordinates = tf.SparseTensor(
        values=bbox_coordinates,
        dense_shape=[5, 5, 2, 2],
        indices=[[f, 0, j, k]
                 for f in frame_indices
                 for j in range(2)
                 for k in range(2)])
    sparse_object_class = tf.SparseTensor(
        values=object_class,
        dense_shape=[5, 5, 1],
        indices=[[f, 0, 0]
                 for f in frame_indices])
    sparse_world_bbox_z = tf.SparseTensor(
        values=world_bbox_z,
        dense_shape=[5, 5, 1],
        indices=[[f, 0, 0]
                 for f in frame_indices])
    sparse_front = tf.SparseTensor(
        values=front,
        dense_shape=[5, 5, 1],
        indices=[[f, 0, 0]
                 for f in frame_indices])
    sparse_back = tf.SparseTensor(
        values=back,
        dense_shape=[5, 5, 1],
        indices=[[f, 0, 0]
                 for f in frame_indices])

    source_weight = [tf.constant(2.0, tf.float32)]

    # Initialize all fields to empty lists (to signify 'optional' fields).
    bbox_2d_label_kwargs = {field_name: []
                            for field_name in types.Bbox2DLabel._fields}

    bbox_2d_label_kwargs.update({
        'frame_id': tf.constant('bogus'),
        'object_class': sparse_object_class,
        'vertices': types.Coordinates2D(
            coordinates=sparse_coordinates, canvas_shape=canvas_shape),
        'world_bbox_z': sparse_world_bbox_z,
        'front': sparse_front,
        'back': sparse_back,
        'source_weight': source_weight})

    return types.Bbox2DLabel(**bbox_2d_label_kwargs)


class TestBaseLabelFilter:
    def test_no_op_dict(self):
        """Test that the base filter acts as no-op with default settings."""
        base_label_filter = BaseLabelFilter()
        original_labels = get_dummy_labels(['class_1', 'class_2', 'class_3'])
        filtered_labels = apply_label_filters(
            [base_label_filter], original_labels)
        with tf.compat.v1.Session() as sess:
            for feature in original_labels:
                original_feature, filtered_feature = sess.run([original_labels[feature],
                                                               filtered_labels[feature]])
                assert np.all(original_feature == filtered_feature)

    def test_no_op_bbox_2d_label(self):
        """Test that the base filter acts as no-op with default settings with bbox 2d labels."""
        base_label_filter = BaseLabelFilter()
        original_labels = _get_bbox_2d_labels()
        filtered_labels = apply_label_filters(
            [base_label_filter], original_labels)
        original_source_shape = tf.shape(
            input=original_labels.object_class.values)
        filtered_source_shape = tf.shape(
            input=filtered_labels.object_class.values)
        with tf.compat.v1.Session() as sess:
            output_original_source_shape = sess.run(original_source_shape)
            output_filtered_source_shape = sess.run(filtered_source_shape)
            np.testing.assert_equal(
                output_original_source_shape, output_filtered_source_shape)


class TestSourceClassLabelFilter:
    def test_class_filtering_dict(self):
        """Test that supplying target class names only keeps those labels in the base filter."""
        source_class_names = ['class_1', 'class_2', 'class_3', 'class_4']
        with tf.compat.v1.Session() as sess:
            for i, _ in enumerate(source_class_names):
                # Exclude one class.
                remaining_class_names = source_class_names[:i] + \
                    source_class_names[i+1:]
                # Get base filter that only applies to the remaining classes.
                label_filter = SourceClassLabelFilter(
                    source_class_names=remaining_class_names)
                # Duplicate an entry, for good measure.
                remaining_class_names = [
                    remaining_class_names[0]] + remaining_class_names
                # Get dummy labels.
                original_labels = get_dummy_labels(
                    remaining_class_names,
                    other_attributes={'target/object_class': remaining_class_names})
                filtered_labels = apply_label_filters(
                    [label_filter], original_labels)
                filtered_class_labels = sess.run(
                    [filtered_labels[OBJECT_CLASS_KEY]])[0]
                # Check that the filter worked.
                assert np.all(remaining_class_names ==
                              filtered_class_labels.astype(str))

    @pytest.mark.parametrize(
        "filtered_class_names,exp_filtered_class_names",
        [
            (['truck', 'automobile'], [b'automobile', b'truck', b'truck']),
            (['pedestrian', 'automobile'], [b'pedestrian', b'automobile']),
        ]
    )
    def test_class_filtering_bbox_2d_label(self, filtered_class_names, exp_filtered_class_names):
        """Test source class label filter could filter label correctly with bbox 2d labels.

        Args:
            filtered_class_names (list of str): Source classes to filter.
            exp_filtered_class_names (list of str): Expected class names returned
                after label filter.

        Raises:
            AssertionError: If the filtering did not behave as expected.
        """
        original_labels = _get_bbox_2d_labels()
        label_filter = SourceClassLabelFilter(
            source_class_names=filtered_class_names)
        filtered_labels = apply_label_filters([label_filter], original_labels)
        filtered_source_classes = filtered_labels.object_class.values
        with tf.compat.v1.Session() as sess:
            output_filtered_source_classes = sess.run(filtered_source_classes)
            for i in range(len(exp_filtered_class_names)):
                assert output_filtered_source_classes[i] == exp_filtered_class_names[i]


def _get_dummy_bbox_labels(source_class_names, heights, widths, other_attributes=None):
    """Generate some dummy labels with given dimensions.

    Args:
        source_class_names (list of str): List of object class names for the target labels.
        heights (list of float): Follows indexing of <source_class_names> and has the corresponding
            height.
        widths (list of float): Likewise but for width.
        other_attributes (dict): Key value pairs with which to update the label examples. e.g.
            {'target/coordinates_x1': ...}

    Returns:
        frame_ground_truth_labels (dict): Contains the keys 'target/object_class' and the
            coordinates' keys.
    """
    num_targets = len(source_class_names)
    assert len(source_class_names) == len(heights) == len(
        widths), "Inputs of different lengths"
    if other_attributes is None:
        other_attributes = dict()
    # TODO(@williamz): Again, these keys should really be better defined than hardcoded in every
    #  file that needs them.
    x1 = np.random.uniform(low=-50.0, high=1000.0, size=num_targets)
    x2 = x1 + widths
    y1 = np.random.uniform(low=-1000.0, high=-200.0, size=num_targets)
    y2 = y1 + heights

    other_attributes['target/bbox_coordinates'] = np.stack(
        [x1, y1, x2, y2], axis=1)

    if 'target/object_class' not in other_attributes:
        other_attributes['target/object_class'] = source_class_names

    return get_dummy_labels(source_class_names, other_attributes=other_attributes)


def check_filtered_labels(original_labels, filtered_labels, kept_indices):
    """Assert that the filtering was done as expected.

    Args:
        original_labels (dict of tf.Tensors): pre-filtering ground truth labels.
        filtered_labels (dict of tf.Tensors): post-filtering ground truth labels.
        kept_indices (list of ints): Indicates which indices should have been kept after applying
            the filter(s).

    Raises:
        AssertionError: if the filtering did not behave as expected.
    """
    for feature_name in original_labels:
        original_feature = original_labels[feature_name]
        filtered_feature = filtered_labels[feature_name]
        # Now check that only the ones that should have been kept are kept.
        assert len(filtered_feature) == len(kept_indices)
        # Check the values.
        if len(kept_indices) == 0:  # This means there should be nothing left.
            assert filtered_feature.size == 0
        else:
            for i, original_kept_index in enumerate(kept_indices):
                np.testing.assert_equal(
                    filtered_feature[i], original_feature[original_kept_index])


class TestBboxDimensionsLabelFilter:
    @pytest.mark.parametrize(
        "min_width,min_height,max_width,max_height,is_valid",
        [
            (-10.0, 0.5, 10.0, 1.0, True),
            (-10.0, 0.5, -11.1, 0.6, False),
            (-10.0, 0.0, -11.0, 0.0, False),
        ]
    )
    def test_bbox_dimensions_ranges(self, min_width, min_height, max_width, max_height, is_valid):
        """Test that the BboxDimensionsLabelFilter makes the necessary checks.

        Args:
            min/max_width/height (float): Thresholds for bbox dimensions.
            is_valid (bool): If True, the instantiation should happen gracefully. If False, an
                AssertionError is expected.
        """
        if is_valid:
            BboxDimensionsLabelFilter(min_width=min_width,
                                      min_height=min_height,
                                      max_width=max_width,
                                      max_height=max_height)
        else:
            with pytest.raises(AssertionError):
                BboxDimensionsLabelFilter(min_width=min_width,
                                          min_height=min_height,
                                          max_width=max_width,
                                          max_height=max_height)

    @pytest.mark.parametrize(
        "source_class_names,heights,widths,params,kept_indices",
        [
            # Since there are no bounds, everything should be kept.
            (["class_1", "class_2", "class_3"], [123.4, 567.8, 90.1], [23., 456., 78.9],
             dict(), range(3)),
            # Now min_width should be in effect.
            (["class_4", "class_5", "class_6"], [123.4, 567.8, 90.1], [23., 456., 78.9],
             {'min_width': np.float64(24.0)}, [1, 2]),  # By default TF casts stuff as float32.
            # Similar test case for min_height.
            (["class_7", "class_8", "class_9"], [123.4, 567.8, 90.1], [23., 456., 78.9],
             {'min_height': np.float64(124.0)}, [1]),
            # Try multiple bounds.
            (["class_7", "class_8", "class_9"], [123.4, 567.8, 90.1], [23., 456., 78.9],
             {'min_height': np.float64(124.0), 'max_width': np.float64(501.0)}, [1]),
            (["class_7", "class_8", "class_9"], [123.4, 567.8, 90.1], [23., 456., 78.9],
             {'max_height': np.float64(124.0), 'min_width': np.float64(401.0)}, []),
            (["class_7", "class_8", "class_9"], [123.4, 567.8, 90.1], [23., 456., 78.9],
             {'max_height': np.float64(124.0), 'min_width': np.float64(55.2)}, [2])
        ]
    )
    def test_bbox_dimensions_label_filter_dict(self,
                                               source_class_names,
                                               heights,
                                               widths,
                                               params,
                                               kept_indices):
        """Test the BboxDimensionsLabelFilter.

        Args:
            source_class_names (list of str): List of object class names to be used for the target
                labels.
            heights (list of float): Follows indexing of <source_class_names> and has the
                corresponding height.
            widths (list of float): Likewise but for width.
            params (dict): Contains the keyword arguments with which the BboxDimensionsLabelFilter
                will be instantiated.
            kept_indices (list): Contains the indices in the input (<source_class_names>, <heights>,
                <width>) that should be kept after the filtering happens.

        Raises:
            AssertionError: if the filtering did not behave as expected.
        """
        original_labels = _get_dummy_bbox_labels(
            source_class_names, heights, widths)
        bbox_dimensions_label_filter = BboxDimensionsLabelFilter(**params)
        filtered_labels = apply_label_filters(
            [bbox_dimensions_label_filter], original_labels)
        # First, check that all keys are kept.
        assert set(original_labels.keys()) == set(filtered_labels.keys())
        with tf.compat.v1.Session() as sess:
            np_original_labels, np_filtered_labels = sess.run(
                [original_labels, filtered_labels])
        check_filtered_labels(np_original_labels,
                              np_filtered_labels, kept_indices)

    @pytest.mark.parametrize(
        "min_width,min_height,max_width,max_height,exp_class_names",
        [
            # Case 1: normal case.
            (2.0, 2.0, 5.0, 4.0,
             [b'unmapped', b'automobile']),
            # Case 2: None input - by default, if part of params are set to None,
            # it loosens the constraint.
            (None, 2.0, 5.0, 4.0,
                [b'pedestrian', b'unmapped', b'automobile'])
        ]
    )
    def test_bbox_dimensions_label_filter_bbox_2d_label(self,
                                                        min_width,
                                                        min_height,
                                                        max_width,
                                                        max_height,
                                                        exp_class_names):
        """Test the BboxDimensionsLabelFilter with bbox 2d labels.

        Args:
            min/max_width/height (float): Thresholds above/below which to keep bounding
                box objects. If None, the corresponding threshold is not used.
            exp_class_names (list of str): expected output class names from filtered labels.

        Raises:
            AssertionError: If the filtering did not behave as expected.
        """
        original_labels = _get_bbox_2d_labels()
        label_filter = BboxDimensionsLabelFilter(min_width=min_width,
                                                 min_height=min_height,
                                                 max_width=max_width,
                                                 max_height=max_height)
        filtered_labels = apply_label_filters([label_filter], original_labels)
        filtered_source_classes = filtered_labels.object_class.values
        with tf.compat.v1.Session() as sess:
            output_filtered_source_classes = sess.run(filtered_source_classes)
            for i in range(len(exp_class_names)):
                assert output_filtered_source_classes[i] == exp_class_names[i]


class TestBboxCropLabelFilter:
    @pytest.mark.parametrize(
        "crop_left,crop_right,crop_top,crop_bottom,is_valid",
        [
            (0, 0, 0, 0, True),
            (0, 10, 0, 10, True),
            (20, 10, 10, 20, False),
            (None, None, None, None, True),
        ]
    )
    def test_bbox_crop_ranges(self, crop_left, crop_right, crop_top, crop_bottom, is_valid):
        """Test that the BboxCropLabelFilter makes the necessary checks.

        Args:
            crop_left/crop_right/crop_top/crop_bottom: crop coordinates.
            is_valid (bool): If True, the instantiation should happen gracefully. If False, an
                AssertionError is expected.
        """
        if is_valid:
            BboxCropLabelFilter(crop_left=crop_left,
                                crop_right=crop_right,
                                crop_top=crop_top,
                                crop_bottom=crop_bottom)
        else:
            with pytest.raises(ValueError):
                BboxCropLabelFilter(crop_left=crop_left,
                                    crop_right=crop_right,
                                    crop_top=crop_top,
                                    crop_bottom=crop_bottom)

    @pytest.mark.parametrize(
        "x1,x2,y1,y2,params,kept_indices",
        [
            # Try multiple bounds.
            ([0., 20., 40.], [10., 30., 50.], [0., 20., 40.], [10., 30., 50.],
             {'crop_left': None, 'crop_right': None,
              'crop_top': None, 'crop_bottom': None}, [0, 1, 2]),
            ([0., 20., 40.], [10., 30., 50.], [0., 20., 40.], [10., 30., 50.],
             {'crop_left': np.int32(0), 'crop_right': np.int32(0),
              'crop_top': np.int32(0), 'crop_bottom': np.int32(0)}, [0, 1, 2]),
            ([0., 20., 40.], [10., 30., 50.], [0., 20., 40.], [10., 30., 50.],
             {'crop_left': np.int32(25), 'crop_right': np.int32(50),
              'crop_top': np.int32(25), 'crop_bottom': np.int32(50)}, [1, 2]),
            ([0., 20., 5.], [10., 25., 15.], [0., 20., 5.], [10., 25., 15.],
             {'crop_left': np.int32(25), 'crop_right': np.int32(50),
              'crop_top': np.int32(25), 'crop_bottom': np.int32(50)}, []),
            ([25., 20., 40.], [30., 30., 50.], [25., 20., 40.], [30., 30., 50.],
             {'crop_left': np.int32(25), 'crop_right': np.int32(50),
              'crop_top': np.int32(25), 'crop_bottom': np.int32(50)}, [0, 1, 2]),
            ([0., 20., 40.], [10., 30., 50.], [0., 20., 40.], [10., 30., 50.],
             {'crop_left': np.int32(5), 'crop_right': np.int32(20),
              'crop_top': np.int32(5), 'crop_bottom': np.int32(20)}, [0]),
        ]
    )
    def test_bbox_crop_label_filter_dict(self, x1, x2, y1, y2, params, kept_indices):
        """Test the BboxCropLabelFilter.

        Args:
            x1/x2/y1/y2: bbox coordinates.
            params (dict): Contains the keyword arguments with which the BboxCropLabelFilter
                will be instantiated.
            kept_indices (list): Contains the indices in the input
                that should be kept after the filtering happens.

        Raises:
            AssertionError: if the filtering did not behave as expected.
        """
        original_labels = dict()
        original_labels['target/object_class'] = tf.constant(
            ['class_1', 'class_2', 'class_3'])
        original_labels['target/bbox_coordinates'] = tf.stack(
            [x1, y1, x2, y2], axis=1)
        bbox_crop_label_filter = BboxCropLabelFilter(**params)
        filtered_labels = apply_label_filters(
            [bbox_crop_label_filter], original_labels)
        # First, check that all keys are kept.
        assert set(original_labels.keys()) == set(filtered_labels.keys())
        with tf.compat.v1.Session() as sess:
            np_original_labels, np_filtered_labels = sess.run(
                [original_labels, filtered_labels])
        check_filtered_labels(np_original_labels,
                              np_filtered_labels, kept_indices)

    @pytest.mark.parametrize(
        "crop_left,crop_right,crop_top,crop_bottom,exp_class_names",
        [
            # Case 1: normal case.
            (2.0, 3.6, 2.0, 3.8,
             [b'unmapped', b'automobile', b'truck']),
            # Case 2: None input - by default, it pass through by returning all Trues.
            (2.0, 3.6, 2.0, None,
                [b'pedestrian', b'unmapped', b'automobile', b'truck', b'truck'])
        ]
    )
    def test_bbox_crop_label_filter_bbox_2d_label(self,
                                                  crop_left,
                                                  crop_right,
                                                  crop_top,
                                                  crop_bottom,
                                                  exp_class_names):
        """Test the BboxCropLabelFilter with bbox 2d labels.

        Args:
            crop_left/right/top/bottom: bbox coordinates.
            exp_class_names (list of str): expected output class names from filtered labels.

        Raises:
            AssertionError: if the filtering did not behave as expected.
        """
        original_labels = _get_bbox_2d_labels()
        label_filter = BboxCropLabelFilter(crop_left=crop_left,
                                           crop_right=crop_right,
                                           crop_top=crop_top,
                                           crop_bottom=crop_bottom)
        filtered_labels = apply_label_filters([label_filter], original_labels)
        filtered_source_classes = filtered_labels.object_class.values
        with tf.compat.v1.Session() as sess:
            output_filtered_source_classes = sess.run(filtered_source_classes)
            for i in range(len(exp_class_names)):
                assert output_filtered_source_classes[i] == exp_class_names[i]


class TestChainedFilters:
    @pytest.mark.parametrize(
        "source_class_names,heights,widths,bbox_params,object_class_params,mode,kept_indices",
        [
            # No-op.
            (["class_1", "class_2", "class_3"], [123.4, 567.8, 90.1], [23., 456., 78.9],
             dict(), dict(), 'or', range(3)),
            # Check logical-or when filters are chained.
            (["class_1", "class_2", "class_3"], [123.4, 567.8, 90.1], [23., 456., 78.9],
             dict(), dict(source_class_names=["class_1", "class_2"]), 'or', range(3)),
            # Check logical-and when filters are chained.
            (["class_1", "class_2", "class_3"], [123.4, 567.8, 90.1], [23., 456., 78.9],
             dict(), dict(source_class_names=["class_1", "class_2"]), 'and', range(2)),
            # Now do some actual filtering.
            (["class_1", "class_2", "class_3"], [123.4, 567.8, 90.1], [23., 456., 78.9],
             # Bbox filter: the only bbox within max_height is the last one.
             dict(max_height=np.float64(100.0)),
             dict(source_class_names=["class_3"]), 'or', [2]),
            # Another 'concrete' filtering.
            (["class_1", "class_2", "class_3"], [123.4, 567.8, 90.1], [23., 456., 78.9],
             # Bbox filter: only keep the first 2 labels.
             dict(min_height=np.float64(100.0)),
             # Object class filter: keep 1st and 2nd class.
             dict(source_class_names=["class_1", "class_2"]), 'or', [0, 1]),
            # A 'concrete' filtering for logical-and.
            (["class_1", "class_2", "class_3"], [123.4, 567.8, 90.1], [23., 456., 78.9],
             # Bbox filter: only keep the first 2 labels.
             dict(min_height=np.float64(100.0)),
             # Object class filter: keep 1st and 2nd class.
             dict(source_class_names=["class_1", "class_3"]), 'and', [0]),
            # Do some width filtering.
            (["class_1", "class_2", "class_3"], [123.4, 567.8, 90.1], [23., 456., 78.9],
             # Bbox filter: only 2nd label satisfies this.
             dict(min_width=np.float64(100.0)),
             # Object class filter: only the 3rd left (but logical-or results in [1, 2] being kept.
             dict(source_class_names=["class_2", "class_3"]), 'or', [1, 2]),
        ]
    )
    def test_bbox_and_object_class_label_filters_dict(self,
                                                      source_class_names,
                                                      heights,
                                                      widths,
                                                      bbox_params,
                                                      object_class_params,
                                                      mode,
                                                      kept_indices):
        original_labels = \
            _get_dummy_bbox_labels(source_class_names, heights, widths)
        # Get the filters.
        bbox_dimensions_label_filter = BboxDimensionsLabelFilter(**bbox_params)
        source_class_label_filter = SourceClassLabelFilter(
            **object_class_params)
        # Chain them.
        filtered_labels = apply_label_filters([bbox_dimensions_label_filter,
                                               source_class_label_filter],
                                              original_labels, mode)
        # First, check that all keys are kept.
        assert set(original_labels.keys()) == set(filtered_labels.keys())
        with tf.compat.v1.Session() as sess:
            np_original_labels, np_filtered_labels = sess.run(
                [original_labels, filtered_labels])
        check_filtered_labels(np_original_labels,
                              np_filtered_labels, kept_indices)

    @pytest.mark.parametrize(
        "mode,exp_class_names",
        [
            ('or', [b'unmapped', b'automobile', b'truck', b'truck']),
            ('and', [b'automobile'])
        ]
    )
    def test_bbox_and_object_class_label_filters_bbox_2d_label(self, mode, exp_class_names):
        """Test chain logic works correctly for multiple label filters with bbox 2d label.
        Args:
            mode (str): The chain mode, which should be 'or' or 'and'.

        Raises:
            AssertionError: if the chain did not behave as expected.
        """
        original_labels = _get_bbox_2d_labels()
        bbox_dimensions_label_filter = BboxDimensionsLabelFilter(min_width=2.0,
                                                                 min_height=2.0,
                                                                 max_width=5.0,
                                                                 max_height=4.0)
        source_class_label_filter = \
            SourceClassLabelFilter(source_class_names=['truck', 'automobile'])
        filtered_labels = apply_label_filters([bbox_dimensions_label_filter,
                                               source_class_label_filter],
                                              original_labels, mode=mode)
        filtered_class_names = filtered_labels.object_class.values

        sess = tf.compat.v1.Session()
        output_filtered_labels = sess.run(filtered_class_names)
        for i in range(len(exp_class_names)):
            assert output_filtered_labels[i] == exp_class_names[i]
