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

"""Tests for ground truths."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types import Bbox2DLabel
from nvidia_tao_tf1.blocks.multi_source_loader.types import Coordinates2D
from nvidia_tao_tf1.cv.detectnet_v2.evaluation.ground_truth import (
    get_ground_truth_objects_from_batch_ground_truth,
    get_ground_truth_objects_from_bbox_label,
    GroundTruth
)

ground_truth1 = GroundTruth(class_name='car',
                            bbox=np.array([1., 1., 2., 2.]),
                            truncation=0.1,
                            truncation_type=0,
                            occlusion=1,
                            is_cvip=False,
                            world_bbox_z=0.0,
                            front=0.0,
                            back=0.0,
                            orientation=None)
ground_truth2 = GroundTruth(class_name='car',
                            bbox=np.array([3., 3., 4., 4.]),
                            truncation=0.2,
                            truncation_type=0,
                            occlusion=2,
                            is_cvip=True,
                            world_bbox_z=0.0,
                            front=0.0,
                            back=0.0,
                            orientation=None)
ground_truth3 = GroundTruth(class_name='car',
                            bbox=np.array([1., 1., 2., 2.]),
                            truncation=0.0,
                            truncation_type=1,
                            occlusion=1,
                            is_cvip=False,
                            world_bbox_z=0.0,
                            front=0.0,
                            back=0.0,
                            orientation=None)
ground_truth4 = GroundTruth(class_name='dontcare',
                            bbox=np.array([1., 1., 2., 2.]),
                            truncation=0.1,
                            truncation_type=0,
                            occlusion=1,
                            is_cvip=False,
                            world_bbox_z=0.0,
                            front=0.0,
                            back=0.0,
                            orientation=None)
ground_truth5 = GroundTruth(class_name='some_facing_object',
                            bbox=np.array([1., 1., 2., 2.]),
                            truncation=0.1,
                            truncation_type=0,
                            occlusion=1,
                            is_cvip=False,
                            world_bbox_z=0.0,
                            front=0.0,
                            back=0.0,
                            orientation=None)

no_objects = [{'target/object_class': [],
               'target/bbox_coordinates': [],
               'target/truncation': [],
               'target/occlusion': []}]

no_objects_expected = [[]]

single_object = [{'target/object_class': ['car'],
                  'target/bbox_coordinates': [[1., 1., 2., 2.]],
                  'target/truncation': [0.1],
                  'target/occlusion': [1],
                  'target/non_facing': [0]},
                 {'target/object_class': ['some_non_facing_object'],
                  'target/bbox_coordinates': [[1., 1., 2., 2.]],
                  'target/truncation': [0.1],
                  'target/occlusion': [1],
                  'target/non_facing': [1]},
                 {'target/object_class': ['some_facing_object'],
                  'target/bbox_coordinates': [[1., 1., 2., 2.]],
                  'target/truncation': [0.1],
                  'target/occlusion': [1],
                  'target/non_facing': [0]}]

single_object_expected = [[ground_truth1], [ground_truth4], [ground_truth5]]

two_frames = no_objects + [{'target/object_class': ['car', 'car'],
                            'target/bbox_coordinates': [[1., 1., 2., 2.], [3., 3., 4., 4.]],
                            'target/truncation': [0.1, 0.2],
                            'target/occlusion': [1, 2],
                            'target/is_cvip': [False, True]}]

two_frames_expected = [[], [ground_truth1, ground_truth2]]

single_object_truncation_type = [{'target/object_class': ['car'],
                                  'target/bbox_coordinates': [[1., 1., 2., 2.]],
                                  'target/truncation_type': [1],
                                  'target/occlusion': [1]}]

single_object_truncation_type_expected = [[ground_truth3]]


def compare_ground_truths(a, b):
    """Compare GroundTruths by their attributes."""
    assert a.class_name == b.class_name

    np.testing.assert_array_almost_equal(a.bbox, b.bbox)
    np.testing.assert_almost_equal(a.truncation, b.truncation)
    np.testing.assert_almost_equal(a.truncation_type, b.truncation_type)
    np.testing.assert_almost_equal(a.occlusion, b.occlusion)

    assert a.is_cvip == b.is_cvip

    return True


@pytest.mark.parametrize("batch_ground_truth,expected_ground_truths",
                         [(no_objects, no_objects_expected),
                          (single_object, single_object_expected),
                          (two_frames, two_frames_expected),
                          (single_object_truncation_type, single_object_truncation_type_expected)])
def test_get_ground_truth_objects_from_batch_ground_truth(monkeypatch, batch_ground_truth,
                                                          expected_ground_truths):
    """Test generation of GroundTruth objects from tensors."""
    ground_truths = get_ground_truth_objects_from_batch_ground_truth(
        batch_ground_truth)

    # Compare objects by their attributes, not ids
    monkeypatch.setattr(GroundTruth, '__eq__', compare_ground_truths)

    assert ground_truths == expected_ground_truths

# The following lines mimic the above test but using Bbox2DLabel (that have been eval'ed).


def _get_empty_bbox_2d_label_kwargs():
    return {field_name: [] for field_name in Bbox2DLabel._fields}


no_objects_kwargs = _get_empty_bbox_2d_label_kwargs()
no_objects_kwargs.update({
    'vertices': Coordinates2D(
        coordinates=tf.compat.v1.SparseTensorValue(
            values=[], dense_shape=[1, 0, 0, 0], indices=[]),
        canvas_shape=None),
    'object_class': tf.compat.v1.SparseTensorValue(
        values=[], dense_shape=[1, 0, 0],
        indices=np.reshape(np.array([], dtype=np.int64), [0, 3]))})

no_objects_bis = Bbox2DLabel(**no_objects_kwargs)

single_object_kwargs = _get_empty_bbox_2d_label_kwargs()
single_object_kwargs.update({
    'vertices': Coordinates2D(
        coordinates=tf.compat.v1.SparseTensorValue(
            values=[1., 1., 2., 2., 1., 1., 2., 2., 1., 1., 2., 2.],
            dense_shape=[3, 1, 2, 2],
            indices=np.array(
                [[i, 0, j, k] for i in range(3) for j in range(2) for k in range(2)])),
        canvas_shape=None),
    'object_class': tf.compat.v1.SparseTensorValue(
        values=['car', 'some_non_facing_object', 'some_facing_object'],
        dense_shape=[3, 1, 1],
        indices=np.array([[i, 0, 0] for i in range(3)])),
    'truncation': tf.compat.v1.SparseTensorValue(
        values=[0.1, 0.1, 0.1],
        dense_shape=[3, 1, 1],
        indices=np.array([[i, 0, 0] for i in range(3)])),
    'occlusion': tf.compat.v1.SparseTensorValue(
        values=[1, 1, 1],
        dense_shape=[3, 1, 1],
        indices=np.array([[i, 0, 0] for i in range(3)])),
    'non_facing': tf.compat.v1.SparseTensorValue(
        values=[0, 1, 0],
        dense_shape=[3, 1, 1],
        indices=np.array([[i, 0, 0] for i in range(3)]))})
single_object_bis = Bbox2DLabel(**single_object_kwargs)

two_frames_kwargs = _get_empty_bbox_2d_label_kwargs()
two_frames_kwargs.update({
    'vertices': Coordinates2D(
        coordinates=tf.compat.v1.SparseTensorValue(
            values=[1., 1., 2., 2., 3., 3., 4., 4.],
            dense_shape=[2, 2, 2, 2],
            indices=np.array(
                [[1, i, j, k] for i in range(2) for j in range(2) for k in range(2)])),
        canvas_shape=None),
    'object_class': tf.compat.v1.SparseTensorValue(
        values=['car', 'car'],
        dense_shape=[2, 2, 1],
        indices=np.array([[1, i, 0] for i in range(2)])),
    'truncation': tf.compat.v1.SparseTensorValue(
        values=[0.1, 0.2],
        dense_shape=[2, 2, 1],
        indices=np.array([[1, i, 0] for i in range(2)])),
    'occlusion': tf.compat.v1.SparseTensorValue(
        values=[1, 2],
        dense_shape=[2, 2, 1],
        indices=np.array([[1, i, 0] for i in range(2)])),
    'is_cvip': tf.compat.v1.SparseTensorValue(
        values=[False, True],
        dense_shape=[2, 2, 1],
        indices=np.array([[1, i, 0] for i in range(2)]))})
two_frames_bis = Bbox2DLabel(**two_frames_kwargs)

single_object_truncation_type_kwargs = _get_empty_bbox_2d_label_kwargs()
single_object_truncation_type_kwargs.update({
    'vertices': Coordinates2D(
        coordinates=tf.compat.v1.SparseTensorValue(
            values=[1., 1., 2., 2.],
            dense_shape=[1, 1, 2, 2],
            indices=np.array([[0, 0, i, j] for i in range(2) for j in range(2)])),
        canvas_shape=None),
    'object_class': tf.compat.v1.SparseTensorValue(
        values=['car'],
        dense_shape=[1, 1, 1],
        indices=np.array([[0, 0, 0]])),
    'truncation_type': tf.compat.v1.SparseTensorValue(
        values=[1],
        dense_shape=[1, 1, 1],
        indices=np.array([[0, 0, 0]])),
    'occlusion': tf.compat.v1.SparseTensorValue(
        values=[1],
        dense_shape=[1, 1, 1],
        indices=np.array([[0, 0, 0]])),
    # Add an empty sparse tensor to test that the code properly handles it.
    'world_bbox_z': tf.compat.v1.SparseTensorValue(
        values=[],
        dense_shape=[1, 1, 1],
        indices=[],
    )
})
single_object_truncation_type_bis = Bbox2DLabel(
    **single_object_truncation_type_kwargs)


@pytest.mark.parametrize("bbox_2d_label,expected_ground_truths",
                         [(no_objects_bis, no_objects_expected),
                          (single_object_bis, single_object_expected),
                          (two_frames_bis, two_frames_expected),
                          (single_object_truncation_type_bis,
                           single_object_truncation_type_expected)])
def test_get_ground_truth_objects_from_bbox_label(monkeypatch, bbox_2d_label,
                                                  expected_ground_truths):
    """Test generation of GroundTruth objects from Bbox2DLabel."""
    ground_truths = get_ground_truth_objects_from_bbox_label(bbox_2d_label)

    # Compare objects by their attributes, not ids
    monkeypatch.setattr(GroundTruth, '__eq__', compare_ground_truths)

    assert ground_truths == expected_ground_truths
