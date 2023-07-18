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

"""Tests for Detection postprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.detection import Detection
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.postprocessing import _filter_by_confidence
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.postprocessing import _patch_detections
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.postprocessing import PostProcessor

test_detections = [[
    Detection(
        class_name='car',
        bbox=[0., 0., 16., 16.],
        confidence=50.0,
        bbox_variance=0.,
        num_raw_bboxes=1),
    Detection(
        class_name='pedestrian',
        bbox=[16., 16., 32., 32.],
        confidence=50.0,
        bbox_variance=0.,
        num_raw_bboxes=1)
]]


def test_patch_detections():
    """Test _patch_detections."""
    confidences = np.array([[0.5], [0.5]])
    updated_detections = _patch_detections(test_detections, confidences)
    assert updated_detections[0][0].confidence == confidences[0][0]
    assert updated_detections[0][1].confidence == confidences[1][0]


def _mocked_cluster_predictions(batch_predictions, clustering_config):
    """Mocked cluster_predictions."""
    return {'car': [[batch_predictions[0][0]]], 'pedestrian': [[batch_predictions[0][1]]]}


def test_filter_detections():
    """Test _filter_by_confidence."""
    # Generate random data for testing.
    test_detections = [[Detection(class_name='car', bbox=[0., 0., 16., 16.], confidence=0.8,
                                  bbox_variance=0., num_raw_bboxes=1),
                        Detection(class_name='car', bbox=[0., 0., 12., 12.], confidence=0.2,
                                  bbox_variance=0., num_raw_bboxes=1)],
                       [Detection(class_name='car', bbox=[0., 0., 10., 10.], confidence=0.7,
                                  bbox_variance=0., num_raw_bboxes=1)]]
    expected_filtered_detections = [[test_detections[0][0]], test_detections[1]]
    filtered_detections = _filter_by_confidence(test_detections, confidence_threshold=0.5)
    np.testing.assert_equal(filtered_detections, expected_filtered_detections)


@pytest.fixture(scope='function')
def target_class_names():
    return ['car', 'pedestrian']


@pytest.fixture(scope='function')
def postprocessor(mocker, target_class_names):
    """Define a PostProcessor object."""
    # Mock clustering.
    mocker.patch("nvidia_tao_tf1.cv.detectnet_v2.postprocessor.postprocessing.cluster_predictions",
                 _mocked_cluster_predictions)

    # Mock confidence config.
    mock_confidence_config = mocker.MagicMock(confidence_threshold=0.3)

    image_size = (32., 32.)
    mock_postprocessing_config = \
        dict.fromkeys(target_class_names,
                      mocker.MagicMock(confidence_config=mock_confidence_config))

    postprocessor = PostProcessor(
        postprocessing_config=mock_postprocessing_config,
        confidence_models=None,
        image_size=image_size)

    return postprocessor


def test_postprocessor(mocker, postprocessor, target_class_names):
    """Test the different steps in the postprocessing pipeline."""
    clustered_detections = postprocessor.cluster_predictions(test_detections)

    assert clustered_detections['car'][0][0] == test_detections[0][0]
    assert clustered_detections['pedestrian'][0][0] == test_detections[0][1]


def test_postprocess_predictions(mocker, postprocessor, target_class_names):
    """Test that a the single postprocess_predictions() call applies all expected steps.

    The end result should be the same as that of <clustered_detections_with_confidence> in the
    test_processor() function.
    """
    final_detections = postprocessor.postprocess_predictions(
        predictions=test_detections,
        target_class_names=target_class_names,
        session=mocker.MagicMock())  # Not required because of the patch.

    assert final_detections['car'][0][0].confidence == 50.0
    assert final_detections['pedestrian'][0][0].confidence == 50.0
