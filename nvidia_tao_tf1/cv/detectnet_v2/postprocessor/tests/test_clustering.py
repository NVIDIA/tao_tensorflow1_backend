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

"""Tests for bbox clustering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.testing as npt
import pytest

from six.moves import zip
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.cluster import cluster_predictions
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.cluster import mean_angle
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.clustering_config import ClusteringConfig
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.confidence_config import ConfidenceConfig
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.detection import Detection
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.postprocessing_config import PostProcessingConfig


class ClusteringTestCase:
    def __init__(self, target_classes, raw_detections, postprocessing_config, outputs):
        self.target_classes = target_classes
        self.raw_detections = raw_detections
        self.postprocessing_config = postprocessing_config
        self.outputs = outputs


def create_default_case(shape=(64, 64)):
    """Create default test case to be modified."""
    target_classes = ['car']

    # (num_images, num_classes, num_outputs, grid_height, grid_width)
    bboxes = np.zeros((1, 1, 4) + shape, dtype=np.float32)
    cov = np.zeros((1, 1, 1) + shape, dtype=np.float32)

    raw_detections = {
        'bbox': bboxes,
        'cov': cov
    }

    clustering_config = ClusteringConfig(
        coverage_threshold=0.005,
        dbscan_eps=0.15,
        dbscan_min_samples=1,
        minimum_bounding_box_height=4,
        clustering_algorithm=0,
        nms_iou_threshold=0.4,
        dbscan_confidence_threshold=0.1,
        nms_confidence_threshold=0.1)
    confidence_config = ConfidenceConfig(confidence_model_filename=None,
                                         confidence_threshold=0.0)
    car_postprocessing_config = PostProcessingConfig(clustering_config, confidence_config)

    postprocessing_config = {}
    postprocessing_config['car'] = car_postprocessing_config

    outputs = [[]]

    default_test_case = ClusteringTestCase(target_classes, raw_detections,
                                           postprocessing_config, outputs)

    return default_test_case


# Test cases and ids (for pytest) are compiled into this lists
test_cases = [create_default_case()]
test_ids = ['empty_prediction']

# Test whether averaging of the bounding box coordinates is done right
case = create_default_case()

case.raw_detections['bbox'][0, 0, 0:2, 0:5, 0:5] = 0.
case.raw_detections['bbox'][0, 0, 2:4, 0:5, 0:5] = 16.
case.raw_detections['bbox'][0, 0, 0:2, 5:10, 5:10] = .1
case.raw_detections['bbox'][0, 0, 2:4, 5:10, 5:10] = 16.1

case.raw_detections['cov'][0, 0, 0, :24, :24] = 1
case.outputs = [[
    Detection(
        class_name='car',
        bbox=[0.05, 0.05, 16.05, 16.05],
        confidence=50.0,
        bbox_variance=0.,
        num_raw_bboxes=1)
]]
test_cases += [case]
test_ids += ['bbox_coordinate_averaging']

# Test whether additional outputs (depth) is clustered right
case = create_default_case()

case.raw_detections['bbox'][0, 0, 0:2, 0:5, 0:5] = 0.
case.raw_detections['bbox'][0, 0, 2:4, 0:5, 0:5] = 16.
case.raw_detections['bbox'][0, 0, 0:2, 5:10, 5:10] = .1
case.raw_detections['bbox'][0, 0, 2:4, 5:10, 5:10] = 16.1

case.raw_detections['depth'] = np.zeros_like(case.raw_detections['cov'])
case.raw_detections['depth'][0, 0, 0, 0:5, 0:5] = 10.0
case.raw_detections['depth'][0, 0, 0, 5:10, 5:10] = 20.0

case.raw_detections['cov'][0, 0, 0, :24, :24] = 1
case.outputs = [[
    Detection(
        class_name='car',
        bbox=[0.05, 0.05, 16.05, 16.05],
        confidence=50.0,
        bbox_variance=0.,
        num_raw_bboxes=1,
        depth=15.0)
]]
test_cases += [case]
test_ids += ['depth_prediction_averaging']

# Test whether coverage_threshold filters grid cells with low coverage values.
case = create_default_case()

case.raw_detections['bbox'][0, 0, 2, :5, :5] = 16
case.raw_detections['bbox'][0, 0, 3, :5, :5] = 16
case.raw_detections['bbox'][0, 0, 0, 5:10, 5:10] = 16
case.raw_detections['bbox'][0, 0, 1, 5:10, 5:10] = 16
case.raw_detections['bbox'][0, 0, 2, 5:10, 5:10] = 32
case.raw_detections['bbox'][0, 0, 3, 5:10, 5:10] = 32

case.raw_detections['cov'][0, 0, 0, :5, :5] = 0.01
case.raw_detections['cov'][0, 0, 0, 5:10, 5:10] = 1

case.outputs = [[
    Detection(
        class_name='car',
        bbox=[16, 16, 32, 32],
        confidence=25.,
        bbox_variance=0.,
        num_raw_bboxes=1)
]]

case.postprocessing_config['car'].clustering_config.coverage_threshold = 0.1
test_cases += [case]
test_ids += ['coverage_thresholding']

# Test whether minimum_bounding_box_height works
case = create_default_case()

case.raw_detections['bbox'][0, 0, 2, :5, :5] = 5
case.raw_detections['bbox'][0, 0, 3, :5, :5] = 5

case.raw_detections['bbox'][0, 0, 0, 5:10, 5:10] = 5
case.raw_detections['bbox'][0, 0, 1, 5:10, 5:10] = 5
case.raw_detections['bbox'][0, 0, 2, 5:10, 5:10] = 15
case.raw_detections['bbox'][0, 0, 3, 5:10, 5:10] = 15

# Add one bbox which shouldn't be considered because it only has one sample
case.raw_detections['bbox'][0, 0, 0, 11, 11] = 15
case.raw_detections['bbox'][0, 0, 1, 11, 11] = 15
case.raw_detections['bbox'][0, 0, 2, 11, 11] = 25
case.raw_detections['bbox'][0, 0, 3, 11, 11] = 25

case.raw_detections['cov'][0, 0, 0, :11, :11] = 1

case.postprocessing_config['car'].clustering_config.minimum_bounding_box_height = 6

case.outputs = [[
    Detection(
        class_name='car',
        bbox=[5, 5, 15, 15],
        confidence=25.,
        bbox_variance=0.,
        num_raw_bboxes=1)
]]

test_cases += [case]
test_ids += ['minimum bounding box height']


# Test clustering of two classes
case = create_default_case()
case.target_classes = ['car', 'pedestrian']

case.raw_detections['cov'] = np.zeros((1, 2, 1, 64, 64))
case.raw_detections['cov'][0, 0, 0, :10, :10] = 1       # first object
case.raw_detections['cov'][0, 1, 0, 10:20, 10:20] = 1   # second

case.raw_detections['bbox'] = np.zeros((1, 2, 4, 64, 64))
case.raw_detections['bbox'][0, 0, 0:2, 0:5, 0:5] = 0.
case.raw_detections['bbox'][0, 0, 2:4, 0:5, 0:5] = 16.
case.raw_detections['bbox'][0, 0, 0:2, 5:10, 5:10] = 0.
case.raw_detections['bbox'][0, 0, 2:4, 5:10, 5:10] = 16.

case.raw_detections['bbox'][0, 1, 0:2, 10:15, 10:15] = 16.
case.raw_detections['bbox'][0, 1, 2:4, 10:15, 10:15] = 32.
case.raw_detections['bbox'][0, 1, 0:2, 15:20, 15:20] = 16.
case.raw_detections['bbox'][0, 1, 2:4, 15:20, 15:20] = 32.

case.outputs = [[
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

# Add clustering parameters for the second class
pedestrian_clustering_config = ClusteringConfig(
    coverage_threshold=0.005,
    dbscan_eps=0.15,
    dbscan_min_samples=1,
    minimum_bounding_box_height=4,
    clustering_algorithm=0,
    nms_iou_threshold=None,
    dbscan_confidence_threshold=0.1,
    nms_confidence_threshold=0.1)
confidence_config = ConfidenceConfig(confidence_model_filename=None,
                                     confidence_threshold=0.0)
pedestrian_postprocessing_config = PostProcessingConfig(pedestrian_clustering_config,
                                                        confidence_config)
case.postprocessing_config['pedestrian'] = pedestrian_postprocessing_config

test_cases += [case]
test_ids += ['two_bounding_boxes']


class TestClustering:
    """Test cluster_predictions."""

    @pytest.mark.parametrize('case', test_cases, ids=test_ids)
    def test_cluster_detections(self, case):
        """Cluster bboxes and test if they are clustered right."""
        target_classes = case.target_classes
        predictions = dict()
        raw_detections = case.raw_detections

        for target_class_idx, target_class in enumerate(target_classes):
            predictions[target_class] = {}
            for objective in raw_detections:
                predictions[target_class][objective] = \
                    raw_detections[objective][:, target_class_idx, :]

        clustered_detections = cluster_predictions(predictions, case.postprocessing_config)

        # Loop all frames for each target class and the number of detections matches the
        # number of expected detections and that bbox coordinates and confidences are the same.
        for target_class in target_classes:
            for frame_idx, frame_expected_detections in enumerate(case.outputs):
                expected_detections = [detection for detection in frame_expected_detections if
                                       detection.class_name == target_class]
                detections = clustered_detections[target_class][frame_idx]

                assert len(detections) == len(expected_detections)

                for detection, expected_detection in zip(detections, expected_detections):
                    npt.assert_allclose(detection.bbox, expected_detection.bbox, atol=1e-5)
                    npt.assert_allclose(detection.confidence, expected_detection.confidence)
                    if expected_detection.depth is not None:
                        npt.assert_allclose(detection.depth, expected_detection.depth, atol=1e-5)


@pytest.mark.parametrize(
    "angles,weights,expected_angle",
    [(np.array([0.0, 1.0, 1.5]), None, 0.8513678),   # None --> equal weighting.
     (np.array([1.2, -0.5, -0.7]), np.array([0.1, 0.2, 0.3]), -0.41795065)
     ]
)
def test_mean_angle(angles, weights, expected_angle):
    """Test that the weighted average of angles is calculated properly.

    Also checks that the periodicity of 2*pi is taken into account.
    """
    # First, use given inputs.
    calculated_angle = mean_angle(angles=angles, weights=weights)
    assert np.allclose(calculated_angle, expected_angle)
    # Now, force a periodic shift.
    num_periods = np.random.randint(low=1, high=10)
    sign = np.random.choice([-1., 1.])
    shifted_angles = angles + sign * num_periods * 2. * np.pi
    calculated_angle = mean_angle(angles=shifted_angles, weights=weights)
    assert np.allclose(calculated_angle, expected_angle)
    # Check that the scaling of weights does not matter.
    if weights is not None:
        # Choose a random scaling factor.
        scale = np.random.uniform(low=0.2, high=5.0)
        calculated_angle = mean_angle(angles=angles, weights=scale*weights)
        assert np.allclose(calculated_angle, expected_angle)
