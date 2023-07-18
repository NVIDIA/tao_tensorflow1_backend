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

"""Test ClusteringConfig builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.protobuf.text_format import Merge as merge_text_proto
import pytest

from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.clustering_config import build_clustering_config
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.clustering_config import ClusteringConfig
from nvidia_tao_tf1.cv.detectnet_v2.proto.experiment_pb2 import Experiment


@pytest.fixture(scope='function')
def experiment_proto():
    experiment_proto = Experiment()
    prototxt = """
    postprocessing_config {
      target_class_config {
        key: "car"
        value: {
          clustering_config {
            coverage_threshold: 0.5
            dbscan_eps: 0.125
            dbscan_min_samples: 1
            minimum_bounding_box_height: 4
            clustering_algorithm: DBSCAN
          }
        }
      }
      target_class_config {
        key: "pedestrian"
        value: {
          clustering_config {
            coverage_threshold: 0.25
            minimum_bounding_box_height: 2
            nms_iou_threshold: 0.40
            clustering_algorithm: NMS
          }
        }
     }
   }
   """

    merge_text_proto(prototxt, experiment_proto)

    return experiment_proto


def test_build_clustering_config(experiment_proto):
    """Test that clustering_config gets parsed correctly."""
    clustering_config = build_clustering_config(experiment_proto.postprocessing_config.
                                                target_class_config['car'].clustering_config)
    assert clustering_config.coverage_threshold == 0.5
    assert clustering_config.dbscan_eps == 0.125
    assert clustering_config.dbscan_min_samples == 1
    assert clustering_config.minimum_bounding_box_height == 4
    assert clustering_config.clustering_algorithm == "dbscan"
    clustering_config = build_clustering_config(experiment_proto.postprocessing_config.
                                                target_class_config['pedestrian'].clustering_config)
    assert clustering_config.coverage_threshold == 0.25
    assert clustering_config.minimum_bounding_box_height == 2
    assert clustering_config.clustering_algorithm == "nms"
    assert clustering_config.nms_iou_threshold


def test_clustering_config_limits():
    """Test that ClusteringConfig constructor raises correct errors."""
    # Invalid coverage_threshold.
    with pytest.raises(ValueError):
        ClusteringConfig(2.0, 0.5, 0.5, 1, 0, 0.4, 0.1, 0.2)

    # Invalid dbscan_eps.
    with pytest.raises(ValueError):
        ClusteringConfig(0.5, 2.0, 0.5, 1, 0, 0.2, 0.1, 0.2)

    # Invalid dbscan_min_samples.
    with pytest.raises(ValueError):
        ClusteringConfig(0.5, 0.5, -1.0, 1, 0, 0.2, 0.1, 0.2)

    # Invalid minimum_bounding_box_height.
    with pytest.raises(ValueError):
        ClusteringConfig(0.5, 0.5, 0.5, -1, 0, 0.2, 0.1, 0.2)

    with pytest.raises(ValueError):
        ClusteringConfig(0.5, 0.5, -1.0, -1, 1, 1.5, 0.1, 0.2)

    with pytest.raises(NotImplementedError):
        ClusteringConfig(0.5, 0.5, 0.75, 4, 2, 0.5, 0.1, 0.2)
