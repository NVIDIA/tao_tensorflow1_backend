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

"""Test PostProcessingConfig builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.protobuf.text_format import Merge as merge_text_proto
import pytest

from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.postprocessing_config import (
    build_postprocessing_config
)
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
          confidence_config {
            confidence_threshold: 0.75
            confidence_model_filename: "car_mlp.hdf5"
          }
        }
      }
      target_class_config {
        key: "pedestrian"
        value: {
          clustering_config {
            coverage_threshold: 0.25
            dbscan_eps: 0.25
            dbscan_min_samples: 1
            minimum_bounding_box_height: 2
            clustering_algorithm: DBSCAN
          }
          confidence_config {
            confidence_threshold: 0.5
            confidence_model_filename: "pedestrian_mlp.hdf5"
          }
        }
     }
   }
   """

    merge_text_proto(prototxt, experiment_proto)

    return experiment_proto


def test_build_postprocessing_config(experiment_proto):
    """Test that postprocessing_config gets parsed correctly."""
    postprocessing_config = build_postprocessing_config(experiment_proto.postprocessing_config)
    assert 'car' in postprocessing_config
    assert 'pedestrian' in postprocessing_config
    assert len(postprocessing_config) == 2
    assert postprocessing_config['car'].clustering_config.coverage_threshold == 0.5
    assert postprocessing_config['car'].clustering_config.dbscan_eps == 0.125
    assert postprocessing_config['car'].clustering_config.dbscan_min_samples == 1
    assert postprocessing_config['car'].clustering_config.minimum_bounding_box_height == 4
    assert postprocessing_config['car'].clustering_config.clustering_algorithm == "dbscan"
    assert postprocessing_config['car'].confidence_config.confidence_threshold == 0.75
    assert postprocessing_config['car'].confidence_config.confidence_model_filename == \
        "car_mlp.hdf5"
    assert postprocessing_config['pedestrian'].clustering_config.coverage_threshold == 0.25
    assert postprocessing_config['pedestrian'].clustering_config.dbscan_eps == 0.25
    assert postprocessing_config['pedestrian'].clustering_config.dbscan_min_samples == 1
    assert postprocessing_config['pedestrian'].clustering_config.clustering_algorithm == "dbscan"
    assert postprocessing_config['pedestrian'].clustering_config.minimum_bounding_box_height == 2
    assert postprocessing_config['pedestrian'].confidence_config.confidence_threshold == 0.5
    assert postprocessing_config['pedestrian'].confidence_config.confidence_model_filename == \
        "pedestrian_mlp.hdf5"
