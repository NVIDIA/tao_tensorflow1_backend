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

"""Test BboxRasterizerConfig builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from unittest.mock import patch
from google.protobuf.text_format import Merge as merge_text_proto
import pytest

import nvidia_tao_tf1.cv.detectnet_v2.proto.bbox_rasterizer_config_pb2 as \
    bbox_rasterizer_config_pb2
from nvidia_tao_tf1.cv.detectnet_v2.rasterizers.build_bbox_rasterizer_config import (
    build_bbox_rasterizer_config
)


@pytest.fixture(scope='function')
def bbox_rasterizer_proto():
    bbox_rasterizer_proto = bbox_rasterizer_config_pb2.BboxRasterizerConfig()
    prototxt = """
target_class_config {
    key: "animal"
    value: {
        cov_center_x: 0.125,
        cov_center_y: 0.25,
        cov_radius_x: 0.375,
        cov_radius_y: 0.5,
        bbox_min_radius: 0.625
    }
}
target_class_config {
    key: "traffic_cone"
    value: {
        cov_center_x: 0.75,
        cov_center_y: 0.875,
        cov_radius_x: 1.0,
        cov_radius_y: 1.125,
        bbox_min_radius: 1.25
    }
}
deadzone_radius: 1.0
"""
    merge_text_proto(prototxt, bbox_rasterizer_proto)

    return bbox_rasterizer_proto


def test_build_bbox_rasterizer_config_keys(bbox_rasterizer_proto):
    """Test that build_bbox_rasterizer_config has the correct keys."""
    bbox_rasterizer_config = build_bbox_rasterizer_config(bbox_rasterizer_proto)
    assert set(bbox_rasterizer_config.keys()) == {'animal', 'traffic_cone'}


@patch(
    "nvidia_tao_tf1.cv.detectnet_v2.rasterizers.build_bbox_rasterizer_config.BboxRasterizerConfig"
)
def test_build_bbox_rasterizer_config_values(MockedBboxRasterizerConfig, bbox_rasterizer_proto):
    """Test that build_bbox_rasterizer_config translates a proto correctly."""
    build_bbox_rasterizer_config(bbox_rasterizer_proto)
    # Check it was called with the expected deadzone_radius values.
    MockedBboxRasterizerConfig.assert_called_with(1.0)
    # Now check the subclasses.
    # Check for "animal".
    # NOTE: these numbers are chosen to go around Python's default float being double precision.
    MockedBboxRasterizerConfig.TargetClassConfig.assert_any_call(
        0.125, 0.25, 0.375, 0.5, 0.625
    )
    # Check for "traffic_cone".
    MockedBboxRasterizerConfig.TargetClassConfig.assert_any_call(
        0.75, 0.875, 1.0, 1.125, 1.25
    )
