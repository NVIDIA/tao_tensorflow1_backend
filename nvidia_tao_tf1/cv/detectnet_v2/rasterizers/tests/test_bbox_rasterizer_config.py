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

"""Test BboxRasterizerConfig."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

from nvidia_tao_tf1.cv.detectnet_v2.rasterizers.bbox_rasterizer_config import BboxRasterizerConfig


@pytest.mark.parametrize(
    "cov_center_x,cov_center_y,cov_radius_x,cov_radius_y,bbox_min_radius,should_raise",
    [
     (-0.1, 0.1, 0.2, 0.3, 0.4, True),
     (0.5, -0.2, 0.6, 0.7, 0.8, True),
     (0.9, 0.11, -0.3, 0.12, 0.13, True),
     (0.14, 0.15, 0.16, -0.4, 0.17, True),
     (0.18, 0.19, 0.20, 0.21, -0.5, True),
     (1.1, 0.22, 0.23, 0.24, 0.25, True),
     (0.26, 1.2, 0.27, 0.28, 0.29, True),
     (0.30, 0.31, 1.3, 0.32, 0.33, False),
     (0.34, 0.35, 0.36, 1.4, 0.37, False),
     (0.38, 0.39, 0.40, 0.41, 1.5, False)
     ]
)
def test_target_class_config_init_ranges(cov_center_x, cov_center_y, cov_radius_x, cov_radius_y,
                                         bbox_min_radius, should_raise):
    """Test that BboxRasterizerConfig.TargetClassConfig raises ValueError on invalid values.

    Args:
        The first 5 are the same as for BboxRasterizerConfig.TargetClassConfig.__init__().
        should_raise (bool): Whether or not the __init__() should raise a ValueError.
    """
    if should_raise:
        with pytest.raises(ValueError):
            BboxRasterizerConfig.TargetClassConfig(cov_center_x, cov_center_y, cov_radius_x,
                                                   cov_radius_y, bbox_min_radius)
    else:
        BboxRasterizerConfig.TargetClassConfig(cov_center_x, cov_center_y, cov_radius_x,
                                               cov_radius_y, bbox_min_radius)


@pytest.mark.parametrize(
    "deadzone_radius,should_raise",
    [(-0.1, True), (0.1, False), (1.1, True), (0.9, False)]
)
def test_bbox_rasterizer_config_init_range(deadzone_radius, should_raise):
    """Test that BboxRasterizerConfig raises ValueError on invalid values.

    Args:
        The first one is the same as for BboxRasterizerConfig.__init__().
        should_raise (bool): Whether or not the __init__() should raise a ValueError.
    """
    if should_raise:
        with pytest.raises(ValueError):
            BboxRasterizerConfig(deadzone_radius)
    else:
        BboxRasterizerConfig(deadzone_radius)
