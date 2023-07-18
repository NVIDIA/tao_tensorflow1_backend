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

"""Test visualizations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

from nvidia_tao_tf1.cv.common.proto.visualizer_config_pb2 import VisualizerConfig

from nvidia_tao_tf1.cv.common.visualizer.tensorboard_visualizer import \
    TensorBoardVisualizer as Visualizer


def test_build_visualizer():
    """Test visualizer config parsing."""
    config = VisualizerConfig()

    # Default values should pass.
    Visualizer.build_from_config(config)

    config.enabled = True
    config.num_images = 3
    Visualizer.build_from_config(config)
    assert Visualizer.enabled is True
    assert Visualizer.num_images == 3


def test_nonbuilt_visualizer():
    """Test that the visualizer needs to be built."""
    Visualizer._built = False
    with pytest.raises(RuntimeError):
        Visualizer.enabled


def test_singleton_behavior():
    """Test the visualizer context handler for disabling visualizations."""
    config = VisualizerConfig()

    config.enabled = False
    Visualizer.build_from_config(config)

    # Disabling a disabled visualizer should keep it disabled.
    with Visualizer.disable():
        assert not Visualizer.enabled, "Visualizer is enabled with Visualizer().disabled."
    assert not Visualizer.enabled, \
        "Disabled Visualizer is enabled after returning from disabled state."

    # Enable the visualizer and check the Disabling context manager.
    config.enabled = True
    Visualizer.build_from_config(config)
    with Visualizer.disable():
        assert not Visualizer.enabled, "Visualizer is enabled with Visualizer.disabled."
    assert Visualizer.enabled, "Visualizer is not enabled after returning from disabled state."
