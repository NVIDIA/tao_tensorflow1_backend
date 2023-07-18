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

"""Tests for FpeNet Visualization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from nvidia_tao_tf1.cv.fpenet.visualization.fpenet_visualization import FpeNetVisualizer


def test_visualize(tmpdir):
    """Test FpeNetVisualizer call."""
    model_id = "model_42"
    checkpoint_dir = os.path.join(str(tmpdir), model_id)
    visualizer = FpeNetVisualizer(checkpoint_dir,
                                  num_images=3)

    image = tf.random.uniform(shape=[4, 1, 80, 80])
    kpts = tf.ones((4, 80, 2))  # batch_size, num_keypoints, num_dim (x, y)[20., 30.]
    visualizer.visualize_images(image, kpts)
