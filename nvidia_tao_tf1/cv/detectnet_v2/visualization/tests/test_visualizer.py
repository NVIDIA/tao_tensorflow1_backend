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

import os

import numpy as np
import pytest
import tensorflow as tf

import nvidia_tao_tf1.core
from nvidia_tao_tf1.cv.detectnet_v2.proto.visualizer_config_pb2 import VisualizerConfig
from nvidia_tao_tf1.cv.detectnet_v2.visualization.visualizer import \
    DetectNetTBVisualizer as Visualizer


def test_build_visualizer():
    """Test visualizer config parsing."""
    config = VisualizerConfig()

    # Default values should pass.
    Visualizer.build_from_config(config)

    config.enabled = True
    config.num_images = 3
    config.target_class_config['car'].coverage_threshold = 1.0
    Visualizer.build_from_config(config)
    assert Visualizer.enabled is True
    assert Visualizer.num_images == 3
    assert Visualizer.target_class_config['car'].coverage_threshold == 1.0


def test_draw_bboxes():
    """Test superposition of bounding boxes and input images."""
    batch_size = 1
    width = 6
    height = 6
    grid_width = 2
    grid_height = 2
    channels = 3

    # Get zero images as the input.
    input_images = np.zeros([batch_size, channels, width, height], dtype=np.float32)

    # All coverage 0 except 2.
    preds = np.zeros([batch_size, 1, grid_height, grid_width], dtype=np.float32)
    preds[0, 0, 0, 1] = 1.
    preds[0, 0, 1, 0] = 1.

    # Absolute coordinates.
    bbox_preds = np.zeros([batch_size, 4, grid_height, grid_width], dtype=np.float32)

    # A large bbox in the bottom left (cov 1).
    bbox_preds[0, :, 1, 0] = [1, 1, 6, 6]  # These indices start from 1 on the image plane.

    # A bbox on the top right (coverage 1).
    bbox_preds[0, :, 0, 1] = [1, 3, 3, 5]

    # Note that this bounding box has 0 coverage.
    bbox_preds[0, :, 1, 1] = [2, 2, 4, 4]

    output_images = Visualizer._draw_bboxes(input_images, preds, bbox_preds,
                                            coverage_threshold=0.5)

    with tf.Session().as_default():
        bbox_images = output_images.eval()

    # Check the function returns [NHWC] format.
    assert bbox_images.shape == (batch_size, height, width, channels)

    # Collapse the bounding box raster on single channel.
    # We do not know what color the tf bounding box drawing will assign to them, so we will
    # only check the binary values.
    bbox_black_black_white = np.int32(np.sum(bbox_images, 3) > 0)

    # 1 large bounding box between pixels 1-6 and a small one framed at pixels (1,3,3,5) expected
    # bounding box (2,2,4,4) drops due to coverage threshold.
    bounding_box_expected = np.array([[[1, 1, 1, 1, 1, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 1, 1, 0, 0, 1],
                                       [1, 0, 1, 0, 0, 1],
                                       [1, 1, 1, 0, 0, 1],
                                       [1, 1, 1, 1, 1, 1]]], dtype=np.int32)

    np.testing.assert_array_equal(bounding_box_expected, bbox_black_black_white)


@pytest.mark.skipif(os.getenv("RUN_ON_CI", "0") == "1", reason="Cannot be run on CI")
def test_visualize_elliptical_bboxes():
    """Test the visualize_elliptical_bboxes staticmethod."""
    target_class_names = ['class_1', 'class_2']
    num_classes = len(target_class_names)

    H, W = 5, 7
    input_image_shape = (2, 3, 2 * H, 2 * W)  # NCHW.
    input_images = \
        tf.constant(np.random.randn(*input_image_shape), dtype=np.float32)
    coverage_shape = (2, num_classes, 1, H, W)
    coverage = tf.constant(np.random.uniform(low=0.0, high=1.0, size=coverage_shape),
                           dtype=tf.float32)
    abs_bboxes_shape = (2, num_classes, 4, H, W)
    abs_bboxes = tf.constant(np.random.uniform(low=-1.0, high=1.0, size=abs_bboxes_shape),
                             dtype=tf.float32)

    # Build the Visualizer.
    config = VisualizerConfig()
    config.enabled = True
    config.num_images = 3
    Visualizer.build_from_config(config)

    Visualizer.visualize_elliptical_bboxes(
        target_class_names=target_class_names,
        input_images=input_images,
        coverage=coverage,
        abs_bboxes=abs_bboxes)

    # Check that we added the expected visualizations to Tensorboard.
    summaries = tf.get_collection(nvidia_tao_tf1.core.hooks.utils.INFREQUENT_SUMMARY_KEY)
    assert len(summaries) == num_classes
