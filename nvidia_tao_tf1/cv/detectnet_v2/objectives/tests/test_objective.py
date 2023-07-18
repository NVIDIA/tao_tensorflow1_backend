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

"""Objective tests base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import zip
import tensorflow as tf

import nvidia_tao_tf1.core
from nvidia_tao_tf1.cv.detectnet_v2.rasterizers.bbox_rasterizer import BboxRasterizer


class TestObjective(object):
    """Objective tests base class."""

    output_width = 5
    output_height = 6
    stride = 16
    input_width = output_width * stride
    input_height = output_height * stride

    def check_label_roundtrip(self, objective, label, expected_values):
        """Test label roundtrip through the objective.

        - Start from a label
        - Construct the target gradient for it
        - Rasterize the target gradient
        - Transform the rasterized tensor to absolute coordinates
        - Check the result matches the original label.
        """
        coords = label['target/output_space_coordinates']
        # Form the gradients
        gradients = objective.target_gradient(label)

        # Rasterize the gradients. Use a small min radius to force
        # some output even for degenerate boxes.
        num_classes = 1
        num_images = 1
        matrices, _, _ = \
            BboxRasterizer.bbox_from_rumpy_params(xmin=coords[0],
                                                  ymin=coords[1],
                                                  xmax=coords[2],
                                                  ymax=coords[3],
                                                  cov_radius_x=tf.constant(
                                                      [1.0]),
                                                  cov_radius_y=tf.constant(
                                                      [1.0]),
                                                  bbox_min_radius=tf.constant(
                                                      [1.0]),
                                                  cov_center_x=tf.constant(
                                                      [0.5]),
                                                  cov_center_y=tf.constant(
                                                      [0.5]),
                                                  deadzone_radius=tf.constant([1.0]))
        bbox_rasterizer = nvidia_tao_tf1.core.processors.BboxRasterizer()
        gradient_flags = [objective.gradient_flag] * objective.num_channels
        tensor = bbox_rasterizer(num_images=num_images,
                                 num_classes=num_classes,
                                 num_gradients=objective.num_channels,
                                 image_height=self.output_height,
                                 image_width=self.output_width,
                                 bboxes_per_image=[1],
                                 bbox_class_ids=[0],
                                 bbox_matrices=matrices,
                                 bbox_gradients=gradients,
                                 bbox_coverage_radii=[[1.0, 1.0]],
                                 bbox_flags=[
                                     nvidia_tao_tf1.core.processors.BboxRasterizer.DRAW_MODE_ELLIPSE],
                                 gradient_flags=gradient_flags)

        # Give the tensor its shape, otherwise predictions_to_absolute does not work
        tensor = tf.reshape(tensor, (num_images, num_classes, objective.num_channels,
                                     self.output_height, self.output_width))

        # Transform to absolute coordinates
        abs_tensor = objective.predictions_to_absolute(tensor)

        # Check all output channels are as expected
        abs_tensors_per_channel = tf.unstack(abs_tensor, axis=2)
        # Assuming gt tensor is non-zero only outside the ellipse
        ellipse_mask = tf.not_equal(tensor[:, :, 0], 0.0)
        with tf.Session() as session:
            for ref_value, test_tensor in zip(expected_values, abs_tensors_per_channel):
                # Test only the values within the ellipse
                test_values_tensor = tf.boolean_mask(test_tensor, ellipse_mask)
                test_values = session.run(test_values_tensor)
                # Check that we actually rasterized something
                assert test_values.size, "No non-zero target tensor values found"
                # Default tolerance is too strict. 1e-6 would already fail
                np.testing.assert_allclose(test_values, ref_value, atol=1e-5)
