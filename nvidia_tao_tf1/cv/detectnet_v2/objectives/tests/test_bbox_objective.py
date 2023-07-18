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

"""Test BboxObjective."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.cv.detectnet_v2.objectives.build_objective import build_objective
from nvidia_tao_tf1.cv.detectnet_v2.objectives.tests.test_objective import TestObjective
from nvidia_tao_tf1.cv.detectnet_v2.proto.model_config_pb2 import ModelConfig


class TestBboxObjective(TestObjective):
    """Bbox objective tests."""

    @pytest.fixture()
    def bbox_objective(self):
        """A BboxObjective instance."""

        # Build the bbox objective
        bbox_config = ModelConfig.BboxObjective()
        bbox_config.scale = 35.0
        bbox_config.offset = 0.5
        bbox_objective = build_objective('bbox', self.output_height, self.output_width,
                                         self.input_height, self.input_width,
                                         objective_config=bbox_config)

        return bbox_objective

    @pytest.mark.parametrize("label_coords,expected_coords",
                             [
                                 # Regular bbox, values should not change
                                 ([1.1, 21.1, 42.1, 66.666],
                                  [1.1, 21.1, 42.1, 66.666]),
                                 # Overly large, bbox should clip to image boundaries
                                 ([-10.1, -30.0, 150.0, 140.0],
                                     [0.0, 0.0, 5*16, 6*16]),
                                 # Negative height, should clip to max vertical coordinate
                                 ([-10.1, 30.2, 50.0, 20.2],
                                     [0.0, 30.2, 50.0, 30.2]),
                                 # Negative width, should clip to max horizontal coordinate
                                 ([30.2, 50.0, 20.2, 60.0],
                                     [30.2, 50.0, 30.2, 60.0]),
                             ])
    def test_bbox_roundtrip(self, bbox_objective, label_coords, expected_coords):
        """Test bbox label roundtrip through the objective.

        - Start from a label
        - Construct the target gradient for it
        - Rasterize the target gradient
        - Transform the rasterized tensor to absolute coordinates
        - Check the result matches the original label.
        """
        # Set up the coordinate tensor and target label
        scale_x = self.output_width / self.input_width
        scale_y = self.output_height / self.input_height
        coords_list = [label_coords[0] * scale_x,
                       label_coords[1] * scale_y,
                       label_coords[2] * scale_x,
                       label_coords[3] * scale_y]
        coords = tf.reshape(tf.constant(coords_list), [4, 1])
        label = {'target/output_space_coordinates': coords}

        self.check_label_roundtrip(bbox_objective, label, expected_coords)

    def test_bbox_predictions_to_absolute_coordinates(self, bbox_objective):
        """Test bbox prediction transform to absolute coordinates.

        Test that the bbox prediction tensor in gridcell center relative ltrb format
        gets transformed to absolute coordinates.
        """
        num_classes = 1
        output_width = self.output_width
        output_height = self.output_height
        input_width = self.input_width
        input_height = self.input_height
        stride = input_height / output_height
        bbox_scale = bbox_objective.scale
        bbox_offset = bbox_objective.offset

        # (batch size, num_classes, channels, height, width)
        ltrb = tf.ones((1, num_classes, 4, output_height, output_width))

        # note that the output absolute coordinates are clipped to image borders
        absolute_coordinates = bbox_objective.predictions_to_absolute(ltrb)

        # compute gridcell center coordinates
        x = np.arange(0, output_width, dtype=np.float32) * stride + bbox_offset
        x = np.tile(x, (output_height, 1))
        y = np.arange(0, output_height, dtype=np.float32) * \
            stride + bbox_offset
        y = np.transpose(np.tile(y, (output_width, 1)))

        # all ltrb values are ones, hence add +/- bbox_scale to the expected values
        x1 = x - bbox_scale
        y1 = y - bbox_scale
        x2 = x + bbox_scale
        y2 = y + bbox_scale

        # clip
        x1 = np.minimum(np.maximum(x1, 0.), input_width)
        y1 = np.minimum(np.maximum(y1, 0.), input_height)
        x2 = np.minimum(np.maximum(x2, x1), input_width)
        y2 = np.minimum(np.maximum(y2, y1), input_height)

        expected_absolute_coordinates = np.stack((x1, y1, x2, y2), axis=0)

        with tf.Session() as session:
            absolute_coordinates_res = session.run([absolute_coordinates])
            np.testing.assert_allclose(absolute_coordinates_res[0][0][0],
                                       expected_absolute_coordinates)
