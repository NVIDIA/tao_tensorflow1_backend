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

"""Tests for RandomGlimpse processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from mock import patch


from parameterized import parameterized
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_glimpse import (
    RandomGlimpse,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Example,
    FEATURE_CAMERA,
    LABEL_MAP,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


def _esc(message):
    """Escape passed in string for regular expressions."""
    return re.escape(message)


class TestRandomGlimpseConstruction(ProcessorTestCase):
    @parameterized.expand([[1, 1], [1, 2], [2, 1]])
    def test_valid_bounds_do_not_raise(self, height, width):
        RandomGlimpse(
            crop_location=RandomGlimpse.CENTER,
            crop_probability=0.5,
            height=height,
            width=width,
        )

    def test_raises_on_invalid_crop_location(self):
        with self.assertRaisesRegexp(
            ValueError,
            _esc(
                "RandomGlimpse.crop_location 'under' is not \
supported. Valid options: center, random."
            ),
        ):
            RandomGlimpse(crop_location="under", height=7, width=7)

    @parameterized.expand([[RandomGlimpse.CENTER], [RandomGlimpse.RANDOM]])
    def test_valid_crop_location_does_not_raise(self, crop_location):
        RandomGlimpse(crop_location=crop_location, height=12, width=24)

    @parameterized.expand([[0.0], [0.5], [1.0]])
    def test_accepts_valid_probability(self, probability):
        RandomGlimpse(
            crop_location=RandomGlimpse.CENTER,
            crop_probability=probability,
            height=7,
            width=7,
        )


class TestRandomGlimpseProcessing(ProcessorTestCase):
    def test_scales_to_requested_height_and_width(self):
        example = self.make_example_128x240()
        expected_frames = tf.ones((1, 64, 120, 3))
        expected_labels = self.make_polygon_label([[60, 0.0], [120, 64], [0.0, 64]])

        with self.test_session():
            random_glimpse = RandomGlimpse(
                crop_location=RandomGlimpse.CENTER,
                crop_probability=0.0,
                height=64,
                width=120,
            )
            scaled = random_glimpse.process(example)

            self.assertAllClose(
                expected_frames.eval(), scaled.instances[FEATURE_CAMERA].eval()
            )
            self.assert_labels_close(expected_labels, scaled.labels[LABEL_MAP])

    def test_crops_center_crop(self):
        frames = tf.ones((1, 128, 240, 3))
        labels = self.make_polygon_label(
            vertices=[[61, 33], [179, 33], [179, 95], [61, 95]]
        )
        example = Example(
            instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: labels}
        )

        expected_frames = tf.ones((1, 64, 120, 3))
        # Center crop from 128x240 to 64x120 moves Xs 60 to the left and Ys 32 towards the top.
        expected_labels = self.make_polygon_label(
            vertices=[[1.0, 1.0], [119.0, 1.0], [119.0, 63.0], [1.0, 63.0]]
        )

        with self.test_session():
            random_glimpse = RandomGlimpse(
                crop_location=RandomGlimpse.CENTER,
                crop_probability=1.0,
                height=64,
                width=120,
            )
            cropped = random_glimpse.process(example)

            self.assertAllClose(
                expected_frames.eval(), cropped.instances[FEATURE_CAMERA].eval()
            )
            self.assert_labels_close(expected_labels, cropped.labels[LABEL_MAP])

    @patch("modulus.processors.augment.random_glimpse.tf.random.uniform")
    def test_crops_random_location(self, mocked_random_uniform):
        frames = tf.ones((1, 128, 240, 3))
        labels = self.make_polygon_label(
            vertices=[[60, 32], [180, 32], [180, 96], [60, 96]]
        )
        example = Example(
            instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: labels}
        )

        # All calls to random_uniform will return 0.5 causing left_x and top_y equal to 0.5.
        # This causes random crop to happen always from (0.5, 0.5) to (64.5, 120.5) which is
        # achieved by shifting all points 0.5 pixels towards the left and top. Always returning
        # 0.5 causes crop to always be performed as 0.5 < 1.0.
        mocked_random_uniform.return_value = tf.constant(0.5, dtype=tf.float32)
        expected_frames = tf.ones((1, 64, 120, 3))
        expected_labels = self.make_polygon_label(
            vertices=[[59.5, 31.5], [179.5, 31.5], [179.5, 95.5], [59.5, 95.5]]
        )

        with self.test_session():
            random_glimpse = RandomGlimpse(
                crop_location=RandomGlimpse.RANDOM,
                crop_probability=1.0,
                height=64,
                width=120,
            )
            cropped = random_glimpse.process(example)

            self.assertAllClose(
                expected_frames.eval(), cropped.instances[FEATURE_CAMERA].eval()
            )
            self.assert_labels_close(expected_labels, cropped.labels[LABEL_MAP])

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        random_glimpse = RandomGlimpse(
            crop_location=RandomGlimpse.RANDOM,
            crop_probability=1.0,
            height=64,
            width=120,
        )
        random_glimpse_dict = random_glimpse.serialize()
        deserialized_random_glimpse = deserialize_tao_object(random_glimpse_dict)
        self.assertEqual(
            str(random_glimpse._transform), str(deserialized_random_glimpse._transform)
        )
