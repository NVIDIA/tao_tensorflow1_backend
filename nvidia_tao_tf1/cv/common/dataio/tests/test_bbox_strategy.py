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

"""Test BoundingBoxStrategy concrete classes."""

import unittest
from nvidia_tao_tf1.cv.common.dataio.bbox_strategy import EyeBboxStrategy, FaceBboxStrategy


class FaceBboxStrategyTest(unittest.TestCase):
    """Test FaceBboxStrategy."""

    def test_get_square_bbox(self):
        face_bbox_strategy = FaceBboxStrategy(
            1280, 800,
            400, 200, 100, 150)

        left, upper, side_len = face_bbox_strategy.get_square_bbox()

        self.assertEqual(left, 352)
        self.assertEqual(upper, 177)
        self.assertEqual(side_len, 195)

    def test_get_square_bbox_sanity_clamp(self):
        face_bbox_strategy = FaceBboxStrategy(
            1280, 800,
            1180, 0, 100, 150)

        left, upper, side_len = face_bbox_strategy.get_square_bbox()

        self.assertEqual(left, 1085)
        self.assertEqual(upper, 0)
        self.assertEqual(side_len, 194)


class EyeBboxStrategyTest(unittest.TestCase):
    """Test EyeBboxStrategy."""

    def test_get_square_bbox(self):
        eye_bbox_strategy = EyeBboxStrategy(
            1280, 800,
            [400, 200, 410, 210])

        left, upper, w, h = eye_bbox_strategy.get_square_bbox()

        self.assertEqual(left, 399)
        self.assertEqual(upper, 199)
        self.assertEqual(w, 10)
        self.assertEqual(h, 10)

    def test_get_square_bbox_sanity_clamp(self):
        eye_bbox_strategy = EyeBboxStrategy(
            1280, 800,
            [1270, 0, 1280, 10])

        left, upper, w, h = eye_bbox_strategy.get_square_bbox()

        self.assertEqual(left, 1269)
        self.assertEqual(upper, 0)
        self.assertEqual(w, 9)
        self.assertEqual(h, 10)
