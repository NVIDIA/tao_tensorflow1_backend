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
"""Tests for instance_mapper.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors.instance_mapper import (
    InstanceMapper,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import Polygon2DLabel
import nvidia_tao_tf1.blocks.multi_source_loader.types.test_fixtures as fixtures

classes = [
    [
        [
            ["drivable-space"],
            ["vehicle:car"],
            ["vehicle:truck"],
            ["vehicle:Car "],
            ["person group"],
            ["avlp-person:person"],
            ["avlp-person:Person"],
        ]
    ],
    [
        [
            ["drivable-space"],
            ["avlp-person:person"],
            ["avlp-person:Person"],
            ["vehicle:Car "],
            ["person group"],
            ["avlp-person:person "],
            ["avlp-person:Person"],
        ]
    ],
]

attributes = [
    [[[], ["object_1"], ["object_1"], ["object_1"], [], ["object_1"], ["object_1"]]],
    [[["object_1"], ["object_2"], ["object_2"], [], [], ["object_1"], ["object_1"]]],
]

except_class = set(["drivable-space", "group"])


class TestInstanceMapper(tf.test.TestCase):
    def test_instance_mapping(self):
        with self.cached_session() as session:
            mapper = InstanceMapper(
                exceptions=except_class,
                default_has_instance=True,
                default_instance_id=0,
            )
            polygon_2d_label = Polygon2DLabel(
                vertices=None,  # vertices currently don't matter
                classes=fixtures.make_tags(classes),
                attributes=fixtures.make_tags(attributes),
            )
            mapped_polygon_2d_label = mapper(polygon_2d_label)
            session.run(tf.compat.v1.tables_initializer())
            self.assertAllEqual(
                [0, 1, 2, 1, 0, 3, 3, 0, 1, 1, 2, 0, 3, 3],
                mapped_polygon_2d_label.classes.values,
            )

    def test_instance_mapping_reverse(self):
        with self.cached_session() as session:
            mapper = InstanceMapper(
                exceptions=except_class,
                default_has_instance=False,
                default_instance_id=-1,
            )
            polygon_2d_label = Polygon2DLabel(
                vertices=None,  # vertices currently don't matter
                classes=fixtures.make_tags(classes),
                attributes=fixtures.make_tags(attributes),
            )
            mapped_polygon_2d_label = mapper(polygon_2d_label)
            session.run(tf.compat.v1.tables_initializer())
            self.assertAllEqual(
                [0, -1, -1, -1, 1, -1, -1, 0, -1, -1, -1, 1, -1, -1],
                mapped_polygon_2d_label.classes.values,
            )
