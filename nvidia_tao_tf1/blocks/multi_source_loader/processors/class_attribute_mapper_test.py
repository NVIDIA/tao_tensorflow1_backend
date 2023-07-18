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
"""Tests for class_attribute_mapper.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors.class_attribute_mapper import (
    ClassAttributeMapper,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import Polygon2DLabel
import nvidia_tao_tf1.blocks.multi_source_loader.types.test_fixtures as fixtures
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object

classes = [  # Examples
    [[["path1"], ["path2"]]],  # Frames  # Shapes  # Tags
    [[["path1"], ["path1"]]],  # Frames  # Shapes  # Tags
    [[["path0"]]],  # Frames  # Shapes  # Tags
    [[[]]],  # Frames  # Shapes  # Tags
]
attributes = [  # Examples
    [  # Frames
        [["LefT EdGe", "Left Exit"], ["Left edge", "Left exit"]]  # Shapes  # Tags
    ],
    [  # Frames
        [  # Shapes
            ["Left Edge", "Left Exit", "different attribute"],  # Tags
            ["Left Edge", "Left Exit"],
        ]
    ],
    [[["Left Edge", "Left Exit"]]],  # Frames  # Shapes  # Tags
    [[["Left Edge"]]],  # Frames  # Shapes  # Tags
]

class_attribute_mapping = [
    {
        "match_any_class": ["path 0", "path0", "path zero"],
        "match_any_attribute": ["Left Edge", "Left Exit"],
        "class_name": "path 0",
        "class_id": 0,
    },
    {
        "match_any_class": ["path 1", "path1", "path one"],
        "match_all_attributes": ["Left Edge", "Left Exit"],
        "class_name": "path 1",
        "class_id": 1,
    },
    {
        "match_any_class": ["path 1", "path1", "path one"],
        "match_all_attributes": ["Left Edge", "Left Exit"],
        "match_all_attributes_allow_others": True,
        "class_name": "others allowed",
        "class_id": 2,
    },
    {
        "match_any_class": [],
        "match_all_attributes": ["Left Edge"],
        "match_all_attributes_allow_others": True,
        "class_name": "others allowed",
        "class_id": 3,
    },
]

class_only_mapping = [
    {
        "match_any_class": ["path 0", "path0", "path zero"],
        "class_name": "path 0",
        "class_id": 0,
    },
    {
        "match_any_class": ["path 1", "path1", "path one"],
        "class_name": "path 1",
        "class_id": 1,
    },
]

attribute_mapping = {"Left Edge": 1, "Left Exit": 2}


class TestClassAttributeMapper(tf.test.TestCase):
    def test_class_attribute_mapping(self):
        with self.cached_session() as sess:
            mapper = ClassAttributeMapper(
                class_attribute_mapping, "Default", -1, attribute_mapping, -1
            )
            polygon_2d_label = Polygon2DLabel(
                vertices=None,  # vertices currently don't matter
                classes=fixtures.make_tags(classes),
                attributes=fixtures.make_tags(attributes),
            )
            mapped_polygon_2d_label = mapper(polygon_2d_label)
            sess.run(tf.compat.v1.tables_initializer())
            self.assertAllEqual(
                [1, -1, 2, 1, 0, 3], mapped_polygon_2d_label.classes.values
            )
            self.assertAllEqual(
                [1, 2, 1, 2, -1, 1, 2, 1, 2, 1, 2, 1],
                mapped_polygon_2d_label.attributes.values,
            )

    # Empty attributes arrays for cases like polenet that only have classes
    def test_empty_attributes_mapping(self):
        with self.cached_session() as sess:
            empty_attributes = tf.SparseTensor(
                indices=tf.zeros((0, 4), tf.int64),
                values=tf.constant([], dtype=tf.string),
                dense_shape=tf.constant((0, 0, 0, 0), dtype=tf.int64),
            )

            polygon_2d_label = Polygon2DLabel(
                vertices=None,  # vertices currently don't matter
                classes=fixtures.make_tags(classes),
                attributes=empty_attributes,
            )
            mapper = ClassAttributeMapper(
                class_attribute_mapping, "Default", -1, attribute_mapping, -1
            )
            mapped_polygon_2d_label = mapper(polygon_2d_label)
            sess.run(tf.compat.v1.tables_initializer())
            self.assertAllEqual(
                [-1, -1, -1, -1, -1], mapped_polygon_2d_label.classes.values
            )

    def test_class_only_mapping(self):
        with self.cached_session() as sess:
            mapper = ClassAttributeMapper(class_only_mapping, "Default", -1, {}, -1)
            polygon_2d_label = Polygon2DLabel(
                vertices=None,  # vertices currently don't matter
                classes=fixtures.make_tags(classes),
                attributes=fixtures.make_tags(attributes),
            )
            mapped_polygon_2d_label = mapper(polygon_2d_label)
            sess.run(tf.compat.v1.tables_initializer())
            self.assertAllEqual(
                [1, -1, 1, 1, 0, -1], mapped_polygon_2d_label.classes.values
            )
            self.assertAllEqual(
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                mapped_polygon_2d_label.attributes.values,
            )

    def test_class_attribute_mapping_removed_attributes(self):
        with self.cached_session() as sess:
            updated_class_attribute_mapping = []
            for mapping in class_attribute_mapping:
                updated_mapping = mapping.copy()
                updated_mapping.update({"remove_matched_attributes": True})
                updated_class_attribute_mapping.append(updated_mapping)
            mapper = ClassAttributeMapper(
                updated_class_attribute_mapping, "Default", -1, attribute_mapping, -1
            )
            polygon_2d_label = Polygon2DLabel(
                vertices=None,  # vertices currently don't matter
                classes=fixtures.make_tags(classes),
                attributes=fixtures.make_tags(attributes),
            )
            mapped_polygon_2d_label = mapper(polygon_2d_label)
            sess.run(tf.compat.v1.tables_initializer())
            self.assertAllEqual(
                [1, -1, 2, 1, 0, 3], mapped_polygon_2d_label.classes.values
            )
            self.assertAllEqual([1, 2, -1], mapped_polygon_2d_label.attributes.values)

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        mapper = ClassAttributeMapper(
            class_attribute_mapping, "Default", -1, attribute_mapping, -1
        )
        mapper_dict = mapper.serialize()

        deserialized_mapper = deserialize_tao_object(mapper_dict)

        self.assertAllEqual(
            mapper._attribute_mappings, deserialized_mapper._attribute_mappings
        )
        self.assertAllEqual(
            mapper._default_attribute_id, deserialized_mapper._default_attribute_id
        )
        self.assertAllEqual(
            mapper._default_class_name, deserialized_mapper._default_class_name
        )
