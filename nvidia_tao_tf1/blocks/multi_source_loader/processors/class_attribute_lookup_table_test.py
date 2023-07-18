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
"""Tests for class_attribute_lookup_table.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from parameterized import parameterized

import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors.class_attribute_lookup_table import (  # noqa
    ClassAttributeLookupTable,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import Polygon2DLabel
from nvidia_tao_tf1.blocks.multi_source_loader.types.test_fixtures import (
    make_tags,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


def generate_polygon_2d_label(class_names, attribute_names):
    """
    Generate a polygon_2d_label with class_names and attribute_names.
    Args:
        class_names (nested lists of strings): Class names of polygons in the polygon_2d_label.
        attribute_names (nested lists of strings): Attribute names of polygons in the
            polygon_2d_label.

    Return:
        (polygon_2d_label): A polygon_2d_label.
    """
    return Polygon2DLabel(
        # Empty vertices
        vertices=tf.SparseTensor(
            indices=tf.reshape(tf.constant([], tf.int64), [0, 5]),
            values=tf.constant([], tf.int64),
            dense_shape=tf.constant([0, 0, 0, 0, 2], tf.int64),
        ),
        classes=make_tags(class_names),
        attributes=make_tags(attribute_names),
    )


class TestClassAttributeLookupTable(tf.test.TestCase):
    @tf.contrib.eager.run_test_in_graph_and_eager_modes
    def test_class_attribute_lookup(self):
        lookup_tables = {
            "attribute_mapping": {
                "ak1": 1,
                "ak2": 2,
                "ak3": 3,
                "ak4": 4,
                "ak5": 5,
                "ak6": 6,
            },
            "default_attribute_value": -1,
            "class_mapping": {
                "ck1": -1,
                "ck2": -2,
                "ck3": -3,
                "ck4": -4,
                "ck5": -5,
                "ck6": -6,
            },
            "default_class_value": 0,
        }
        class_attribute_lookup_table = ClassAttributeLookupTable(**lookup_tables)
        # Single example, single frame and two polygons in this frame. The first polygon
        # has one class and two attributes. The second polygon has one class and one attribute.
        polygon_2d_label = generate_polygon_2d_label(
            attribute_names=[[[["aK1", "ak3"], ["ak4"]]]],
            class_names=[[[["ck1"], [" ck2"]]]],
        )
        mapped_polygon_2d_label = class_attribute_lookup_table(polygon_2d_label)

        if not tf.executing_eagerly():
            self.evaluate(tf.compat.v1.tables_initializer())
        self.assertAllEqual([-1, -2], mapped_polygon_2d_label.classes.values)
        self.assertAllEqual([1, 3, 4], mapped_polygon_2d_label.attributes.values)

    @tf.contrib.eager.run_test_in_graph_and_eager_modes
    def test_class_attribute_lookup_not_in_table(self):
        lookup_tables = {
            "attribute_mapping": {
                "AK1": 1,
                "ak2": 2,
                "ak3": 3,
                "ak4": 4,
                "ak5": 5,
                "ak6": 6,
            },
            "default_attribute_value": -1,
            "class_mapping": {
                "ck1": -1,
                "ck2": -2,
                "ck3": -3,
                "ck4": -4,
                "ck5": -5,
                "ck6": -6,
            },
            "default_class_value": 0,
        }
        class_attribute_lookup_table = ClassAttributeLookupTable(**lookup_tables)
        polygon_2d_label = generate_polygon_2d_label(
            attribute_names=[[[["ak1", " ak3"], ["ak8"]]]],
            class_names=[[[["ck1"], ["ck0 "]]]],
        )
        mapped_polygon_2d_label = class_attribute_lookup_table(polygon_2d_label)

        if not tf.executing_eagerly():
            self.evaluate(tf.compat.v1.tables_initializer())
        self.assertAllEqual([-1, 0], mapped_polygon_2d_label.classes.values)
        self.assertAllEqual([1, 3, -1], mapped_polygon_2d_label.attributes.values)

    @tf.contrib.eager.run_test_in_graph_and_eager_modes
    def test_class_attribute_lookup_many_to_one(self):
        lookup_tables = {
            "attribute_mapping": {
                "ak1": 1,
                "ak2": 1,
                "Ak3": 1,
                "ak4": 4,
                "ak5": 5,
                "ak6": 6,
            },
            "default_attribute_value": -1,
            "class_mapping": {
                "ck1": -1,
                "ck2": -2,
                "ck3": -3,
                "ck4": -4,
                "ck5": -5,
                "ck6": -6,
            },
            "default_class_value": 0,
        }
        class_attribute_lookup_table = ClassAttributeLookupTable(**lookup_tables)
        polygon_2d_label = generate_polygon_2d_label(
            attribute_names=[[[["ak1", "ak2"], ["ak3"]]]],
            class_names=[[[["CK1"], ["ck2"]]]],
        )
        mapped_polygon_2d_label = class_attribute_lookup_table(polygon_2d_label)

        if not tf.executing_eagerly():
            self.evaluate(tf.compat.v1.tables_initializer())
        self.assertAllEqual([-1, -2], mapped_polygon_2d_label.classes.values)
        self.assertAllEqual([1, 1, 1], mapped_polygon_2d_label.attributes.values)

    @tf.contrib.eager.run_test_in_graph_and_eager_modes
    def test_class_attribute_trimming(self):
        lookup_tables = {
            "attribute_mapping": {"ak1": 1, "ak2": 2},
            "default_attribute_value": 0,
            "class_mapping": {"ck1": 1, "ck2 ": 2, "ck3": 3, "ck3      ": 3},
            "default_class_value": 0,
        }
        class_attribute_lookup_table = ClassAttributeLookupTable(**lookup_tables)
        polygon_2d_label = generate_polygon_2d_label(
            attribute_names=[[[[" ak1"], ["  ak2 "]]]],
            class_names=[[[["   ck3 "], ["ck1     "]]]],
        )
        mapped_polygon_2d_label = class_attribute_lookup_table(polygon_2d_label)

        if not tf.executing_eagerly():
            self.evaluate(tf.compat.v1.tables_initializer())
        self.assertAllEqual([1, 2], mapped_polygon_2d_label.attributes.values)
        self.assertAllEqual([3, 1], mapped_polygon_2d_label.classes.values)

    @tf.contrib.eager.run_test_in_graph_and_eager_modes
    def test_class_attribute_trimming_error(self):
        lookup_tables = {
            "attribute_mapping": {},
            "default_attribute_value": -1,
            "class_mapping": {"ck1": 1, "ck1  ": 2},
            "default_class_value": 0,
        }
        self.assertRaises(ValueError, ClassAttributeLookupTable, **lookup_tables)

    @tf.contrib.eager.run_test_in_graph_and_eager_modes
    def test_class_attribute_lower_error(self):
        lookup_tables = {
            "attribute_mapping": {},
            "default_attribute_value": -1,
            "class_mapping": {"CK1": 1, "ck1": 2},
            "default_class_value": 0,
        }
        self.assertRaises(ValueError, ClassAttributeLookupTable, **lookup_tables)

    @tf.contrib.eager.run_test_in_graph_and_eager_modes
    def test_class_attribute_lookup_empty_table(self):
        lookup_tables = {
            "attribute_mapping": {},
            "default_attribute_value": -1,
            "class_mapping": {},
            "default_class_value": 0,
        }
        class_attribute_lookup_table = ClassAttributeLookupTable(**lookup_tables)
        polygon_2d_label = generate_polygon_2d_label(
            attribute_names=[[[["ak1", "ak3"], ["ak4"]]]],
            class_names=[[[["ck1"], ["ck2"]]]],
        )
        mapped_polygon_2d_label = class_attribute_lookup_table(polygon_2d_label)
        if not tf.executing_eagerly():
            self.evaluate(tf.compat.v1.tables_initializer())
        self.assertAllEqual([0, 0], mapped_polygon_2d_label.classes.values)
        self.assertAllEqual([-1, -1, -1], mapped_polygon_2d_label.attributes.values)

    @tf.contrib.eager.run_test_in_graph_and_eager_modes
    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        lookup_tables = {
            "attribute_mapping": {"ak1": 1, "ak2": 2},
            "default_attribute_value": 0,
            "class_mapping": {"ck1": 1, "ck2 ": 2, "ck3": 3, "ck3      ": 3},
            "default_class_value": 0,
        }
        lookup_table = ClassAttributeLookupTable(**lookup_tables)
        lookup_table_dict = lookup_table.serialize()
        deserialized_lookup_table_dict = deserialize_tao_object(lookup_table_dict)

        self.assertAllEqual(
            lookup_table.class_keys, deserialized_lookup_table_dict.class_keys
        )
        self.assertAllEqual(
            lookup_table.class_values, deserialized_lookup_table_dict.class_values
        )
        self.assertAllEqual(
            lookup_table.default_class_value,
            deserialized_lookup_table_dict.default_class_value,
        )
        self.assertAllEqual(
            lookup_table.attribute_keys, deserialized_lookup_table_dict.attribute_keys
        )
        self.assertAllEqual(
            lookup_table.attribute_values,
            deserialized_lookup_table_dict.attribute_values,
        )

    @tf.contrib.eager.run_test_in_graph_and_eager_modes
    @parameterized.expand(
        [[{"ak1": 0}, ["ak1"], [0]], [None, ["ak1"], None], [None, ["ak1", "ak2"], [0]]]
    )
    def test_validate_key_value_lists_mapping(
        self, attribute_mapping, attribute_keys, attribute_values
    ):
        lookup_tables = {
            "attribute_mapping": attribute_mapping,
            "attribute_keys": attribute_keys,
            "attribute_values": attribute_values,
            "default_attribute_value": -1,
            "class_keys": ["ck1"],
            "class_values": [1],
            "default_class_value": 0,
        }
        self.assertRaises(ValueError, ClassAttributeLookupTable, **lookup_tables)
