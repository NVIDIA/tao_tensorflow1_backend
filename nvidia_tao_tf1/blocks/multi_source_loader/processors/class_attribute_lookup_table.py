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
"""A lookup table for mapping input classes and attributes to output classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types import Polygon2DLabel
from nvidia_tao_tf1.core.coreobject import TAOObject, save_args
from nvidia_tao_tf1.core.processors import LookupTable


def _string_normalize(s):
    return s.strip().lower()


def _normalize_reduce(keys, values):
    """Returns a tuple of (keys,values) with unique keys after string trimming."""
    reduced = dict()
    for k, v in zip(keys, values):
        strip_key = _string_normalize(k)
        if strip_key not in reduced:
            reduced[strip_key] = v
        elif reduced[strip_key] != v:
            raise ValueError(
                "Error: Duplicate keys after trimming had different values."
            )
    keys = []
    values = []
    for k, v in reduced.items():
        keys.append(k)
        values.append(v)
    return keys, values


def _tf_normalize(x):
    y = np.empty(x.shape, dtype=x.dtype)
    for i, val in enumerate(x):
        y[i] = _string_normalize(val)
    return y


def _normalize_tf_strings(inputs):
    return tf.compat.v1.py_func(_tf_normalize, [inputs], tf.string, stateful=False)


def _validate_key_value_lists(mapping, key_list, value_list):
    if key_list and mapping:
        raise ValueError(
            "Specify the key and values as separate lists, or a dictionary"
            " that maps keys to values, not both."
        )

    if mapping:
        return list(mapping.keys()), list(mapping.values())

    if not key_list and not value_list:
        return [], []

    if not (key_list and value_list):
        raise ValueError("Both lists need to be specified.")

    if len(key_list) != len(value_list):
        raise ValueError("Keys and values must have same length.")
    return key_list, value_list


class ClassAttributeLookupTable(TAOObject):
    """A lookup table for mapping input classes and attributes to output classes."""

    # TODO(mlehr): Delete the key and value lists once all the specs are updated.
    @save_args
    def __init__(
        self,
        default_attribute_value,
        default_class_value,
        attribute_mapping=None,
        class_mapping=None,
        attribute_keys=None,
        attribute_values=None,
        class_keys=None,
        class_values=None,
        **kwargs
    ):
        """
        Construct a ClassAttributeLookupTable class.

        Args:
            default_attribute_value (int): Default value for attribute lookup table.
            default_class_value (int): Default value for class lookup table.
            attribute_mapping (dict): Mapping from attribute names to attribute ids.
            class_mapping (dict): Mapping from class names to class ids.
            attribute_keys (list of strings): Keys of attirbute lookup table.
            attribute_values (list of strings): Values of attribute lookup table.
            class_keys (list of strings): Keys of class_id lookup table.
            class_values (list of strings): Values of class_id lookup table.

        Raises:
            ValueError: If multiple ``keys`` after string normalizing become duplicates, but their
                        ``values`` are different. The keys for attributes, as well as classes,
                        are normalized.
                        Example:
                            keys = ["cls1", " cLs1  "]
                            values = [1, 2]

                        Current string normalization includes whitespace trimming on both ends,
                        as well as case insensitivity.
        """
        super(ClassAttributeLookupTable, self).__init__(**kwargs)

        attribute_keys, attribute_values = _validate_key_value_lists(
            attribute_mapping, attribute_keys, attribute_values
        )
        class_keys, class_values = _validate_key_value_lists(
            class_mapping, class_keys, class_values
        )

        self.class_keys, self.class_values = _normalize_reduce(class_keys, class_values)
        self.default_class_value = default_class_value
        self.attribute_keys, self.attribute_values = _normalize_reduce(
            attribute_keys, attribute_values
        )
        self.default_attribute_value = default_attribute_value

    def __call__(self, polygon_2d_label):
        """Map text class and attribute names to numeric ids.

        Args:
            polygon_2d_label (Polygon2DLabel): A label containing 2D polygons/polylines and their
                associated classes and attributes. The first two dimensions of each tensor
                that this structure contains should be batch/example(B) followed by a frame/time
                dimension(T). The rest of the dimensions encode type specific information. See
                Polygon2DLabel documentation for details

        Returns:
            (Polygon2DLabel): The label with the classes and attributes mapped to numeric ids.
        """
        if self.class_keys:
            class_lookup_table = LookupTable(
                keys=self.class_keys,
                values=self.class_values,
                default_value=self.default_class_value,
            )
        else:
            class_lookup_table = None

        if self.attribute_keys:
            attribute_lookup_table = LookupTable(
                keys=self.attribute_keys,
                values=self.attribute_values,
                default_value=self.default_attribute_value,
            )
        else:
            attribute_lookup_table = None

        classes = polygon_2d_label.classes
        if class_lookup_table is not None:
            trim_values = _normalize_tf_strings(classes.values)
            mapped_class_ids = class_lookup_table(trim_values)
        else:
            mapped_class_ids = (
                tf.ones_like(classes.values, dtype=tf.int32) * self.default_class_value
            )

        attributes = polygon_2d_label.attributes
        if attribute_lookup_table is not None:
            trim_attributes = _normalize_tf_strings(attributes.values)
            mapped_attribute_ids = attribute_lookup_table(trim_attributes)
        else:
            mapped_attribute_ids = (
                tf.ones_like(attributes.values, dtype=tf.int32)
                * self.default_attribute_value
            )

        return Polygon2DLabel(
            vertices=polygon_2d_label.vertices,
            classes=tf.SparseTensor(
                values=mapped_class_ids,
                indices=classes.indices,
                dense_shape=classes.dense_shape,
            ),
            attributes=tf.SparseTensor(
                values=mapped_attribute_ids,
                indices=attributes.indices,
                dense_shape=attributes.dense_shape,
            ),
        )
