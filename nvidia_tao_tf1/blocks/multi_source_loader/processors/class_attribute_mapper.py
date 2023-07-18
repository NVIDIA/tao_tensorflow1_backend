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
"""Class for mapping input classes and attributes to output classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors import sparse_generators
from nvidia_tao_tf1.blocks.multi_source_loader.types import Polygon2DLabel
from nvidia_tao_tf1.core.coreobject import TAOObject, save_args

logger = logging.getLogger(__name__)


class ClassAttributeMapper(TAOObject):
    """ClassAttributeMapper maps input classes and attributes to output classes."""

    @save_args
    def __init__(
        self,
        output_class_mappings,
        default_class_name,
        default_class_id,
        attribute_mappings,
        default_attribute_id,
        **kwargs
    ):
        """Construct a ClassAttributeMapper.

        Args
            output_class_mappings(ordered collection of output class maps): The output class
                mappings to be applied.

            default_class_name (string): The default class name. To be applied if not of the
                output_class_mappings match.

            default_class_id (int): The default class id. To be applied if not of the
                output_class_mappings match.

            attribute_mappings(dict of attribute name -> id mappings): The attribute name to id
                mappings to be applied.

            default_attribute_id(int): The default attribute id, To be applied if not in the
                attribute_mappings
        """
        self._attribute_mappings = {
            k.strip().lower(): v for k, v in attribute_mappings.items()
        }
        self._default_attribute_id = default_attribute_id

        self._default_class_id = default_class_id
        self._default_class_name = default_class_name
        self._class_matchers = [_Matcher(**oc) for oc in output_class_mappings]

        super(ClassAttributeMapper, self).__init__(**kwargs)

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
        # Classes are stored in a 4D tensor of shape [B, T, S, C], where
        # B=Batch (example within batch), T=Time step, S=Shape, C=Class(always 1)
        classes = polygon_2d_label.classes

        # Attributes are stored in a 4D tensor of shape [B, T, S, A ], where
        # B=Batch (example within batch), T=Time step, S=Shape, A=Attribute
        # 0 or more attributes per shape
        attributes = polygon_2d_label.attributes

        # Want to match all indices other than the class/attribute index.
        index_prefix_size = 3

        def _mapper(
            class_values,
            class_indices,
            class_shape,
            attribute_values,
            attributes_indices,
            attributes_shape,
        ):
            class_ids = []
            class_ids_indices = []

            attribute_ids = []
            attribute_ids_indices = []

            for (
                sub_index,
                class_names,
                _,
                attribute_names,
                _,
            ) in sparse_generators.matching_indices_generator(
                index_prefix_size,
                class_values,
                class_indices,
                attribute_values,
                attributes_indices,
            ):
                matched_class_id = None
                matched_attributes = None
                for matcher in self._class_matchers:
                    matched_class_id, matched_attributes = matcher.match(
                        class_names, attribute_names
                    )
                    if matched_class_id is not None:
                        break

                class_ids.append(
                    matched_class_id
                    if matched_class_id is not None
                    else self._default_class_id
                )

                class_ids_indices.append(sub_index + [0])

                attribute_index_counter = 0
                # Sets don't iterate deterministically in Python 3, convert to a sorted list so this
                # function gives back consistent return values (useful for testing).
                matched_attributes = sorted(list(matched_attributes))
                for attribute in matched_attributes:
                    attribute_ids.append(
                        self._attribute_mappings.get(
                            attribute, self._default_attribute_id
                        )
                    )
                    attribute_ids_indices.append(sub_index + [attribute_index_counter])
                    attribute_index_counter += 1

            class_ids = np.array(class_ids, dtype=np.int32)
            class_ids_indices = np.array(class_ids_indices, dtype=np.int64)

            attribute_ids = np.array(attribute_ids, dtype=np.int32)
            attribute_ids_indices = np.array(attribute_ids_indices, dtype=np.int64)

            return (
                class_ids,
                class_ids_indices,
                class_shape,
                attribute_ids,
                attribute_ids_indices,
                attributes_shape,
            )

        (
            mapped_class_ids,
            mapped_class_indices,
            mapped_class_shape,
            mapped_attribute_ids,
            mapped_attribute_indices,
            mapped_attribute_shape,
        ) = tf.compat.v1.py_func(
            _mapper,
            [
                classes.values,
                classes.indices,
                classes.dense_shape,
                attributes.values,
                attributes.indices,
                attributes.dense_shape,
            ],
            [tf.int32, tf.int64, tf.int64, tf.int32, tf.int64, tf.int64],
            stateful=False,
        )

        return Polygon2DLabel(
            vertices=polygon_2d_label.vertices,
            classes=tf.SparseTensor(
                values=mapped_class_ids,
                indices=mapped_class_indices,
                dense_shape=mapped_class_shape,
            ),
            attributes=tf.SparseTensor(
                values=mapped_attribute_ids,
                indices=mapped_attribute_indices,
                dense_shape=mapped_attribute_shape,
            ),
        )


def _normalized_set(strings):
    if strings is None:
        return set()

    return {_decode(s.strip().lower()) for s in strings}


def _decode(obj):
    """Decodes byte strings into unicode strings in Python 3."""
    if isinstance(obj, str):
        return obj
    return obj.decode()


class _Matcher(object):
    """Matcher class which matches against a single class match specification."""

    def __init__(
        self,
        class_id,
        class_name,
        match_any_class,
        match_any_attribute=None,
        match_all_attributes=None,
        match_all_attributes_allow_others=False,
        remove_matched_attributes=False,
    ):
        self._any_class = _normalized_set(match_any_class)
        self._any_attributes = _normalized_set(match_any_attribute)
        self._all_attributes = _normalized_set(match_all_attributes)
        self._all_attributes_allow_others = match_all_attributes_allow_others

        self._class_id = class_id
        self._class_name = class_name
        self._remove_matched_attributes = remove_matched_attributes

    def match(self, class_names, attribute_names):
        class_names = _normalized_set(class_names)
        attribute_names = _normalized_set(attribute_names)
        # Possible match in case class_names intersects with any class or if both are empty,
        # otherwise return no match here
        if not (class_names & self._any_class) and (class_names or self._any_class):
            return None, attribute_names

        if self._any_attributes and not (attribute_names & self._any_attributes):
            return None, attribute_names

        if self._all_attributes:
            if self._all_attributes_allow_others:
                if len(attribute_names & self._all_attributes) != len(
                    self._all_attributes
                ):
                    return None, attribute_names
            else:
                if not self._all_attributes == attribute_names:
                    return None, attribute_names

        if self._remove_matched_attributes:
            attribute_names = attribute_names.difference(self._any_attributes)
            attribute_names = attribute_names.difference(self._all_attributes)

        return self._class_id, attribute_names
