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
"""Class for mapping objects to output unique instance ids."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors import sparse_generators
from nvidia_tao_tf1.blocks.multi_source_loader.types import Polygon2DLabel
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors import Processor


class InstanceMapper(Processor):
    """InstanceMapper maps instance labels to instance ids."""

    @save_args
    def __init__(self, default_has_instance, default_instance_id, exceptions, **kwargs):
        """Construct an InstanceMapper.

        Args
            default_has_instance (bool): The default hasInstance flag. To be applied if class name
            contains no substring in exceptions.

            default_instance_id (int): The default instance id. To be applied if class does not
            have instances by definition.

            exceptions(set of strings): Class will be excluded from default instance id
            assignment if class name contains any substring in this set.

        """
        self._default_has_instance = default_has_instance
        self._default_instance_id = default_instance_id
        self._exception_set = exceptions
        super(InstanceMapper, self).__init__(**kwargs)

    def call(self, polygon_2d_label):
        """Map text class and attribute names to numeric ids.

        Args:
            polygon_2d_label (Polygon2DLabel): A label containing 2D polygons and their
                associated classes and attributes. If a 2D polygon captures a complete instance,
                its attribute is empty. If an instance consists of multiple polygons, these polygons
                will have a common attribute. Attributes are unique (unless empty) within class.

        Returns:
            (Polygon2DLabel): The label with the classes and attributes mapped to unique numeric
            id for each instance.
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

        def _mapper(class_values, class_indices, attribute_values, attributes_indices):

            # Initiate a dictionary to track occluded instances and assign id if instance exists.
            mapped_objects = {}

            # Initiate id counter.
            instance_id_counter = self._default_instance_id + 1
            instance_ids = []

            current_batch_index = current_frame_index = 0
            for (
                _,
                class_name,
                class_name_index,
                attribute_names,
                _,
            ) in sparse_generators.matching_indices_generator(
                index_prefix_size,
                class_values,
                class_indices,
                attribute_values,
                attributes_indices,
            ):
                # Clear id dictionary and reset instance id counter for every new frame.
                if (
                    current_batch_index != class_name_index[0][0]
                    or current_frame_index != class_name_index[0][1]
                ):
                    mapped_objects.clear()
                    current_batch_index = class_name_index[0][0]
                    current_frame_index = class_name_index[0][1]
                    instance_id_counter = self._default_instance_id + 1

                if len(class_name) != 1:
                    print(class_name, end=" ")
                    print(
                        "Polygon is tagged with more than one class. Proceed with the first one."
                    )

                class_name = class_name[0].strip().lower().decode()
                _hasInstance = self._default_has_instance
                if any(elem in class_name for elem in self._exception_set):
                    _hasInstance = not _hasInstance

                # Only update polygons that hasInstance is True by definition,
                # otherwise default id will be assigned.
                if not _hasInstance:
                    instance_ids.append(self._default_instance_id)
                    continue

                if attribute_names == []:
                    instance_ids.append(instance_id_counter)
                    instance_id_counter += 1
                else:
                    # Only polygons on same frame, with same class name and attribute
                    # will be given identical instance id.
                    _object_key = "{}_{}".format(
                        class_name, attribute_names[0].strip().lower()
                    )
                    if _object_key not in mapped_objects:
                        mapped_objects[_object_key] = instance_id_counter
                        instance_id_counter += 1

                    instance_ids.append(mapped_objects[_object_key])

            return np.array(instance_ids, dtype=np.int32)

        mapped_ids = tf.compat.v1.py_func(
            _mapper,
            [classes.values, classes.indices, attributes.values, attributes.indices],
            tf.int32,
            stateful=False,
        )

        return Polygon2DLabel(
            vertices=polygon_2d_label.vertices,
            classes=tf.SparseTensor(
                values=mapped_ids,
                indices=classes.indices,
                dense_shape=classes.dense_shape,
            ),
            attributes=tf.SparseTensor(
                values=attributes.values,
                indices=attributes.indices,
                dense_shape=attributes.dense_shape,
            ),
        )
