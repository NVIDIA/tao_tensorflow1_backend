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

"""Main test for synthetic_data_source.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader.sources.synthetic_data_source import (
    SyntheticDataSource,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import FEATURE_CAMERA
from nvidia_tao_tf1.blocks.multi_source_loader.types import test_fixtures
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


def test_length():
    source = SyntheticDataSource(5, template=test_fixtures.make_example_3d(504, 960))
    assert len(source) == 5


def test_yields_specified_shapes():
    source = SyntheticDataSource(5, template=test_fixtures.make_example_3d(504, 960))
    dataset = source()
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    example = iterator.get_next()
    with tf.compat.v1.Session() as sess:
        count = 0
        with pytest.raises(tf.errors.OutOfRangeError):
            while True:
                image = sess.run(example.instances[FEATURE_CAMERA].images)
                count += 1
                assert image.shape == (3, 504, 960)
    assert count == 5


def test_with_tracker_dict():
    tracker_dict = {}
    source = SyntheticDataSource(
        5, template=test_fixtures.make_example_3d(504, 960), tracker_dict=tracker_dict
    )
    dataset = source()
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    getnext = iterator.get_next()
    with tf.compat.v1.Session() as sess:
        for _ in range(4):
            sess.run(getnext)
    assert "call_count" in tracker_dict
    assert tracker_dict["call_count"] == 4


def test_sampling_ratio_defaults_to_one():
    source = SyntheticDataSource(
        example_count=1, template=test_fixtures.make_example_3d(504, 960)
    )
    assert source.sample_ratio == 1.0


def test_sampling_ratio_is_set():
    sample_ratio = 0.2
    source = SyntheticDataSource(
        example_count=1,
        template=test_fixtures.make_example_3d(504, 960),
        sample_ratio=sample_ratio,
    )
    assert source.sample_ratio == sample_ratio


def test_serialization_and_deserialization():
    """Test TAOObject serialization and deserialization on SyntheticDataSource."""
    source = SyntheticDataSource(
        example_count=1,
        template=test_fixtures.make_example_3d(504, 960),
        sample_ratio=0.2,
    )
    source_dict = source.serialize()
    deserialized_source = deserialize_tao_object(source_dict)
    assert source._example_count == deserialized_source._example_count
    assert source.sample_ratio == deserialized_source.sample_ratio
