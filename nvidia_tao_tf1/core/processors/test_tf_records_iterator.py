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
"""Test the tf records iterator in processors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pytest
import tensorflow as tf

import nvidia_tao_tf1.core
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


@pytest.fixture
def dummy_data():
    dataset_size = 10
    dummy_data = list(range(dataset_size))
    return dummy_data


@pytest.fixture
def get_tf_records_iterator_tensors(dummy_data, tmpdir):
    def _get_tf_records_iterator_tensors(
        shuffle_buffer_size, batch_size, shuffle, repeat, sequence_length, batch_as_list
    ):
        # Set up dummy values
        dummy_file_name = "dummy.tfrecords"
        dummy_file_path = os.path.join(str(tmpdir), dummy_file_name)

        # Write TFRecords file
        writer = tf.io.TFRecordWriter(dummy_file_path)
        for data in dummy_data:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "value": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[data])
                        )
                    }
                )
            )
            writer.write(example.SerializeToString())
        writer.close()

        # Open TFRecords file
        iterator = nvidia_tao_tf1.core.processors.TFRecordsIterator(
            dummy_file_path,
            shuffle_buffer_size=shuffle_buffer_size,
            batch_size=batch_size,
            shuffle=shuffle,
            repeat=repeat,
            sequence_length=sequence_length,
            batch_as_list=batch_as_list,
        )

        # Parse example
        example = iterator()
        features = {"value": tf.io.FixedLenFeature([], dtype=tf.int64)}
        if batch_as_list:
            value = [
                nvidia_tao_tf1.core.processors.ParseExampleProto(features=features, single=True)(
                    record
                )["value"]
                for record in example
            ]
            return value, iterator
        value = nvidia_tao_tf1.core.processors.ParseExampleProto(features=features, single=False)(
            example
        )
        return value["value"], iterator

    return _get_tf_records_iterator_tensors


@pytest.mark.parametrize("shuffle_buffer_size", [1, 10, 100])
def test_tf_records_iterator_shuffle_buffer_size(
    shuffle_buffer_size, get_tf_records_iterator_tensors, dummy_data
):
    """Test that different buffer sizes work as expected with TFRecordsIterator."""
    # Get the value tensors from TFRecords
    value, _ = get_tf_records_iterator_tensors(
        shuffle_buffer_size,
        batch_size=10,
        shuffle=True,
        repeat=True,
        sequence_length=0,
        batch_as_list=False,
    )
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.get_collection("iterator_init"))
    # Get data for 10 epochs
    per_epoch_values = []
    for _ in range(10):
        per_epoch_values.append(list(sess.run(value)))

    # In case of shuffle_buffer_size being 1, we test that every epoch has the same data order
    # which is the same as the order of original data.
    if shuffle_buffer_size == 1:
        assert all(
            epoch_values == dummy_data for epoch_values in per_epoch_values
        ), "Parsed examples do not match original data."
    # Else we test that in 10 epochs we get all the values in the original dataset
    else:
        all_values = [val for epoch_values in per_epoch_values for val in epoch_values]
        assert set(all_values) == set(
            dummy_data
        ), "Parsed examples do not match original data."


@pytest.mark.parametrize("batch_size", [1, 5, 100])
def test_tf_records_iterator_batch_size(batch_size, get_tf_records_iterator_tensors):
    """Test that different batch sizes work as expected with TFRecordsIterator."""
    # Get the value tensors from TFRecords
    value, _ = get_tf_records_iterator_tensors(
        shuffle_buffer_size=1,
        batch_size=batch_size,
        shuffle=True,
        repeat=True,
        sequence_length=0,
        batch_as_list=False,
    )
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.get_collection("iterator_init"))
    # Get 100 batches of data
    for _ in range(100):
        values = sess.run(value)
        assert values.shape[0] == batch_size, "Parsed examples aren't batch size long."


@pytest.mark.parametrize("shuffle", [True, False])
def test_tf_records_iterator_shuffliness(
    shuffle, get_tf_records_iterator_tensors, dummy_data
):
    """Test that shuffle option works as expected with TFRecordsIterator."""
    # Get the value tensors from TFRecords
    shuffle_buffer_size = 10 if shuffle else 0
    value, _ = get_tf_records_iterator_tensors(
        shuffle_buffer_size=shuffle_buffer_size,
        batch_size=10,
        shuffle=shuffle,
        repeat=True,
        sequence_length=0,
        batch_as_list=False,
    )
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.get_collection("iterator_init"))
    # Get data for 10 epochs
    per_epoch_values = []
    for _ in range(10):
        per_epoch_values.append(list(sess.run(value)))

    # Check that all samples are unique in each epoch.
    assert all(
        len(e) == len(set(e)) for e in per_epoch_values
    ), "Epochs contain duplicate samples."

    # In case of shuffle being True, we check that there's at least two epochs
    # with different order of data
    if shuffle:
        assert any(
            e1 != e2 for e1 in per_epoch_values for e2 in per_epoch_values
        ), "All the epochs use the same data in the same order even though shuffle is True."
    # Else we test that all the epochs have exactly the same data, that is the original data
    else:
        assert all(
            epoch_values == dummy_data for epoch_values in per_epoch_values
        ), "Some epoch use different data even though shuffle option is False."


@pytest.mark.parametrize("repeat", [True, False])
def test_tf_records_iterator_repeat(
    repeat, get_tf_records_iterator_tensors, dummy_data
):
    """Test that repeat option works as expected with TFRecordsIterator."""
    # Get the value tensors from TFRecords
    value, _ = get_tf_records_iterator_tensors(
        shuffle_buffer_size=0,
        batch_size=10,
        shuffle=False,
        repeat=repeat,
        sequence_length=0,
        batch_as_list=False,
    )
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.get_collection("iterator_init"))
    # Get data for 10 epochs if repeat, else only a single epoch is possible
    no_epochs = 10 if repeat else 1
    all_values = []
    for _ in range(no_epochs):
        all_values.append(list(sess.run(value)))

    # Check that the same dummy values are repeated
    assert all(
        value == dummy_data for value in all_values
    ), "Not all of original data has been observed when pulling data from TFRecordsIterator."


def test_tf_records_iterator_reset(get_tf_records_iterator_tensors, dummy_data):
    """Test that reset option works as expected with TFRecordsIterator."""
    # Get the value tensors from TFRecords
    value, iterator = get_tf_records_iterator_tensors(
        shuffle_buffer_size=0,
        batch_size=10,
        shuffle=False,
        repeat=False,
        sequence_length=0,
        batch_as_list=False,
    )

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.get_collection("iterator_init"))
    # Get data 10 times, but reset the iterator after each iteration,
    # it should work even with repeat=False
    for _ in range(10):
        values = list(sess.run(value))
        # Check that the same dummy values are used after every reset
        assert (
            values == dummy_data
        ), "Not all of original data has been observed when pulling data\
             from TFRecordsIterator."

        iterator.reset(sess)


@pytest.mark.parametrize("sequence_length", [2, 5])
def test_tf_records_iterator_block_shuffle(
    sequence_length, get_tf_records_iterator_tensors, dummy_data
):
    """Test that block shuffling works as expected with TFRecordsIterator"""
    # Get the value tensors from TFRecords
    value, _ = get_tf_records_iterator_tensors(
        shuffle_buffer_size=10,
        batch_size=10,
        shuffle=True,
        repeat=True,
        sequence_length=sequence_length,
        batch_as_list=False,
    )
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.get_collection("iterator_init"))
    all_values = sess.run(value)
    # the number of elements in data would be (batch_size * sequence_length)
    # with having blocks of sequence_length consecutive elements.
    # eg: [0,1,4,5,2,3,8,9....] for sequence_length=2
    for i in range(0, len(all_values), sequence_length):
        assert all(
            all_values[j] + 1 == all_values[j + 1]
            for j in range(i, i + sequence_length - 1)
        ), "Some elements in a block are shuffled"


@pytest.mark.parametrize("batch_as_list", [True, False])
def test_tf_records_iterator_batch_as_list(
    batch_as_list, get_tf_records_iterator_tensors, dummy_data
):
    """Test that batch_as_list returns same output when True and False with TFRecordsIterator"""
    # Get the value tensors from TFRecords
    value, _ = get_tf_records_iterator_tensors(
        shuffle_buffer_size=0,
        batch_size=10,
        shuffle=False,
        repeat=True,
        sequence_length=0,
        batch_as_list=batch_as_list,
    )

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.get_collection("iterator_init"))
    assert (
        list(sess.run(value)) == dummy_data
    ), "batch_as_list = {0} doesn't perform as expected".format(batch_as_list)


def test_serialization_and_deserialization_spatial_transform():
    """Test that it is a MaglevObject that can be serialized and deserialized."""
    iterator = nvidia_tao_tf1.core.processors.TFRecordsIterator(
        "dummy.tfrecords",
        shuffle_buffer_size=8,
        batch_size=8,
        shuffle=True,
        repeat=False,
        sequence_length=0,
        batch_as_list=False,
    )
    iterator_dict = iterator.serialize()
    deserialized_iterator = deserialize_tao_object(iterator_dict)
    assert iterator.file_list == deserialized_iterator.file_list
    assert iterator.batch_size == deserialized_iterator.batch_size
    assert iterator.shuffle_buffer_size == deserialized_iterator.shuffle_buffer_size
    assert iterator.shuffle == deserialized_iterator.shuffle
    assert iterator.repeat == deserialized_iterator.repeat
    assert iterator.batch_as_list == deserialized_iterator.batch_as_list
    assert iterator.sequence_length == deserialized_iterator.sequence_length
