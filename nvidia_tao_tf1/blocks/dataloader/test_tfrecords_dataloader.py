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

"""Tests for TFRecordsDataLoader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import numpy as np
from PIL import Image
import pytest
import tensorflow as tf

from nvidia_tao_tf1.blocks.dataloader.tfrecords_dataloader import TFRecordsDataLoader

IMAGE_SHAPE = [42, 42]
TRAIN = "train"
VAL = "val"


def _process_record(record_example):
    feature = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.float32),
    }
    example = tf.io.parse_single_example(serialized=record_example, features=feature)
    image = tf.image.decode_jpeg(example["image"], channels=3)
    return image, example["label"]


def _get_image_and_label(dataloader, mode):
    tf_dataset = dataloader(mode)
    iterator = tf.compat.v1.data.make_one_shot_iterator(tf_dataset())
    image, label = iterator.get_next()
    return image, label


def _create_test_tfrecord(tmpdir):
    record_file = str(tmpdir.join("test.tfrecord"))
    writer = tf.io.TFRecordWriter(record_file)
    temp_img = Image.new("RGB", IMAGE_SHAPE)
    output = io.BytesIO()
    temp_img.save(output, format="JPEG")
    fake_image = output.getvalue()
    feature = {
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[fake_image])),
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=[42])),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    writer.write(example.SerializeToString())


@pytest.mark.parametrize("mode", [TFRecordsDataLoader.TRAIN, TFRecordsDataLoader.VAL])
def test_tfrecords_dataloader(mode, tmpdir):
    """Test TFRecord dataloader using generated tfrecords and images."""
    _create_test_tfrecord(tmpdir)
    if mode == TFRecordsDataLoader.TRAIN:
        dataloader = TFRecordsDataLoader(str(tmpdir), "", IMAGE_SHAPE, 1)
    else:
        dataloader = TFRecordsDataLoader("", str(tmpdir), IMAGE_SHAPE, 1)
    # Override process record method to handle features defined for the test case.
    dataloader._process_record = _process_record

    image, label = _get_image_and_label(dataloader, mode)
    with tf.compat.v1.Session() as sess:
        image_eval, label_eval = sess.run([tf.reduce_sum(input_tensor=image), label])
        np.testing.assert_equal(image_eval, 0.0)
        np.testing.assert_equal(label_eval, 42.0)
