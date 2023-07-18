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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader.sources.tfrecords_data_source import (
    TFRecordsDataSource,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    FEATURE_CAMERA,
    Images2DReference,
    LABEL_MAP,
    Polygon2DLabel,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


class TFRecordsDataSourceTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image_dir = (
            "moduluspy/modulus/blocks/data_loaders/"
            "multi_source_loader/sources/testdata/"
        )
        cls.tfrecords_path = (
            "moduluspy/modulus/blocks/data_loaders/"
            "multi_source_loader/sources/testdata/test_1.tfrecords"
        )
        cls.tfrecords_path2 = (
            "moduluspy/modulus/blocks/data_loaders/"
            "multi_source_loader/sources/testdata/test_2.tfrecords"
        )

    def test_raises_on_when_image_dir_does_not_exist(self):
        with pytest.raises(ValueError) as exception:
            TFRecordsDataSource(
                tfrecord_path=self.tfrecords_path,
                image_dir="/no/such/image_dir",
                extension=".jpg",
                height=1208,
                width=1920,
                channels=3,
                subset_size=0,
            )

        assert "/no/such/image_dir" in str(exception)

    def test_raises_on_when_tfrecords_file_does_not_exist(self):
        with pytest.raises(ValueError) as exception:
            TFRecordsDataSource(
                tfrecord_path="/no/such/tfrecords/file",
                image_dir=self.image_dir,
                extension=".jpg",
                height=1208,
                width=1920,
                channels=3,
                subset_size=0,
            )

        assert "/no/such/tfrecords/file" in str(exception)

    def test_length(self):
        source = TFRecordsDataSource(
            tfrecord_path=self.tfrecords_path,
            image_dir=self.image_dir,
            extension=".jpg",
            height=604,
            width=960,
            channels=3,
            subset_size=0,
        )

        assert 3 == len(source)

    def test_length_with_list_of_tfrecords(self):
        source = TFRecordsDataSource(
            tfrecord_path=[self.tfrecords_path, self.tfrecords_path2],
            image_dir=self.image_dir,
            extension=".jpg",
            height=604,
            width=960,
            channels=3,
            subset_size=0,
        )

        assert 6 == len(source)

    def test_end_to_end(self):
        source = TFRecordsDataSource(
            tfrecord_path=self.tfrecords_path,
            image_dir=self.image_dir,
            extension=".jpg",
            height=1208,
            width=1920,
            channels=3,
            subset_size=0,
        )

        dataset = source()
        dataset = dataset.apply(source.parse_example)

        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        get_next = iterator.get_next()
        with self.test_session() as sess:
            read_examples = 0
            while True:
                try:
                    example = sess.run(get_next)
                    read_examples += 1
                    image_references = example.instances[FEATURE_CAMERA]
                    assert Images2DReference == type(image_references)
                    assert (1208,) == image_references.canvas_shape.height.shape
                    assert (1920,) == image_references.canvas_shape.width.shape

                    label = example.labels[LABEL_MAP]
                    assert isinstance(label, Polygon2DLabel)
                    assert (3,) == label.vertices.coordinates.dense_shape.shape
                except tf.errors.OutOfRangeError:
                    break

            assert len(source) == read_examples

    def test_zero_sampling_ratio_defaults_to_one(self):
        source = TFRecordsDataSource(
            tfrecord_path=self.tfrecords_path,
            image_dir=self.image_dir,
            extension=".jpg",
            height=1208,
            width=1920,
            channels=3,
            subset_size=0,
        )
        assert source.sample_ratio == 1.0

    def test_sampling_ratio_is_set(self):
        sample_ratio = 0.2
        source = TFRecordsDataSource(
            tfrecord_path=self.tfrecords_path,
            image_dir=self.image_dir,
            extension=".jpg",
            height=1208,
            width=1920,
            channels=3,
            subset_size=0,
            sample_ratio=sample_ratio,
        )
        assert source.sample_ratio == sample_ratio

    def test_serialization_and_deserialization(self):
        sample_ratio = 0.2
        source = TFRecordsDataSource(
            tfrecord_path=self.tfrecords_path,
            image_dir=self.image_dir,
            extension=".jpg",
            height=1208,
            width=1920,
            channels=3,
            subset_size=0,
            sample_ratio=sample_ratio,
        )
        source_dict = source.serialize()
        deserialized_source = deserialize_tao_object(source_dict)
        assert source.tfrecord_path == deserialized_source.tfrecord_path
        assert source.image_dir == deserialized_source.image_dir
        assert source.extension == deserialized_source.extension
        assert source.height == deserialized_source.height
        assert source.width == deserialized_source.width
        assert source.channels == deserialized_source.channels
        assert source.subset_size == deserialized_source.subset_size
        assert source.sample_ratio == deserialized_source.sample_ratio
