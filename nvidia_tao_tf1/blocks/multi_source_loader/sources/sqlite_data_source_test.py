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

import os
import shutil
import tempfile

import mock
import numpy as np
from parameterized import parameterized
import pytest
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.sources.sqlite_data_source import (
    SqliteDataSource,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Bbox2DLabel,
    FEATURE_CAMERA,
    FEATURE_SESSION,
    Images2DReference,
    LABEL_MAP,
    LABEL_OBJECT,
    Polygon2DLabel,
)
from modulus.dataloader.testdata.db_test_builder import DbTestBuilder
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


def _expected_frames(
    schema_version, skip_empty=False, sequence_length=None, include_unlabeled=False
):
    sequence_length = sequence_length
    if schema_version < 8:
        return 24
    # V8 example dataset has extra frames to exercise empty and unlabeled frame functionality.
    if skip_empty:
        num_frames = 24
    elif include_unlabeled:
        num_frames = 27
    else:
        num_frames = 26

    if sequence_length:
        return (num_frames // sequence_length) * sequence_length

    return num_frames


class SqliteDataSourceTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._db_builder = DbTestBuilder()

    @classmethod
    def tearDownClass(cls):
        cls._db_builder.cleanup()

    def _sqlite_path(self, schema=None):
        return self.__class__._db_builder.get_db("test_dataset", schema)

    def _split_path(self, schema=None):
        return self.__class__._db_builder.get_db("test_dataset_split", schema)

    def _mixed_res_path(self):
        return self.__class__._db_builder.get_db("test_dataset_mixed_res", 8)

    def setUp(self):
        self.export_path = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.export_path)

    @parameterized.expand([[5], [8]])
    def test_raises_on_when_image_dir_does_not_exist(self, schema):
        with pytest.raises(ValueError) as exception:
            SqliteDataSource(
                sqlite_path=self._sqlite_path(schema),
                image_dir="/no/such/image_dir",
                export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
            )

            assert "/no/such/image_dir" in str(exception)

    def test_raises_on_when_database_does_not_exist(self):
        with pytest.raises(ValueError) as exception:
            SqliteDataSource(
                sqlite_path="/no/such/dataset.sqlite",
                image_dir=self.export_path,
                export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
            )

        assert "/no/such/dataset.sqlite" in str(exception)

    @parameterized.expand(
        [
            (
                SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
                None,
                142200,
                b"0ce4c786-92eb-5a2d-8e95-fffbc30ebd1a",
                (960,),
                (604,),
                (3,),
                (2,),
                19,
                5,
                False,
            ),
            (
                SqliteDataSource.FORMAT_JPEG,
                4,
                [142200, 33960, 101460, 222960],
                [
                    b"0ce4c786-92eb-5a2d-8e95-fffbc30ebd1a",
                    b"2b3899b9-5d2c-50f4-8dab-0bf642bc4158",
                    b"2b3899b9-5d2c-50f4-8dab-0bf642bc4158",
                    b"418acd69-a14d-5486-a06e-ec2a349e06cd",
                ],
                (4, 960),
                (4, 604),
                (4,),
                (3,),
                33,
                5,
                False,
            ),
            (
                SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
                None,
                142200,
                b"0ce4c786-92eb-5a2d-8e95-fffbc30ebd1a",
                (960,),
                (604,),
                (3,),
                (2,),
                19,
                8,
                False,
            ),
            (
                SqliteDataSource.FORMAT_JPEG,
                4,
                [142200, 33960, 101460, 222960],
                [
                    b"0ce4c786-92eb-5a2d-8e95-fffbc30ebd1a",
                    b"2b3899b9-5d2c-50f4-8dab-0bf642bc4158",
                    b"2b3899b9-5d2c-50f4-8dab-0bf642bc4158",
                    b"418acd69-a14d-5486-a06e-ec2a349e06cd",
                ],
                (4, 960),
                (4, 604),
                (4,),
                (3,),
                33,
                8,
                False,
            ),
            (
                SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
                None,
                142200,
                b"0ce4c786-92eb-5a2d-8e95-fffbc30ebd1a",
                (960,),
                (604,),
                (3,),
                (2,),
                19,
                8,
                True,
            ),
            (
                SqliteDataSource.FORMAT_JPEG,
                4,
                [142200, 33960, 101460, 222960],
                [
                    b"0ce4c786-92eb-5a2d-8e95-fffbc30ebd1a",
                    b"2b3899b9-5d2c-50f4-8dab-0bf642bc4158",
                    b"2b3899b9-5d2c-50f4-8dab-0bf642bc4158",
                    b"418acd69-a14d-5486-a06e-ec2a349e06cd",
                ],
                (4, 960),
                (4, 604),
                (4,),
                (3,),
                33,
                8,
                True,
            ),
        ]
    )
    def test_end_to_end(
        self,
        export_format,
        sequence_length,
        expected_frame_number,
        expected_sequence_id,
        expected_canvas_width,
        expected_canvas_height,
        expected_map_vertices_dims,
        expected_map_dims,
        expected_num_objects,
        schema,
        include_unlabeled,
    ):
        def _make_frame_id(sequence_id, frame_number):
            return os.path.join(
                sequence_id.decode(), "video_B0_FC_60", str(frame_number)
            )

        source = SqliteDataSource(
            sqlite_path=self._sqlite_path(schema),
            image_dir=self.export_path,
            export_format=export_format,
            supported_labels=["POLYGON", "BOX"],
            include_unlabeled_frames=include_unlabeled,
        )
        source.set_sequence_length(sequence_length)

        if sequence_length is None:
            expected_frame_id = [
                _make_frame_id(expected_sequence_id, expected_frame_number)
            ]
        else:
            expected_frame_id = [
                [
                    _make_frame_id(expected_sequence_id[i], expected_frame_number[i])
                    for i in range(sequence_length)
                ]
            ]

        dataset = source()
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        get_next = iterator.get_next()
        with self.test_session() as sess:
            sess.run([tf.compat.v1.tables_initializer(), iterator.initializer])
            example = sess.run(get_next)
            image_references = example.instances[FEATURE_CAMERA]
            session = example.instances[FEATURE_SESSION]

            # Check image related portion.
            assert isinstance(image_references, Images2DReference)
            assert expected_canvas_height == image_references.canvas_shape.height.shape
            assert expected_canvas_width == image_references.canvas_shape.width.shape

            np.testing.assert_equal(session.uuid, expected_sequence_id)
            np.testing.assert_equal(session.frame_number, expected_frame_number)

            # Check LABEL_MAP.
            label_map = example.labels[LABEL_MAP]
            assert isinstance(label_map, Polygon2DLabel)
            assert (
                expected_map_vertices_dims
                == label_map.vertices.coordinates.dense_shape.shape
            )
            assert expected_map_dims == label_map.classes.dense_shape.shape
            assert expected_map_dims == label_map.attributes.dense_shape.shape
            assert (604,) == label_map.vertices.canvas_shape.height
            assert (960,) == label_map.vertices.canvas_shape.width

            # Check LABEL_OBJECT.
            label_object = example.labels[LABEL_OBJECT]
            assert isinstance(label_object, Bbox2DLabel)
            actual_frame_id = np.vectorize(lambda s: s.decode())(label_object.frame_id)
            np.testing.assert_equal(actual_frame_id, expected_frame_id)
            # Check that LTRB order is preserved, and no bounding boxes whose dimensions are less
            # than 1.0 pixels are included.
            ltrb_coords = label_object.vertices.coordinates.values.reshape([-1, 4])
            self.assertAllGreaterEqual(ltrb_coords[:, 2] - ltrb_coords[:, 0], 1.0)
            self.assertAllGreaterEqual(ltrb_coords[:, 3] - ltrb_coords[:, 1], 1.0)
            # Check that the rest of the features have sane shapes.
            for feature_name in Bbox2DLabel.TARGET_FEATURES:
                feature_tensor = getattr(label_object, feature_name)
                if feature_name == "vertices":
                    assert (
                        len(feature_tensor.coordinates.values) // 4
                        == expected_num_objects
                    )
                elif feature_name in {"world_bbox_z", "truncation"}:
                    # These are features not directly found in HumanLoop exports, but that are
                    # legacy that we carry around for BW compatibility's sake with DriveNet
                    # TFRecords.
                    assert isinstance(feature_tensor, np.ndarray)
                    assert feature_tensor.size == 0  # Un-populated field.
                else:
                    assert len(feature_tensor.values) == expected_num_objects

    @parameterized.expand([[1, 5], [4, 5], [5, 5], [1, 8], [4, 8], [5, 8]])
    def test_length_and_image_size(self, sequence_length, schema):
        source = SqliteDataSource(
            sqlite_path=self._sqlite_path(schema),
            image_dir=self.export_path,
            export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
            supported_labels=["POLYLINE", "BOX"],
        )
        source.set_sequence_length(sequence_length)
        source.initialize()
        assert _expected_frames(
            schema, sequence_length=sequence_length
        ) // sequence_length == len(source)
        assert 960 == source.max_image_width
        assert 604 == source.max_image_height

    @parameterized.expand(
        [
            (4, 1, False, 5),
            (4, 1, True, 5),
            (3, 2, False, 5),
            (3, 0, True, 5),
            (4, 1, False, 8),
            (4, 1, True, 8),
            (3, 2, False, 8),
            (3, 0, True, 8),
        ]
    )
    def test_pseudo_shard_setting(self, num_shards, shard_id, pseudo_sharding, schema):
        source = SqliteDataSource(
            sqlite_path=self._sqlite_path(schema),
            image_dir=self.export_path,
            export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
            supported_labels=["POLYLINE", "BOX"],
        )
        source.set_shard(num_shards, shard_id, pseudo_sharding=pseudo_sharding)
        source.initialize()
        assert _expected_frames(schema) == len(source)
        assert source._dataset._num_shards == num_shards
        assert source._dataset._shard_id == shard_id
        assert source._dataset._pseudo_sharding == pseudo_sharding

    @parameterized.expand(
        [
            (["train"], 19, 5),
            ([b"val0"], 2, 5),
            (["val1"], 3, 5),
            (["val0", "val1"], 5, 5),
            (["train", "val0", "val1"], 24, 5),
            (["train"], 21, 8),
            ([b"val0"], 2, 8),
            (["val1"], 3, 8),
            (["val0", "val1"], 5, 8),
            (["train", "val0", "val1"], 26, 8),
        ]
    )
    def test_split(self, split_tags, expected_length, schema):
        source = SqliteDataSource(
            sqlite_path=self._sqlite_path(schema),
            image_dir=self.export_path,
            export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
            split_db_filename=self._split_path(),
            split_tags=split_tags,
            supported_labels=["POLYLINE", "BOX"],
        )
        source.initialize()
        assert expected_length == len(source)
        assert 960 == source.max_image_width
        assert 604 == source.max_image_height

    @parameterized.expand([[5], [8]])
    def test_zero_sampling_ratio_defaults_to_one(self, schema):
        source = SqliteDataSource(
            sqlite_path=self._sqlite_path(schema),
            image_dir=self.export_path,
            export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
        )
        assert source.sample_ratio == 1.0

    @parameterized.expand([[5], [8]])
    def test_sampling_ratio_is_set(self, schema):
        sample_ratio = 0.2
        source = SqliteDataSource(
            sqlite_path=self._sqlite_path(schema),
            image_dir=self.export_path,
            export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
            sample_ratio=sample_ratio,
        )
        assert source.sample_ratio == sample_ratio

    def test_query_split_tags(self):
        tags = SqliteDataSource.query_split_tags(self._split_path())
        assert tags == ["train", "val0", "val1"]

    @parameterized.expand(
        [[None, 5], [True, 5], [False, 5], [None, 8], [True, 8], [False, 8]]
    )
    def test_skip_empty_frames_config(self, skip_empty_frames, schema):
        source = SqliteDataSource(
            sqlite_path=self._sqlite_path(schema),
            image_dir=self.export_path,
            export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
            sample_ratio=1.0,
            skip_empty_frames=skip_empty_frames,
        )
        source.initialize()
        hldataset = source._dataset
        if skip_empty_frames:
            self.assertTrue(hldataset.skip_empty_frames)
        else:
            self.assertFalse(hldataset.skip_empty_frames)

    @parameterized.expand([[5], [8]])
    def test_no_subset_size(self, schema):
        source = SqliteDataSource(
            sqlite_path=self._sqlite_path(schema),
            image_dir=self.export_path,
            export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
            supported_labels=["POLYGON", "BOX"],
            subset_size=None,
        )
        source.initialize()
        assert source.num_samples == _expected_frames(schema)

    @parameterized.expand([[5], [8]])
    def test_serialization_and_deserialization(self, schema):
        """Test TAOObject serialization and deserialization on SqliteDataSource."""
        source = SqliteDataSource(
            sqlite_path=self._sqlite_path(schema),
            image_dir=self.export_path,
            export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
            sample_ratio=0.1,
        )
        source.initialize()
        source_dict = source.serialize()
        deserialized_source = deserialize_tao_object(source_dict)

        self.assertEqual(source.sqlite_path, deserialized_source.sqlite_path)
        self.assertEqual(source.image_dir, deserialized_source.image_dir)
        self.assertEqual(source.export_format, deserialized_source.export_format)
        self.assertEqual(source.sample_ratio, deserialized_source.sample_ratio)
        self.assertEqual(source.sample_ratio, deserialized_source.sample_ratio)

    @parameterized.expand([[5], [8]])
    def test_subset_size_positive_and_restricts(self, schema):
        source = SqliteDataSource(
            sqlite_path=self._sqlite_path(schema),
            image_dir=self.export_path,
            export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
            supported_labels=["POLYGON", "BOX"],
            subset_size=2,
        )
        source.initialize()
        assert source.num_samples == 2

    @parameterized.expand([[5], [8]])
    def test_subset_size_positive_and_large(self, schema):
        source = SqliteDataSource(
            sqlite_path=self._sqlite_path(schema),
            image_dir=self.export_path,
            export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
            supported_labels=["POLYGON", "BOX"],
            subset_size=1024768,  # > dataset's size (24)
        )
        source.initialize()
        assert source.num_samples == _expected_frames(schema)

    @parameterized.expand([[5], [8]])
    def test_subset_size_zero(self, schema):
        source = SqliteDataSource(
            sqlite_path=self._sqlite_path(schema),
            image_dir=self.export_path,
            export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
            supported_labels=["POLYGON", "BOX"],
            subset_size=0,
        )
        source.initialize()
        assert source.num_samples == _expected_frames(schema)

    @parameterized.expand([[5], [8]])
    def test_subset_size_negative(self, schema):
        with pytest.raises(ValueError):
            SqliteDataSource(
                sqlite_path=self._sqlite_path(schema),
                image_dir=self.export_path,
                export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
                supported_labels=["POLYGON", "BOX"],
                subset_size=-13,
            )

    @parameterized.expand([[5], [8]])
    def test_oversample_is_called(self, schema):
        oversampling_strategy = mock.Mock()
        oversampling_strategy.oversample.side_effect = (
            lambda frame_groups, count_lookup: frame_groups
        )
        sqlite_data_source = SqliteDataSource(
            sqlite_path=self._sqlite_path(schema),
            image_dir=self.export_path,
            export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
            supported_labels=["POLYGON", "BOX"],
            oversampling_strategy=oversampling_strategy,
        )

        sqlite_data_source()

        oversampling_strategy.oversample.assert_called()

    @parameterized.expand([[5], [8]])
    def test_raise_on_missing_split_tag(self, schema):
        with pytest.raises(ValueError) as value_error:
            source = SqliteDataSource(
                sqlite_path=self._sqlite_path(schema),
                image_dir=self.export_path,
                export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
                split_db_filename=self._split_path(),
                split_tags=["non-existent_tag"],
                supported_labels=["POLYLINE", "BOX"],
            )
            source.initialize()
        self.assertIn("'non-existent_tag'] not in", str(value_error))

    def test_raise_on_additional_conditions_referencing_multiple_tables(self):
        for bad_condition in [["sequences features"], ["no valid tables"]]:
            with pytest.raises(ValueError) as value_error:
                SqliteDataSource(
                    sqlite_path=self._sqlite_path(8),
                    image_dir=self.export_path,
                    export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
                    additional_conditions=bad_condition,
                )
            self.assertIn("Each condition can only ", str(value_error))

        # Valid test case
        SqliteDataSource(
            sqlite_path=self._sqlite_path(8),
            image_dir=self.export_path,
            export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
            additional_conditions=["sequences.id=1", "features.id=1", "features.id=2"],
        )

    def test_additional_sequence_conditions(self):
        # Will default to biggest resolution. Whether or not we should raise an error is an open
        # question.
        source = SqliteDataSource(
            sqlite_path=self._mixed_res_path(),
            image_dir=self.export_path,
            export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
        )
        assert source.max_image_width == 1924
        assert source.max_image_height == 1084

        # Will filter and only get the sequences that are the smaller dimension.
        source = SqliteDataSource(
            sqlite_path=self._mixed_res_path(),
            image_dir=self.export_path,
            export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
            additional_conditions=["sequences.width = 1920"],
        )
        assert source.max_image_width == 960
        assert source.max_image_height == 604
