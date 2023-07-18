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

"""Tests for DataLoader object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
from mock import patch
from parameterized import parameterized
import pytest
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader import processors
from nvidia_tao_tf1.blocks.multi_source_loader.data_loader import (
    _normalize_images,
    _pick_largest_image_dtype,
    DataLoader,
)
from nvidia_tao_tf1.blocks.multi_source_loader.sources.data_source import (
    DataSource,
)
from nvidia_tao_tf1.blocks.multi_source_loader.sources.synthetic_data_source import (
    SyntheticDataSource,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    FEATURE_CAMERA,
    Images2D,
    SequenceExample,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import test_fixtures

from nvidia_tao_tf1.core.coreobject import deserialize_tao_object, register_tao_function


class FP16DataSource(DataSource):
    """Mocked DataSource with image dtype tf.float16."""

    def call(self):
        pass

    @property
    def image_dtype(self):
        return tf.float16

    def __len__(self):
        return 1


class FP32DataSource(DataSource):
    """Mocked DataSource with image dtype tf.float32."""

    def call(self):
        pass

    @property
    def image_dtype(self):
        return tf.float32

    def __len__(self):
        return 1


class UINT8DataSource(DataSource):
    """Mocked DataSource with image dtype tf.uint8."""

    def call(self):
        pass

    @property
    def image_dtype(self):
        return tf.uint8

    def __len__(self):
        return 1


def return_args(args):
    return args


@register_tao_function
def _synthetic_data_source_template_fn():
    return test_fixtures.make_example_3d(height=17, width=19, label_name="LABEL_A")


@pytest.fixture(scope="function")
@patch(
    "nvidia_tao_tf1.blocks.multi_source_loader."
    "sources.tfrecords_data_source.TFRecordsDataSource"
)
def _data_sources(mocked_tfrecords_source):
    # we assume this data_source has 128 frames
    mocked_tfrecords_source.__len__.return_value = 128
    mocked_tfrecords_source.get_image_properties.return_value = 980, 604
    data_sources = []
    data_sources.append(mocked_tfrecords_source)
    return data_sources


def _datasource_mock(width=980, height=604):
    return mock.Mock(__len__=lambda _: 42, get_image_properties=lambda: (width, height))


def test_defaults_to_single_shard(_data_sources):
    dataloader = DataLoader(
        _data_sources, augmentation_pipeline=[], batch_size_per_gpu=32
    )
    dataloader.set_shard()
    assert dataloader.batch_size_per_gpu == 32
    assert dataloader.steps == 4


def test_two_shards_halves_batch_size(_data_sources):
    dataloader = DataLoader(
        data_sources=_data_sources,
        augmentation_pipeline=[],
        batch_size=32,
        shuffle=True,
    )
    dataloader.set_shard(shard_count=2, shard_index=0)
    assert dataloader.batch_size_per_gpu == 16


def test_two_shards_same_steps(_data_sources):
    dataloader = DataLoader(
        data_sources=_data_sources,
        augmentation_pipeline=[],
        batch_size=32,
        shuffle=True,
    )
    dataloader.set_shard(shard_count=2, shard_index=0)
    assert dataloader.steps == 4


def test_shard_count_less_than_one_raises(_data_sources):
    dataloader = DataLoader(
        data_sources=_data_sources,
        augmentation_pipeline=[],
        batch_size=32,
        shuffle=True,
    )
    with pytest.raises(ValueError):
        dataloader.set_shard(shard_count=0, shard_index=0)


def test_shard_index_outside_of_range_raises(_data_sources):
    dataloader = DataLoader(
        data_sources=_data_sources,
        augmentation_pipeline=[],
        batch_size=32,
        shuffle=True,
    )
    with pytest.raises(ValueError):
        dataloader.set_shard(shard_count=2, shard_index=2)


class DataLoaderTest(tf.test.TestCase):
    """Test the DataLoader."""

    def test_no_data_source(self):
        """Test that, if no sources are provided, the correct exception is raised."""
        with pytest.raises(ValueError):
            DataLoader(data_sources=[], augmentation_pipeline=[], batch_size=1)

    @parameterized.expand([(1, [1.0]), (2, [0.5, 0.5])])
    @patch(
        "nvidia_tao_tf1.blocks.multi_source_loader."
        "data_loader.tf.data.experimental.sample_from_datasets"
    )
    def test_data_source_combination_uniform_sampling(
        self, number_of_sources, expected_weights, mocked_sample_from_datasets
    ):
        """Test that the DataLoader interleaves the sources correctly with uniform sampling."""
        sources = []
        for _ in range(number_of_sources):
            sources.append(
                SyntheticDataSource(
                    preprocessing=[],
                    example_count=1,
                    template=test_fixtures.make_example_3d(12, 24),
                )
            )

        dataloader = DataLoader(
            data_sources=sources, augmentation_pipeline=[], batch_size=1
        )
        dataloader.set_shard()
        assert len(dataloader) == number_of_sources

        dataloader()

        mocked_sample_from_datasets.assert_called_once()
        call_args = mocked_sample_from_datasets.call_args
        weights = call_args[1]["weights"]
        assert weights == expected_weights

    @patch(
        "nvidia_tao_tf1.blocks.multi_source_loader."
        "data_loader.tf.data.experimental.sample_from_datasets"
    )
    def test_data_source_combination_specific_sampling(
        self, mocked_sample_from_datasets
    ):
        """Test that the DataLoader interleaves the sources correctly with given sampling ratios."""
        ratios = [0.2, 0.8]
        sources = [
            SyntheticDataSource(
                preprocessing=[],
                example_count=1,
                template=test_fixtures.make_example_3d(12, 24),
                sample_ratio=ratio,
            )
            for ratio in ratios
        ]

        dataloader = DataLoader(
            data_sources=sources, augmentation_pipeline=[], batch_size=1
        )
        dataloader.set_shard()

        assert dataloader._temporal_size is None
        assert len(dataloader) == len(ratios)

        dataloader()

        mocked_sample_from_datasets.assert_called_once()
        call_args = mocked_sample_from_datasets.call_args
        weights = call_args[1]["weights"]
        assert weights == [0.2, 0.8]

    @parameterized.expand(
        [("proportional", [1.0 / 6.0, 1.0 / 3.0, 0.5]), ("uniform", [1.0 / 3.0] * 3)]
    )
    @patch(
        "nvidia_tao_tf1.blocks.multi_source_loader."
        "data_loader.tf.data.experimental.sample_from_datasets"
    )
    def test_data_source_default_sampling_modes(
        self, sampling, expected_ratios, mocked_sample_from_datasets
    ):
        """Test that the DataLoader sets up expected sampling ratios for supported modes."""
        sources = [
            SyntheticDataSource(
                preprocessing=[],
                example_count=i + 1,
                template=test_fixtures.make_example_3d(12, 24),
            )
            for i in range(3)
        ]

        dataloader = DataLoader(
            data_sources=sources,
            augmentation_pipeline=[],
            batch_size=1,
            sampling=sampling,
        )
        dataloader.set_shard()

        dataloader()

        mocked_sample_from_datasets.assert_called_once()
        call_args = mocked_sample_from_datasets.call_args
        weights = call_args[1]["weights"]
        self.assertAllClose(weights, expected_ratios)

    def test_mixing_transform_and_nontransform_processors_does_not_fail(self):
        """Test that the dataloader works with source and dataloader processors."""
        window_size = 1
        batch_size = 1
        source = SyntheticDataSource(
            preprocessing=[processors.Scale(height=6, width=12)],
            example_count=1,
            template=test_fixtures.make_example_3d(12, 24),
        )
        dataloader = DataLoader(
            data_sources=[source],
            augmentation_pipeline=[
                processors.RandomZoom(),
                processors.RandomGaussianBlur(probability=1.0),
            ],
            preprocessing=[processors.TemporalBatcher(size=window_size)],
            batch_size=batch_size,
        )
        dataloader.set_shard()
        with self.cached_session() as sess:
            batch = dataloader()
            # First, initialize the iterator.
            sess.run(tf.compat.v1.get_collection("iterator_init"))
            example = sess.run(batch)
            assert (batch_size, window_size, 3, 6, 12) == example.instances[
                FEATURE_CAMERA
            ].images.shape

    @parameterized.expand([(1, [2]), (3, [1, 2, 3])])
    @patch(
        "nvidia_tao_tf1.blocks.multi_source_loader.data_loader.tf.data.Dataset.concatenate"
    )
    def test_data_source_concatenation_calls(
        self, number_of_sources, example_count, mocked_concatenate
    ):
        mocked_concatenate.side_effect = return_args
        sources = []
        for idx in range(number_of_sources):
            sources.append(
                SyntheticDataSource(
                    preprocessing=[],
                    example_count=example_count[idx],
                    template=test_fixtures.make_example_3d(12, 24),
                )
            )

        dataloader = DataLoader(
            data_sources=sources,
            augmentation_pipeline=[],
            batch_size=1,
            shuffle=False,
            repeat=False,
            sampling=None,
        )
        dataloader.set_shard()
        assert len(dataloader) == sum(
            [len(source) for source in dataloader._data_sources]
        )
        dataloader()
        assert mocked_concatenate.call_count == number_of_sources - 1

    def test_data_source_concatenation_number_examples(self):
        sources = []
        number_of_sources = 3
        example_count = [1, 2, 3]
        for idx in range(number_of_sources):
            sources.append(
                SyntheticDataSource(
                    preprocessing=[],
                    example_count=example_count[idx],
                    template=test_fixtures.make_example_3d(12, 24),
                )
            )

        dataloader = DataLoader(
            data_sources=sources,
            augmentation_pipeline=[],
            batch_size=1,
            shuffle=False,
            repeat=False,
            sampling=None,
        )
        dataloader.set_shard()
        number_of_examples = 0
        with self.cached_session() as sess:
            batch = dataloader()
            # First, initialize the iterator.
            sess.run(tf.compat.v1.get_collection("iterator_init"))
            while True:
                try:
                    sess.run(batch)
                    number_of_examples += 1
                except tf.errors.OutOfRangeError:
                    break
        assert number_of_examples == len(dataloader)

    @parameterized.expand([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3]])
    def test_produces_5d_examples(self, batch_size, window_size):
        source = SyntheticDataSource(
            preprocessing=[],
            example_count=window_size * 2,
            template=test_fixtures.make_example_3d(12, 24),
        )

        dataloader = DataLoader(
            data_sources=[source],
            augmentation_pipeline=[],
            batch_size=batch_size,
            preprocessing=[processors.TemporalBatcher(size=window_size)],
        )
        dataloader.set_shard()
        assert dataloader._temporal_size == window_size
        assert len(dataloader) == 2

        with self.cached_session() as sess:
            batch = dataloader()
            # First, initialize the iterator.
            sess.run(tf.compat.v1.get_collection("iterator_init"))
            example = sess.run(batch)
            assert (batch_size, window_size, 3, 12, 24) == example.instances[
                FEATURE_CAMERA
            ].images.shape

    def test_invalid_dataset_size(self):
        window_size = 6
        batch_size = 2
        source = SyntheticDataSource(
            preprocessing=[],
            example_count=window_size - 1,
            template=test_fixtures.make_example_3d(12, 24),
        )
        dataloader = DataLoader(
            data_sources=[source],
            augmentation_pipeline=[],
            batch_size=batch_size,
            preprocessing=[processors.TemporalBatcher(size=window_size)],
        )
        with self.assertRaises(ValueError):
            dataloader.set_shard()

    def test_produces_4d_examples_when_window_size_not_set(self):
        source = SyntheticDataSource(
            preprocessing=[],
            example_count=1,
            template=test_fixtures.make_example_3d(12, 24),
        )

        dataloader = DataLoader(
            data_sources=[source], augmentation_pipeline=[], batch_size=1
        )

        dataloader.set_shard()
        example = dataloader()
        assert (1, 3, 12, 24) == example.instances[FEATURE_CAMERA].images.shape

    def test_normalize_images(self):
        images = tf.zeros([2, 2, 2], dtype=tf.uint8)
        example = SequenceExample(
            instances={
                FEATURE_CAMERA: Images2D(images=images, canvas_shape=mock.Mock())
            },
            labels=mock.Mock(),
        )
        _normalize_images(example)

        assert example.instances[FEATURE_CAMERA].images.dtype == tf.float32
        with self.cached_session() as sess:
            normalized_images = sess.run(example.instances[FEATURE_CAMERA].images)
        self.assertAllInRange(normalized_images, 0.0, 1.0)

    @parameterized.expand(
        [
            [[UINT8DataSource()], tf.uint8],
            [[FP16DataSource()], tf.float16],
            [[FP32DataSource()], tf.float32],
            [[UINT8DataSource(), UINT8DataSource(), UINT8DataSource()], tf.uint8],
            [[FP16DataSource(), FP16DataSource(), FP16DataSource()], tf.float16],
            [[FP32DataSource(), FP32DataSource(), FP32DataSource()], tf.float32],
            [[FP16DataSource(), FP32DataSource(), UINT8DataSource()], tf.float32],
            [[UINT8DataSource(), UINT8DataSource(), FP16DataSource()], tf.float16],
            [[FP32DataSource(), FP16DataSource(), FP32DataSource()], tf.float32],
        ]
    )
    def test_pick_image_dtype(self, data_sources, expected_dtype):
        dtype = _pick_largest_image_dtype(data_sources)
        assert expected_dtype == dtype

    def test_stringifies(self):
        source = _datasource_mock()

        dataloader = DataLoader(
            data_sources=[source], augmentation_pipeline=[], batch_size=1
        )
        dataloader.set_shard()

        self.assertEqual(
            "  - examples: 42\n"
            "  - steps: 42\n"
            "  - batch size per gpu: 1\n"
            "  - shuffle: True\n"
            "  - shard count: 1\n"
            "  - shard index: 0\n"
            "  - pseudo-sharding: False\n"
            "  - serial augmentation: False\n"
            "  - sources:\n"
            "    Source 0: Mock\n",
            str(dataloader),
        )

    def test_normalizes_image_sizes(self):
        mocks = [_datasource_mock(64, 504), _datasource_mock(128, 42)]
        # The data loader should query each data source, and gather their image
        # dimensions. Finally, it should tell each what the max dimensions are
        # across all data sources.
        dataloader = DataLoader(
            data_sources=mocks, augmentation_pipeline=[], batch_size=1
        )
        dataloader.set_shard()

        for ds_mock in mocks:
            ds_mock.set_image_properties.assert_called_once_with(128, 504)

    @mock.patch("nvidia_tao_tf1.blocks.multi_source_loader.data_loader.print")
    def test_summarizes_to_stdout(self, mocked_print):
        dataloader = DataLoader(
            data_sources=[_datasource_mock()], augmentation_pipeline=[], batch_size=1
        )
        dataloader.set_shard()

        dataloader.summary()
        mocked_print.assert_has_calls(
            [
                mock.call("  - examples: 42"),
                mock.call("  - steps: 42"),
                mock.call("  - batch size per gpu: 1"),
                mock.call("  - shuffle: True"),
                mock.call("  - shard count: 1"),
                mock.call("  - shard index: 0"),
                mock.call("  - pseudo-sharding: False"),
                mock.call("  - serial augmentation: False"),
                mock.call("  - sources:"),
            ]
        )

    def test_summarizes_to_print_fn(self):
        dataloader = DataLoader(
            data_sources=[_datasource_mock()], augmentation_pipeline=[], batch_size=1
        )
        dataloader.set_shard()

        print_fn = mock.Mock()
        dataloader.summary(print_fn=print_fn)
        print_fn.assert_has_calls(
            [
                mock.call("  - examples: 42"),
                mock.call("  - steps: 42"),
                mock.call("  - batch size per gpu: 1"),
                mock.call("  - shuffle: True"),
                mock.call("  - shard count: 1"),
                mock.call("  - shard index: 0"),
                mock.call("  - pseudo-sharding: False"),
                mock.call("  - serial augmentation: False"),
                mock.call("  - sources:"),
            ]
        )

    @parameterized.expand([(7, True), (13, False), (16, True)])
    def test_serialization_roundtrip(self, batch_size, shuffle):
        _data_sources = []
        for _ in range(2):
            _data_sources.append(
                SyntheticDataSource(
                    example_count=3,  # some prime number.
                    template_fn=_synthetic_data_source_template_fn,
                    tracker_dict=dict(),
                )
            )

        dataloader = DataLoader(
            data_sources=_data_sources,
            augmentation_pipeline=[],
            batch_size=batch_size,
            shuffle=shuffle,
        )
        dataloader.set_shard()

        s = dataloader.serialize()

        dataloader_2 = deserialize_tao_object(s)
        dataloader_2.set_shard()

        # Check some properties / attributes are as expected.
        assert dataloader_2.steps == dataloader.steps
        assert dataloader_2.batch_size_per_gpu == dataloader.batch_size_per_gpu
        assert len(dataloader_2) == len(dataloader)

    @mock.patch(
        "nvidia_tao_tf1.blocks.multi_source_loader.data_loader.DataLoader._configure_sources"
    )
    def test_set_shard(self, mocked_configure_sources):
        sources = []
        for _ in range(2):
            sources.append(
                SyntheticDataSource(
                    example_count=16,  # some prime number.
                    template_fn=_synthetic_data_source_template_fn,
                    tracker_dict=dict(),
                )
            )

        dataloader = DataLoader(
            data_sources=sources, augmentation_pipeline=[], batch_size=16
        )
        dataloader.set_shard(shard_count=4, shard_index=3)
        assert dataloader._shard_index == 3
        assert dataloader._shard_count == 4
        assert dataloader._batch_size_per_gpu == 4
        assert dataloader.steps == 2
        assert len(dataloader) == 32
        mocked_configure_sources.assert_called_once()

    def test_set_shard_adjust_on_batch_size(self):
        sources = []
        batch_size = 16
        shard_count = 8
        for _ in range(2):
            sources.append(
                SyntheticDataSource(
                    example_count=16,  # some even number.
                    template_fn=_synthetic_data_source_template_fn,
                    tracker_dict=dict(),
                )
            )
        dataloader = DataLoader(
            data_sources=sources, augmentation_pipeline=[], batch_size=batch_size
        )
        # When batch_size is set,
        # dataloader.batch_size = batch_size / shard_count.
        dataloader.set_shard(shard_count=shard_count, shard_index=0)
        assert batch_size / shard_count == dataloader.batch_size_per_gpu

    def test_set_shard_adjust_on_batch_size_per_gpu(self):
        sources = []
        batch_size_per_gpu = 4
        for _ in range(2):
            sources.append(
                SyntheticDataSource(
                    example_count=16,  # some even number.
                    template_fn=_synthetic_data_source_template_fn,
                    tracker_dict=dict(),
                )
            )
        dataloader = DataLoader(
            data_sources=sources,
            augmentation_pipeline=[],
            batch_size_per_gpu=batch_size_per_gpu,
        )
        # When batch_size_per_gpu is set,
        # dataloader.batch_size = batch_size_per_gpu.
        dataloader.set_shard(shard_count=8, shard_index=0)
        assert batch_size_per_gpu == dataloader.batch_size_per_gpu

    def test_raises_error_calling_to_len_before_set_shard(self):
        dataloader = DataLoader(
            data_sources=[_datasource_mock()], augmentation_pipeline=[], batch_size=1
        )
        with self.assertRaises(ValueError):
            len(dataloader)

    def test_raises_error_calling_steps_before_set_shard(self):
        dataloader = DataLoader(
            data_sources=[_datasource_mock()], augmentation_pipeline=[], batch_size=1
        )
        with self.assertRaises(ValueError):
            dataloader.steps

    @parameterized.expand(
        [
            (-1, -1, "Exactly one of batch_size and batch_size_per_gpu must be set."),
            (1, 1, "Exactly one of batch_size and batch_size_per_gpu must be set."),
            (1, -1, "Exactly one of batch_size and batch_size_per_gpu must be set."),
            (-1, 1, "Exactly one of batch_size and batch_size_per_gpu must be set."),
            (
                None,
                None,
                "Exactly one of batch_size and batch_size_per_gpu must be set.",
            ),
            (None, -1, ".* must be positive."),
            (None, -1, ".* must be positive."),
            (None, 0, ".* must be positive."),
            (0, None, ".* must be positive."),
        ]
    )
    def test_raise_error_with_illegal_batch_size(
        self, batch_size, batch_size_per_gpu, message
    ):
        with self.assertRaisesRegexp(ValueError, message):
            DataLoader(
                data_sources=[_datasource_mock()],
                augmentation_pipeline=[],
                batch_size=batch_size,
                batch_size_per_gpu=batch_size_per_gpu,
            )

    def test_raise_error_when_neither_batch_size_is_set(self):
        message = "Exactly one of batch_size and batch_size_per_gpu must be set."
        with self.assertRaisesRegexp(ValueError, message):
            DataLoader(data_sources=[_datasource_mock()], augmentation_pipeline=[])
