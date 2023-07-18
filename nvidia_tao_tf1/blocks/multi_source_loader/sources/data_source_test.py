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

"""Tests for DataSource."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import mock
import pytest
import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader.sources.data_source import (
    DataSource,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


class TestDataSource(DataSource):
    """Mocked DataSource class for testing."""

    def __init__(self, sample_ratio=1.0, extension=".jpg"):
        super(TestDataSource, self).__init__(
            sample_ratio=sample_ratio, preprocessing=[], extension=extension
        )

    def call(self):
        """Returns a Mock object for testing."""
        return mock.Mock()

    def __len__(self):
        return 1


class JpegDataSource(DataSource):
    """Mocked Datasource with extension being .jpeg"""

    def __init__(self):
        super().__init__(extension=".jpeg")

    def call(self):
        """Returns a Mock object for testing."""
        return mock.Mock()

    def __len__(self):
        return 1


class PNGDataSource(DataSource):
    """Mocked Datasource with extension being .jpeg"""

    def __init__(self):
        super().__init__(extension=".png")

    def call(self):
        """Returns a Mock object for testing."""
        return mock.Mock()

    def __len__(self):
        return 1


class FP16DataSource(DataSource):
    """Mocked Datasource with extension being .fp16"""

    def __init__(self):
        super().__init__(extension=".fp16")

    def call(self):
        """Returns a Mock object for testing."""
        return mock.Mock()

    def __len__(self):
        return 1


class FP32DataSource(DataSource):
    """Mocked Datasource with extension being .fp32"""

    def __init__(self):
        super().__init__(extension=".fp32")

    def call(self):
        """Returns a Mock object for testing."""

    def __len__(self):
        return 1


class DataSourceTest(unittest.TestCase):
    def test_stringifies(self):
        source = TestDataSource()

        self.assertEqual(
            "  - samples: 1\n" "  - sample ratio: 1.0\n" "  - extension: .jpg\n",
            str(source),
        )

    @staticmethod
    @mock.patch(
        "nvidia_tao_tf1.blocks.multi_source_loader.sources.data_source.print"
    )
    def test_summarizes_to_stdout(mocked_print):
        source = TestDataSource()

        source.summary()

        mocked_print.assert_has_calls(
            [
                mock.call("  - samples: 1"),
                mock.call("  - sample ratio: 1.0"),
                mock.call("  - extension: .jpg"),
            ]
        )

    @staticmethod
    def test_summarizes_using_print_fn():
        source = TestDataSource()
        print_fn = mock.Mock()

        source.summary(print_fn=print_fn)

        print_fn.assert_has_calls(
            [
                mock.call("  - samples: 1"),
                mock.call("  - sample ratio: 1.0"),
                mock.call("  - extension: .jpg"),
            ]
        )

    def test_serialization_and_deserialization(self):
        """Test TAOObject serialization and deserialization on TestDataSource."""
        source = TestDataSource()
        source_dict = source.serialize()
        deserialized_source = deserialize_tao_object(source_dict)
        self.assertEqual(str(source), str(deserialized_source))

    def test_raises_error_with_negative_sample_ratio(self):
        with pytest.raises(ValueError) as exception:
            TestDataSource(sample_ratio=-1.0)
        assert str(exception.value) == "Sample ratio -1.0 cannot be < 0."


@pytest.mark.parametrize(
    "data_source, expected_image_dtype",
    [
        (JpegDataSource(), tf.uint8),
        (FP16DataSource(), tf.float16),
        (FP32DataSource(), tf.float32),
        (PNGDataSource(), tf.uint8),
    ],
)
def test_image_dtype(data_source, expected_image_dtype):
    assert data_source.image_dtype == expected_image_dtype
