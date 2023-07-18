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

"""Test BboxClipper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from parameterized import parameterized
import pytest
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors.bbox_clipper import (
    BboxClipper,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import Bbox2DLabel
from nvidia_tao_tf1.blocks.multi_source_loader.types import Coordinates2D
from nvidia_tao_tf1.blocks.multi_source_loader.types import LABEL_OBJECT
from nvidia_tao_tf1.blocks.multi_source_loader.types import SequenceExample
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    sparsify_dense_coordinates,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import TransformedExample
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    vector_and_counts_to_sparse_tensor,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object
from modulus.types import Canvas2D
from modulus.types import Example


@pytest.mark.parametrize(
    "crop_left,crop_right,crop_top,crop_bottom",
    [
        (1, 0, 0, 2),  # crop_left > crop_right.
        (1, 1, 0, 2),  # crop_left = crop_right.
        (0, 5, 7, 6),  # crop_top > crop_bottom.
        (0, 5, 7, 7),  # crop_top = crop_bottom.
    ],
)
def test_bbox_clipper_raises(crop_left, crop_right, crop_top, crop_bottom):
    """Test that the appropriate error is raised when the input args are bogus."""
    with pytest.raises(ValueError):
        BboxClipper(
            crop_left=crop_left,
            crop_right=crop_right,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
        )


def test_bbox_clipper_process_raises_on_transformed_example():
    """Test that the appropriate error is raised when a TransformedExample is supplied."""
    bbox_clipper = BboxClipper()
    transformed_example = TransformedExample(example=None, transformation=None)
    with pytest.raises(ValueError):
        bbox_clipper.process(transformed_example)


class TestBboxClipper(tf.test.TestCase):
    """Test BboxClipper."""

    def _get_example(self, x, y, example_type=Example):
        """Get an example with a Bbox2DLabel for testing.

        Args:
            x (list): x coordinates. Expects a series of [xmin, xmax, xmin, xmax, ...].
            y (list): corresponding y coordinates.
            example_type (class): One of SequenceExample, Example.

        Returns:
            Properly populated Example with a Bbox2dLabel.
        """
        num_bboxes = tf.cast(tf.size(input=x) / 2, dtype=tf.int32)
        coordinates = sparsify_dense_coordinates(
            dense_coordinates=tf.stack([x, y], axis=1),
            vertex_counts_per_polygon=2 * tf.ones(num_bboxes, dtype=tf.int32),
        )
        truncation_type = vector_and_counts_to_sparse_tensor(
            vector=tf.zeros(num_bboxes, dtype=tf.int32),
            counts=tf.ones(num_bboxes, dtype=tf.int32),
        )
        # Initialize to empty fields.
        label_kwargs = {
            field_name: []
            for field_name in Bbox2DLabel.TARGET_FEATURES
            + Bbox2DLabel.FRAME_FEATURES
            + Bbox2DLabel.ADDITIONAL_FEATURES
        }

        label_kwargs["vertices"] = Coordinates2D(
            coordinates=coordinates,
            canvas_shape=Canvas2D(height=tf.constant(604), width=tf.constant(960)),
        )
        label_kwargs["truncation_type"] = truncation_type

        return example_type(
            instances=[], labels={LABEL_OBJECT: Bbox2DLabel(**label_kwargs)}
        )

    @parameterized.expand(
        [
            [[0.0, 1.0, 2.0, -3.0], [-4.0, 0.0, 2.0, 1.5], Example],
            [[0.0, 1.0, 2.0, -3.0], [-4.0, 0.0, 2.0, 1.5], SequenceExample],
            [[1.0 + i for i in range(8)], [2.1 + i for i in range(8)], Example],
        ]
    )
    def test_no_op(self, x, y, example_type):
        """Test that if crop coordinates are all 0, nothing happens."""
        bbox_clipper = BboxClipper(crop_left=0, crop_right=0, crop_top=0, crop_bottom=0)

        example = self._get_example(x=x, y=y, example_type=example_type)

        processed_example = bbox_clipper.process(example)

        with self.session() as session:
            original_label, output_label = session.run(
                [example.labels[LABEL_OBJECT], processed_example.labels[LABEL_OBJECT]]
            )

        original_coords = original_label.vertices.coordinates.values
        output_coords = output_label.vertices.coordinates.values

        self.assertAllEqual(original_coords, output_coords)

        original_truncation_type = original_label.truncation_type.values
        output_truncation_type = output_label.truncation_type.values

        self.assertAllEqual(original_truncation_type, output_truncation_type)

    @parameterized.expand(
        [
            # 1st bbox is half outside: should get clipped.
            [
                [-1.0, 3.0, 4.0, 7.5],
                [6.0, 7.0, 8.0, 12.0],  # x, y.
                [0.0, 6.0, 3.0, 7.0, 4.0, 8.0, 7.0, 11.0],  # expected_coords.
                [1, 1],  # expected_truncation_type.
            ],  # End of first test case.
            # 1st bbox is completely outside: should not appear in output.
            [
                [-3.0, -1.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0],  # x, y.
                [4.0, 8.0, 5.0, 9.0],  # expected_coords.
                [0],  # expected_truncation_type.
            ],  # End of second test case.
        ]
    )
    def test_process(self, x, y, expected_coords, expected_truncation_type):
        """Test that process does what it advertises (when it is not a no-op)."""
        bbox_clipper = BboxClipper(
            crop_left=0, crop_right=7, crop_top=0, crop_bottom=11
        )

        example = self._get_example(x=x, y=y)

        processed_example = bbox_clipper.process(example)

        with self.session() as session:
            output_label = session.run(processed_example.labels[LABEL_OBJECT])

        # Check coordinates.
        self.assertAllEqual(output_label.vertices.coordinates.values, expected_coords)

        # Check truncation_type.
        self.assertAllEqual(
            output_label.truncation_type.values, expected_truncation_type
        )

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        bbox_clipper = BboxClipper(
            crop_left=0, crop_right=7, crop_top=0, crop_bottom=11
        )
        bbox_clipper_dict = bbox_clipper.serialize()
        deserialized_dict = deserialize_tao_object(bbox_clipper_dict)
        self.assertAllEqual(bbox_clipper._crop_left, deserialized_dict._crop_left)
        self.assertAllEqual(bbox_clipper._crop_right, deserialized_dict._crop_right)
        self.assertAllEqual(bbox_clipper._crop_bottom, deserialized_dict._crop_bottom)
        self.assertAllEqual(bbox_clipper._crop_top, deserialized_dict._crop_top)
