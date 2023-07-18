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

"""Tests for PolylineToPolygon processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from parameterized import parameterized
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors.polyline_to_polygon import (
    PolylineToPolygon,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor_test_case import (
    ProcessorTestCase,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.coordinates2d import (
    Coordinates2D,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.polygon2d_label import (
    Polygon2DLabel,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sparse_tensor_builder import (
    SparseTensorBuilder,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


###
# Convenience shorthands for building out the coordinate tensors
###
class C(SparseTensorBuilder):
    """Coordinates."""

    pass


class Poly(SparseTensorBuilder):
    """Polygon/Polyline."""

    pass


class Frame(SparseTensorBuilder):
    """Frame."""

    pass


class Timestep(SparseTensorBuilder):
    """Timestep."""

    pass


class Batch(SparseTensorBuilder):
    """Batch."""

    pass


class Label(SparseTensorBuilder):
    """Class Label."""

    pass


###


def _get_label(label_builder, label_classes_builder):
    sparse_example = label_builder.build(val_type=tf.float32)
    sparse_classes = label_classes_builder.build(val_type=tf.int32)

    coordinates = Coordinates2D(coordinates=sparse_example, canvas_shape=tf.zeros(1))

    label = Polygon2DLabel(
        vertices=coordinates, classes=sparse_classes, attributes=tf.zeros(1)
    )
    return label


class TestPolylineToPolygon(ProcessorTestCase):
    @parameterized.expand([[tuple()], [(5,)], [(12, 2)]])
    def test_no_examples(self, batch_args):
        empty_polygon2d = self.make_empty_polygon2d_labels(*batch_args)

        processor = PolylineToPolygon(2, line_width=2.0, debug=True)

        with self.session() as sess:
            processed_input, processed_output = sess.run(
                [empty_polygon2d, processor.process(empty_polygon2d)]
            )

            self.assertSparseEqual(
                processed_input.vertices.coordinates,
                processed_output.vertices.coordinates,
            )
            self.assertSparseEqual(processed_input.classes, processed_output.classes)

    def test_no_convert_single_example(self):
        example = Frame(
            Poly(C(0, 0), C(1, 0), C(1, 1), C(0, 1)), Poly(C(0, 2), C(1, 1), C(0, 0))
        )
        example_classes = Frame(Label(0), Label(5))

        self._test_no_convert(example, example_classes)

    def test_no_convert_batch(self):
        example = Batch(
            Frame(Poly(C(0, 0), C(2, 2), C(2, 0))),
            Frame(
                Poly(C(5, 5), C(6, 5), C(6, 6), C(5, 6)),
                Poly(C(1, 5), C(5, 5), C(6, 5)),
            ),
        )
        example_classes = Batch(Frame(Label(0)), Frame(Label(1), Label(3)))

        self._test_no_convert(example, example_classes)

    def test_convert_simple(self):
        example = Frame(Poly(C(0, 0), C(1, 0), C(1, 1)))
        example_classes = Frame(Label(2))  # Gets converted

        target_example = Frame(
            Poly(C(0, -1), C(1, -1), C(2, 0), C(1, 1), C(0, 1), C(-1, 0)),
            Poly(C(2, 0), C(2, 1), C(1, 2), C(0, 1), C(0, 0), C(1, -1)),
        )
        target_example_classes = Frame(
            Label(2),  # The polyline has 2 segments, so 2 polygons will be created
            Label(2),
        )

        self._test_convert(
            example, example_classes, target_example, target_example_classes
        )

    def test_convert_empty_first(self):
        example = Batch(Timestep(Frame(), Frame(Poly(C(0, 0), C(1, 0)))))
        example_classes = Batch(Timestep(Frame(), Frame(Label(2))))  # Gets converted

        target_example = Batch(
            Timestep(
                Frame(),
                Frame(Poly(C(0, -1), C(1, -1), C(2, 0), C(1, 1), C(0, 1), C(-1, 0))),
            )
        )
        target_example_classes = example_classes

        self._test_convert(
            example, example_classes, target_example, target_example_classes
        )

    def test_convert_piece(self):
        example = Frame(
            Poly(C(1, 1), C(3, 1), C(2, 2)),
            Poly(C(0, 0), C(0, 1)),
            Poly(C(5, 5), C(6, 5), C(6, 6), C(5, 6)),
        )
        example_classes = Frame(Label(0), Label(2), Label(1))  # Gets converted

        target_example = Frame(
            Poly(C(1, 1), C(3, 1), C(2, 2)),
            Poly(C(1, 0), C(1, 1), C(0, 2), C(-1, 1), C(-1, 0), C(0, -1)),
            Poly(C(5, 5), C(6, 5), C(6, 6), C(5, 6)),
        )
        target_example_classes = example_classes

        self._test_convert(
            example, example_classes, target_example, target_example_classes
        )

    def _test_no_convert(self, example, example_classes):
        self._test_convert(example, example_classes, example, example_classes)

    def _test_convert(
        self, example, example_classes, target_example, target_example_classes
    ):
        input_labels = _get_label(example, example_classes)
        target_labels = _get_label(target_example, target_example_classes)

        processor = PolylineToPolygon(2, line_width=2.0, debug=True)

        with self.session() as sess:
            processed_converted, processed_target = sess.run(
                [processor.process(input_labels), target_labels]
            )

            self.assertSparseEqual(
                processed_target.vertices.coordinates,
                processed_converted.vertices.coordinates,
            )
            self.assertSparseEqual(
                processed_target.classes, processed_converted.classes
            )

    def test_serialization_and_deserialization(self):
        """Test that it is a TAOObject that can be serialized and deserialized."""
        processor = PolylineToPolygon(class_id=2, line_width=2.0, debug=True)
        processor_dict = processor.serialize()
        deserialized_dict = deserialize_tao_object(processor_dict)
        assert processor.class_id == deserialized_dict.class_id
        assert processor.line_width == deserialized_dict.line_width
        assert processor.debug == deserialized_dict.debug
