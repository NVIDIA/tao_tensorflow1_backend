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

"""Unit tests for Coordinates2D."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from parameterized import parameterized
import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    test_fixtures as fixtures,
)
from nvidia_tao_tf1.core.types import Transform


class Coordinates2DTest(tf.test.TestCase):
    @parameterized.expand(
        [
            [[[1]]],
            [[[1, 2]]],
            [[[1], [2]]],
            [[[1, 2], [2, 3]]],
            [[[1, 2, 3], [4, 5, 6]]],
        ]
    )
    def test_apply_succeeds(self, shapes_per_frame):
        with self.session() as sess:
            example_count = len(shapes_per_frame)
            coordinates = fixtures.make_coordinates2d(
                shapes_per_frame=shapes_per_frame, height=604, width=960
            )

            transform = fixtures.make_identity_transform(
                example_count,
                604,
                960,
                timesteps=coordinates.canvas_shape.width.shape[1],
            )
            transformed = coordinates.apply(transform)
            self.assertAllClose(
                sess.run(coordinates.coordinates), sess.run(transformed.coordinates)
            )
            self.assertAllClose(
                sess.run(coordinates.canvas_shape), sess.run(transformed.canvas_shape)
            )

    @parameterized.expand(
        [
            [[[1]], [(5, 10)]],  # 1 example with 1 frame and 1 shape:
            # translate x by 5, y by 10 pixels
            [
                [[1, 2]],
                [(5, 10)],
            ],  # 1 example with 2 frames. First frame has 1 shape second has 2.
            # translate x by 5, y by 10 pixels
            [[[1], [2]], [(5, 10), (7, 42)]],
            [[[1, 2], [2, 3]], [(10, 5), (42, 7)]],
            [
                [[1, 2, 3], [4, 5, 6]],
                [(3, 6), (7, 14)],
            ],  # 2 examples each with 3 frames. Translate
            # first example x by 3 and y by 6 pixels
            # second example x by 7 and y by 14 pixels
        ]
    )
    def test_applies_translations_per_example(
        self, shapes_per_frame, per_example_translations
    ):
        with self.session() as sess:
            example_count = len(shapes_per_frame)
            translate_count = len(per_example_translations)
            assert example_count == translate_count

            height = 604
            width = 960
            transform = fixtures.make_identity_transform(example_count, height, width)

            def make_translate_matrix(x, y):
                return tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [x, y, 1.0]])

            transform = Transform(
                canvas_shape=fixtures.make_canvas2d(example_count, height, width),
                color_transform_matrix=tf.stack(
                    [tf.eye(4, dtype=tf.float32) for _ in range(example_count)]
                ),
                spatial_transform_matrix=tf.stack(
                    [make_translate_matrix(x, y) for x, y in per_example_translations]
                ),
            )

            coordinates = fixtures.make_coordinates2d(
                shapes_per_frame=shapes_per_frame, height=height, width=width
            )
            transformed = coordinates.apply(transform)

            offset_accumulator = 0
            for example_index, example_frame_shapes in enumerate(shapes_per_frame):
                coordinate_count = 3 * sum(example_frame_shapes)

                start_offset = offset_accumulator
                end_offset = offset_accumulator + coordinate_count

                x_translate = per_example_translations[example_index][0]
                y_translate = per_example_translations[example_index][1]

                x_expected = (
                    tf.reshape(coordinates.coordinates.values, (-1, 2))[
                        start_offset:end_offset, 0
                    ]
                    - x_translate
                )
                x_transformed = tf.reshape(transformed.coordinates.values, (-1, 2))[
                    start_offset:end_offset, 0
                ]

                self.assertAllClose(
                    sess.run(x_expected), sess.run(x_transformed), rtol=1e-3, atol=1e-3
                )

                y_expected = (
                    tf.reshape(coordinates.coordinates.values, (-1, 2))[
                        start_offset:end_offset, 1
                    ]
                    - y_translate
                )
                y_transformed = tf.reshape(transformed.coordinates.values, (-1, 2))[
                    start_offset:end_offset, 1
                ]

                self.assertAllClose(
                    sess.run(y_expected), sess.run(y_transformed), rtol=1e-3, atol=1e-3
                )

                offset_accumulator += coordinate_count
