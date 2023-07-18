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

"""Unit tests for Bbox2DLabel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import mock

import numpy as np
from parameterized import parameterized
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    test_fixtures as fixtures,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.bbox_2d_label import (
    _get_begin_and_end_indices,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.bbox_2d_label import _to_ltrb
from nvidia_tao_tf1.blocks.multi_source_loader.types.bbox_2d_label import (
    augment_marker_labels,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.bbox_2d_label import (
    Bbox2DLabel,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.bbox_2d_label import (
    filter_bbox_label_based_on_minimum_dims,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.coordinates2d import (
    Coordinates2D,
)
from modulus.processors.augment.spatial import get_random_spatial_transformation_matrix
from modulus.types import Canvas2D
from modulus.types import Transform


def _get_bbox_2d_label(shapes_per_frame, height=604, width=960):
    """
    shapes_per_frame: Outer list is for example, inner list is for frames.
    """
    label_kwargs = dict()
    label_kwargs["frame_id"] = []  # Bogus.
    label_kwargs["vertices"] = fixtures.make_coordinates2d(
        shapes_per_frame=shapes_per_frame,
        height=height,
        width=width,
        coordinates_per_polygon=2,
    )

    # tags = [examples[frames[shapes[tags,]]]]
    def get_tags(tag):
        tags = [
            [[[tag] * num_shapes] for num_shapes in example]
            for example in shapes_per_frame
        ]

        return fixtures.make_tags(tags)

    label_kwargs["object_class"] = get_tags("van")
    label_kwargs["occlusion"] = get_tags(0)
    label_kwargs["truncation"] = get_tags(0.0)
    label_kwargs["truncation_type"] = get_tags(0)
    label_kwargs["is_cvip"] = get_tags(False)
    label_kwargs["world_bbox_z"] = get_tags(-1.0)
    label_kwargs["non_facing"] = get_tags(False)
    label_kwargs["front"] = get_tags(-1.0)
    label_kwargs["back"] = get_tags(-1.0)

    label_kwargs["source_weight"] = get_tags(1.0)
    return Bbox2DLabel(**label_kwargs)


class Bbox2DLabelTest(tf.test.TestCase):
    """Test Bbox2dLabel and helper functions."""

    @parameterized.expand([[[[2, 3], [4]]]])
    def test_apply_calls_coordinates_transform(self, shapes_per_frame):
        """Test that the .vertices are calling the apply() method of Coordinates2D."""
        example_count = len(shapes_per_frame)
        height, width = 604, 960
        bbox_2d_label = _get_bbox_2d_label(shapes_per_frame, height, width)
        transform = fixtures.make_identity_transform(example_count, height, width)

        with mock.patch.object(
            bbox_2d_label.vertices, "apply", side_effect=bbox_2d_label.vertices.apply
        ) as spied_vertices_apply:
            bbox_2d_label.apply(transform)

            spied_vertices_apply.assert_called_with(transform)

    @parameterized.expand([[[[3, 2], [1]]], [[[4, 6], [0]]]])
    def test_to_ltrb(self, shapes_per_frame):
        """Test that _to_ltrb() works as advertised."""
        height, width = 604, 960
        coordinates_2d = fixtures.make_coordinates2d(
            shapes_per_frame=shapes_per_frame,
            height=height,
            width=width,
            coordinates_per_polygon=2,
        )

        with self.session() as session:
            ltrb_coordinates = session.run(_to_ltrb(coordinates_2d.coordinates.values))

        xmin = ltrb_coordinates[::4]
        xmax = ltrb_coordinates[2::4]
        ymin = ltrb_coordinates[1::4]
        ymax = ltrb_coordinates[3::4]

        self.assertTrue((xmax >= xmin).all())
        self.assertTrue((ymax >= ymin).all())

    @parameterized.expand(
        [
            [
                [[2, 3], [4, 5]],  # shapes_per_frame
                [0, 5],  # expected_begin_indices
                [5, 14],  # expected_end_indices
                range(2),
            ]
        ]
    )
    def test_get_begin_and_end_indices(
        self,
        shapes_per_frame,
        expected_begin_indices,
        expected_end_indices,
        expected_indices_index,
    ):
        """Test _get_begin_and_end_indices."""
        height, width = 604, 960
        bbox_2d_label = _get_bbox_2d_label(shapes_per_frame, height, width)
        # Use object_class as a tf.SparseTensor to test with.
        sparse_tensor = bbox_2d_label.object_class

        with self.session() as session:
            begin_indices, end_indices, indices_index = session.run(
                _get_begin_and_end_indices(sparse_tensor)
            )

        self.assertAllEqual(begin_indices, expected_begin_indices)
        self.assertAllEqual(end_indices, expected_end_indices)
        self.assertAllEqual(indices_index, expected_indices_index)

    @parameterized.expand([[[[2, 3], [4]]]])
    def test_apply_calls_to_ltrb(self, shapes_per_frame):
        """Test that the apply method calls _to_ltrb."""
        example_count = len(shapes_per_frame)
        height, width = 604, 960
        bbox_2d_label = _get_bbox_2d_label(shapes_per_frame, height, width)
        transform = fixtures.make_identity_transform(example_count, height, width)

        with mock.patch(
            "nvidia_tao_tf1.blocks.multi_source_loader.types.bbox_2d_label._to_ltrb",
            side_effect=_to_ltrb,
        ) as spied_to_ltrb:
            bbox_2d_label.apply(transform)

            spied_to_ltrb.assert_called_once()

    @parameterized.expand([[[[3, 2], [1]], [0, 4]]])
    def test_filter(self, shapes_per_frame, valid_indices):
        """Test that the filter method works as advertised."""
        height, width = 604, 960
        bbox_2d_label = _get_bbox_2d_label(shapes_per_frame, height, width)

        # Convert test arg to appropriate format.
        num_indices = sum(functools.reduce(lambda x, y: x + y, shapes_per_frame))
        valid_indices_tensor = tf.constant(
            [index in valid_indices for index in range(num_indices)]
        )
        filtered_label = bbox_2d_label.filter(valid_indices_tensor)

        with self.session() as session:
            original_output, filtered_output = session.run(
                [bbox_2d_label, filtered_label]
            )

        num_values = len(valid_indices)
        # Check that the output sizes are sane, and that their contents are as
        # expected.
        for target_feature_name in Bbox2DLabel.TARGET_FEATURES:
            if target_feature_name == "vertices":
                original_feature = original_output.vertices.coordinates.values.reshape(
                    (-1, 4)
                )
                filtered_feature = filtered_output.vertices.coordinates.values
                filtered_feature = filtered_feature.reshape((-1, 4))
                assert filtered_feature.size == num_values * 4
            else:
                original_feature = getattr(original_output, target_feature_name).values
                filtered_feature = getattr(filtered_output, target_feature_name).values
                assert filtered_feature.size == num_values
            for filtered_index, original_index in enumerate(valid_indices):
                self.assertAllEqual(
                    original_feature[original_index], filtered_feature[filtered_index]
                )

    @parameterized.expand(
        [
            (0.0, [0.1, 0.2, 0.3, -1.0], [0.1, 0.2, 0.3, -1.0]),
            (1.0, [0.1, 0.2, 0.3, -1.0], [0.9, 0.8, 0.7, -1.0]),
        ]
    )
    def test_augment_marker_labels(
        self, hflip_probability, marker_labels, expected_labels
    ):
        """Test that augment_marker_labels augments the marker values correctly."""
        # Get an stm.
        stm = get_random_spatial_transformation_matrix(
            width=10,
            height=11,
            flip_lr_prob=hflip_probability,
            # The below shouldn't affect the test, so it is ok that they are random.
            translate_max_x=5,
            translate_max_y=3,
            zoom_ratio_min=0.7,
            zoom_ratio_max=1.2,
        )

        # Get some orientation labels.
        marker_labels_tensor = tf.constant(marker_labels)

        augmented_marker_labels_tensor = augment_marker_labels(
            marker_labels_tensor, stm
        )

        with self.session() as session:
            augmented_marker_labels = session.run(augmented_marker_labels_tensor)

        self.assertAllClose(augmented_marker_labels, expected_labels)

    @parameterized.expand([[[[2, 3], [4]]]])
    def test_apply_calls_augment_marker_labels(self, shapes_per_frame):
        """Test that the apply method calls augment_marker_labels the expected number of times."""
        example_count = len(shapes_per_frame)
        height, width = 604, 960
        bbox_2d_label = _get_bbox_2d_label(shapes_per_frame, height, width)
        transform = fixtures.make_identity_transform(example_count, height, width)
        with mock.patch(
            "nvidia_tao_tf1.blocks.multi_source_loader."
            "types.bbox_2d_label.augment_marker_labels",
            side_effect=augment_marker_labels,
        ) as spied_augment_marker_labels:
            bbox_2d_label.apply(transform)

            assert spied_augment_marker_labels.call_count == len(shapes_per_frame)

    @parameterized.expand([([1.5, 0.8], [[3], [2, 1]])])
    def test_augment_depth(self, zoom_factors, shapes_per_frame):
        """Test that depth values are augmented as expected.

        Args:
            zoom_factors (list): List of zoom factors (float). Each element is the zoom factor
               to apply to the corresponding example.
            shapes_per_frame (list of lists): As expected by the make_coordinates2d test fixture.
        """
        # First, get random spatial augmentation matrices with the expected zoom factors.
        height, width = 604, 960
        transform = Transform(
            canvas_shape=fixtures.make_canvas2d(
                count=len(shapes_per_frame), height=height, width=width
            ),
            color_transform_matrix=tf.stack(
                [tf.eye(4, dtype=tf.float32) for _ in shapes_per_frame]
            ),
            spatial_transform_matrix=tf.stack(
                [
                    get_random_spatial_transformation_matrix(
                        width=width,
                        height=height,
                        flip_lr_prob=0.5,
                        translate_max_x=4,
                        translate_max_y=6,
                        zoom_ratio_min=zoom_factor,
                        zoom_ratio_max=zoom_factor,
                    )
                    for zoom_factor in zoom_factors
                ]
            ),
        )
        bbox_2d_label = _get_bbox_2d_label(shapes_per_frame, height, width)

        transformed_label = bbox_2d_label.apply(transform)

        with self.session() as session:
            original_depth, output_depth = session.run(
                [bbox_2d_label.world_bbox_z, transformed_label.world_bbox_z]
            )

        i = 0
        for zoom_factor, shape_list in zip(zoom_factors, shapes_per_frame):
            num_shapes = sum(shape_list)
            for _ in range(num_shapes):
                self.assertAllClose(
                    zoom_factor * original_depth.values[i], output_depth.values[i]
                )
                i += 1

    def test_filter_bbox_label_based_on_minimum_dims(self):
        """Test that fitler_bbox_label_based_on_minimum_dims works as advertised."""
        num_objects = np.random.randint(low=5, high=10)
        # Choose some indices for which the minimum dimensions will be satisfied.
        expected_valid_indices = np.random.choice(num_objects, 4, replace=False)
        random_ltrb_coords = []

        min_height, min_width = np.random.uniform(low=0.0, high=3.0, size=2)

        for i in range(num_objects):
            x1, y1 = np.random.uniform(low=0.0, high=100.0, size=2)
            if i in expected_valid_indices:
                # Create coordinates for which the minimum dimensions will be satisfied.
                x2 = x1 + min_width + 1.0
                y2 = y1 + min_height + 1.0
            else:
                # Create coordinates for which the minimum dimensions WILL NOT be satisfied.
                x2 = x1 + min_width - 1.0
                y2 = y1 + min_height - 1.0
            random_ltrb_coords.extend([x1, y1, x2, y2])

        # Now, cast it into a ``Bbox2DLabel``.
        kwargs = {field_name: [] for field_name in Bbox2DLabel._fields}
        kwargs["vertices"] = Coordinates2D(
            coordinates=tf.SparseTensor(
                values=tf.constant(random_ltrb_coords),
                indices=[[i] for i in range(num_objects * 4)],
                dense_shape=[num_objects * 4],
            ),
            canvas_shape=Canvas2D(height=604, width=960),
        )
        bbox_2d_label = Bbox2DLabel(**kwargs)

        with mock.patch(
            "nvidia_tao_tf1.blocks.multi_source_loader.types.bbox_2d_label.Bbox2DLabel.filter"
        ) as mocked_filter:
            filter_bbox_label_based_on_minimum_dims(
                bbox_2d_label=bbox_2d_label, min_height=min_height, min_width=min_width
            )

            # Now, check that filter was called correctly.
            computed_valid_indices_tensor = mocked_filter.call_args[1].pop(
                "valid_indices"
            )

        with self.session() as session:
            computed_valid_indices = session.run(computed_valid_indices_tensor)

        self.assertEqual(computed_valid_indices.shape, (num_objects,))

        for idx, computed_validity in enumerate(computed_valid_indices):
            expected_validity = idx in expected_valid_indices
            self.assertEqual(computed_validity, expected_validity)
