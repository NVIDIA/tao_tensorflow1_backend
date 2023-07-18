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

"""Main test for polyline_clipper.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors.polyline_clipper import (
    PolylineClipper,
)


@pytest.fixture(scope="session")
def _clipper():
    return PolylineClipper(5)


# Model input width and height is 960 x 504.
polylines = np.array(
    [
        [300, 500],
        [300, 375],
        [300, 250],
        [300, 125],
        [300, 4],
        [660, 4],
        [660, 125],
        [660, 250],
        [660, 375],
        [660, 500],
        [390, 350],
        [390, 300],
        [390, 250],
        [390, 200],
        [390, 150],
        [570, 150],
        [570, 200],
        [570, 250],
        [570, 300],
        [570, 350],
        [950, 400],
        [715, 400],
        [480, 400],
        [245, 400],
        [10, 400],
        [10, 100],
        [245, 100],
        [480, 100],
        [715, 100],
        [950, 100],
        [720, 325],
        [600, 325],
        [480, 325],
        [360, 325],
        [240, 325],
        [240, 175],
        [360, 175],
        [480, 175],
        [600, 175],
        [720, 175],
    ],
    dtype=float,
)
vertices_per_polyline = np.array([5, 5, 5, 5, 5, 5, 5, 5], dtype=int)
class_ids_per_polyline = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=int)
attributes_per_polyline = np.array([-1, 1, -1, 1, -1, 1, -1, 1], dtype=int)

polylines_all_inside = np.array(
    [
        [300, 500],
        [300, 375],
        [300, 250],
        [300, 125],
        [300, 4],
        [660, 4],
        [660, 125],
        [660, 250],
        [660, 375],
        [660, 500],
        [390, 350],
        [390, 300],
        [390, 250],
        [390, 200],
        [390, 150],
        [570, 150],
        [570, 200],
        [570, 250],
        [570, 300],
        [570, 350],
        [10, 400],
        [245, 400],
        [480, 400],
        [715, 400],
        [950, 400],
        [950, 100],
        [715, 100],
        [480, 100],
        [245, 100],
        [10, 100],
        [240, 325],
        [360, 325],
        [480, 325],
        [600, 325],
        [720, 325],
        [720, 175],
        [600, 175],
        [480, 175],
        [360, 175],
        [240, 175],
    ],
    dtype=float,
)

polylines_width_test = np.array(
    [
        [30, 500],
        [30, 375],
        [30, 250],
        [30, 125],
        [30, 4],
        [930, 4],
        [930, 125],
        [930, 250],
        [930, 375],
        [930, 500],
        [255, 350],
        [255, 300],
        [255, 250],
        [255, 200],
        [255, 150],
        [705, 150],
        [705, 200],
        [705, 250],
        [705, 300],
        [705, 350],
        [0, 400],
        [480, 400],
        [960, 400],
        [960, 100],
        [480, 100],
        [0, 100],
        [0, 325],
        [180, 325],
        [480, 325],
        [780, 325],
        [960, 325],
        [960, 175],
        [780, 175],
        [480, 175],
        [180, 175],
        [0, 175],
    ],
    dtype=float,
)

polylines_width_test_maintained_vertices = np.array(
    [
        [30, 500],
        [30, 375],
        [30, 250],
        [30, 125],
        [30, 4],
        [930, 500],
        [930, 375],
        [930, 250],
        [930, 125],
        [930, 4],
        [255, 350],
        [255, 300],
        [255, 250],
        [255, 200],
        [255, 150],
        [705, 350],
        [705, 300],
        [705, 250],
        [705, 200],
        [705, 150],
        [0, 400],
        [480, 400],
        [960, 400],
        [960, 400],
        [960, 400],
        [960, 100],
        [480, 100],
        [0, 100],
        [0, 100],
        [0, 100],
        [0, 325],
        [180, 325],
        [480, 325],
        [780, 325],
        [960, 325],
        [960, 175],
        [780, 175],
        [480, 175],
        [180, 175],
        [0, 175],
    ],
    dtype=float,
)

polylines_height_test = np.array(
    [
        [300, 504],
        [300, 250],
        [300, 0],
        [660, 0],
        [660, 250],
        [660, 504],
        [390, 500],
        [390, 375],
        [390, 250],
        [390, 125],
        [390, 0],
        [570, 0],
        [570, 125],
        [570, 250],
        [570, 375],
        [570, 500],
        [240, 437.5],
        [360, 437.5],
        [480, 437.5],
        [600, 437.5],
        [720, 437.5],
        [720, 62.5],
        [600, 62.5],
        [480, 62.5],
        [360, 62.5],
        [240, 62.5],
    ],
    dtype=float,
)

polylines_height_test_maintained_vertices = np.array(
    [
        [300, 504],
        [300, 250],
        [300, 0],
        [300, 0],
        [300, 0],
        [660, 504],
        [660, 504],
        [660, 504],
        [660, 250],
        [660, 0],
        [390, 500],
        [390, 375],
        [390, 250],
        [390, 125],
        [390, 0],
        [570, 500],
        [570, 375],
        [570, 250],
        [570, 125],
        [570, 0],
        [240, 437.5],
        [360, 437.5],
        [480, 437.5],
        [600, 437.5],
        [720, 437.5],
        [720, 62.5],
        [600, 62.5],
        [480, 62.5],
        [360, 62.5],
        [240, 62.5],
    ],
    dtype=float,
)

polylines_u_test = np.array(
    [
        [210, 504],
        [210, 437.5],
        [210, 250],
        [210, 62.5],
        [210, 0],
        [345, 400],
        [345, 325],
        [345, 250],
        [345, 175],
        [345, 100],
        [615, 100],
        [615, 175],
        [615, 250],
        [615, 325],
        [615, 400],
        [960, 475],
        [832.5, 475],
        [480, 475],
        [127.5, 475],
        [0, 475],
        [840, 362.5],
        [660, 362.5],
        [480, 362.5],
        [300, 362.5],
        [120, 362.5],
        [120, 137.5],
        [300, 137.5],
        [480, 137.5],
        [660, 137.5],
        [840, 137.5],
    ],
    dtype=float,
)

polylines_u_test_maintained_vertices = np.array(
    [
        [210, 504],
        [210, 437.5],
        [210, 250],
        [210, 62.5],
        [210, 0],
        [210, 0],
        [210, 0],
        [210, 0],
        [210, 0],
        [210, 0],
        [345, 400],
        [345, 325],
        [345, 250],
        [345, 175],
        [345, 100],
        [615, 400],
        [615, 325],
        [615, 250],
        [615, 175],
        [615, 100],
        [960, 475],
        [832.5, 475],
        [480, 475],
        [127.5, 475],
        [0, 475],
        [0, 475],
        [0, 475],
        [0, 475],
        [0, 475],
        [0, 475],
        [840, 362.5],
        [660, 362.5],
        [480, 362.5],
        [300, 362.5],
        [120, 362.5],
        [120, 137.5],
        [300, 137.5],
        [480, 137.5],
        [660, 137.5],
        [840, 137.5],
    ],
    dtype=float,
)

polylines_reordered = np.array(
    [
        [300, 500],
        [300, 375],
        [300, 250],
        [300, 125],
        [300, 4],
        [660, 500],
        [660, 375],
        [660, 250],
        [660, 125],
        [660, 4],
        [390, 350],
        [390, 300],
        [390, 250],
        [390, 200],
        [390, 150],
        [570, 350],
        [570, 300],
        [570, 250],
        [570, 200],
        [570, 150],
        [10, 400],
        [245, 400],
        [480, 400],
        [715, 400],
        [950, 400],
        [950, 100],
        [715, 100],
        [480, 100],
        [245, 100],
        [10, 100],
        [240, 325],
        [360, 325],
        [480, 325],
        [600, 325],
        [720, 325],
        [720, 175],
        [600, 175],
        [480, 175],
        [360, 175],
        [240, 175],
    ],
    dtype=float,
)


# Test possible configurations of the paths and image mask.
clipping_tests = [
    # all inside.
    (
        polylines,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline,
        0,
        0,
        1,
        1,
        False,
        polylines_all_inside,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline,
    ),
    # all outside.
    (
        polylines,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline,
        0,
        504,
        1,
        1,
        False,
        np.empty((0, 2)),
        [],
        [],
        [],
    ),
    # stradle width
    (
        polylines,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline,
        -720,
        0,
        2.5,
        1,
        False,
        polylines_width_test,
        [5, 5, 5, 5, 3, 3, 5, 5],
        class_ids_per_polyline,
        attributes_per_polyline,
    ),
    # stradle height
    (
        polylines,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline,
        0,
        -375,
        1,
        2.5,
        False,
        polylines_height_test,
        [3, 3, 5, 5, 5, 5],
        [0, 0, 1, 1, 3, 3],
        [-1, 1, -1, 1, -1, 1],
    ),
    # increase in number (u-shape)
    (
        polylines,
        np.array([10, 10, 10, 10], dtype=int),
        np.array([0, 1, 2, 3], dtype=int),
        np.array([1, 1, 1, 1], dtype=int),
        -240,
        -125,
        1.5,
        1.5,
        False,
        polylines_u_test,
        [5, 10, 5, 10],
        [0, 1, 2, 3],
        [1, 1, 1, 1],
    ),
    # all inside.
    (
        polylines,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline,
        0,
        0,
        1,
        1,
        True,
        polylines_reordered,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline,
    ),
    # all outside.
    (
        polylines,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline,
        0,
        504,
        1,
        1,
        True,
        np.empty((0, 2)),
        [],
        [],
        [],
    ),
    # stradle width
    (
        polylines,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline,
        -720,
        0,
        2.5,
        1,
        True,
        polylines_width_test_maintained_vertices,
        [5, 5, 5, 5, 5, 5, 5, 5],
        class_ids_per_polyline,
        attributes_per_polyline,
    ),
    # stradle height
    (
        polylines,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline,
        0,
        -375,
        1,
        2.5,
        True,
        polylines_height_test_maintained_vertices,
        [5, 5, 5, 5, 5, 5],
        [0, 0, 1, 1, 3, 3],
        [-1, 1, -1, 1, -1, 1],
    ),
    # increase in number (u-shape)
    (
        polylines,
        np.array([10, 10, 10, 10], dtype=int),
        np.array([0, 1, 2, 3], dtype=int),
        np.array([1, 1, 1, 1], dtype=int),
        -240,
        -125,
        1.5,
        1.5,
        True,
        polylines_u_test_maintained_vertices,
        [10, 10, 10, 10],
        [0, 1, 2, 3],
        [1, 1, 1, 1],
    ),
]


# Model input width and height is 960 x 504.
polylines = np.array(
    [
        [300, 500],
        [300, 375],
        [300, 250],
        [300, 125],
        [300, 4],
        [660, 4],
        [660, 125],
        [660, 250],
        [660, 375],
        [660, 500],
        [390, 350],
        [390, 300],
        [390, 250],
        [390, 200],
        [390, 150],
        [570, 150],
        [570, 200],
        [570, 250],
        [570, 300],
        [570, 350],
        [950, 400],
        [715, 400],
        [480, 400],
        [245, 400],
        [10, 400],
        [10, 100],
        [245, 100],
        [480, 100],
        [715, 100],
        [950, 100],
        [720, 325],
        [600, 325],
        [480, 325],
        [360, 325],
        [240, 325],
        [240, 175],
        [360, 175],
        [480, 175],
        [600, 175],
        [720, 175],
    ],
    dtype=float,
)
vertices_per_polyline = np.array([5, 5, 5, 5, 5, 5, 5, 5], dtype=int)
class_ids_per_polyline = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=int)
attributes_per_polyline = np.array([-1, 1, -1, 1, -1, 1, -1, 1], dtype=int)


@pytest.mark.parametrize(
    "polylines, vertices_per_polyline, class_ids_per_polyline, "
    "attributes_per_polyline, translate_x, translate_y, "
    "scale_x, scale_y, maintain_vertex_number, expected_polylines, "
    "expected_vertices, expected_classes, expected_attributes",
    clipping_tests,
)
def test_polyline_clipping(
    _clipper,
    polylines,
    vertices_per_polyline,
    class_ids_per_polyline,
    attributes_per_polyline,
    translate_x,
    translate_y,
    scale_x,
    scale_y,
    maintain_vertex_number,
    expected_polylines,
    expected_vertices,
    expected_classes,
    expected_attributes,
):
    """Test the polyline clipper."""
    # Adjust coordinates based on inputs.
    polylines_modified = polylines.copy()
    polylines_modified[:, 0] = (polylines[:, 0] * scale_x) + translate_x
    polylines_modified[:, 1] = (polylines[:, 1] * scale_y) + translate_y
    image_height = 504
    image_width = 960

    polygon_mask = tf.constant(
        [
            [0, 0],
            [0, image_height],
            [image_width, image_height],
            [image_width, 0],
            [0, 0],
        ],
        tf.float32,
    )

    sess = tf.compat.v1.Session()
    (
        clipped_polylines,
        _,
        clipped_vertices_per_polyline,
        clipped_class_ids_per_polyline,
        clipped_attributes_per_polyline,
    ) = sess.run(
        _clipper.clip(
            tf.constant(polylines_modified, dtype=tf.float32),
            tf.constant(vertices_per_polyline, dtype=tf.int32),
            tf.constant(class_ids_per_polyline, dtype=tf.int32),
            tf.constant(attributes_per_polyline, dtype=tf.int32),
            maintain_vertex_number=maintain_vertex_number,
            polygon_mask=polygon_mask,
        )
    )

    np.testing.assert_almost_equal(clipped_polylines, expected_polylines, 4)
    np.testing.assert_almost_equal(clipped_vertices_per_polyline, expected_vertices, 4)
    np.testing.assert_almost_equal(clipped_class_ids_per_polyline, expected_classes, 4)
    np.testing.assert_almost_equal(
        clipped_attributes_per_polyline, expected_attributes, 4
    )


unspliting_tests = [
    # no splits.
    (
        polylines,
        vertices_per_polyline,
        np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int),
        polylines,
        vertices_per_polyline,
        np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int),
    ),
    # two splits.
    (
        polylines,
        vertices_per_polyline,
        np.array([0, 0, 1, 2, 3, 3, 4, 5], dtype=int),
        np.take(
            polylines,
            np.hstack(
                (np.hstack((np.arange(0, 5), np.arange(10, 25))), np.arange(30, 40))
            ),
            axis=0,
        ),
        vertices_per_polyline[0:6],
        np.array([0, 1, 2, 3, 4, 5], dtype=int),
    ),
]


@pytest.mark.parametrize(
    "polylines, vertices_per_polyline, polyline_index_map, "
    "expected_polylines, expected_vertices, expected_indices",
    unspliting_tests,
)
def test_remove_split_polylines(
    _clipper,
    polylines,
    vertices_per_polyline,
    polyline_index_map,
    expected_polylines,
    expected_vertices,
    expected_indices,
):
    (
        clipped_polylines,
        clipped_vertices_per_polyline,
        clipped_polyline_index_map,
    ) = _clipper._remove_split_polylines(
        polylines, vertices_per_polyline, polyline_index_map
    )
    np.testing.assert_almost_equal(clipped_polylines, expected_polylines, 4)
    np.testing.assert_almost_equal(clipped_vertices_per_polyline, expected_vertices, 4)
    np.testing.assert_almost_equal(clipped_polyline_index_map, expected_indices, 4)


polylines_bottom_up = np.array(
    [
        [300, 500],
        [300, 375],
        [300, 250],
        [300, 125],
        [300, 4],
        [660, 500],
        [660, 375],
        [660, 250],
        [660, 125],
        [660, 4],
        [390, 350],
        [390, 300],
        [390, 250],
        [390, 200],
        [390, 150],
        [570, 350],
        [570, 300],
        [570, 250],
        [570, 200],
        [570, 150],
        [950, 400],
        [715, 400],
        [480, 400],
        [245, 400],
        [10, 400],
        [10, 100],
        [245, 100],
        [480, 100],
        [715, 100],
        [950, 100],
        [720, 325],
        [600, 325],
        [480, 325],
        [360, 325],
        [240, 325],
        [240, 175],
        [360, 175],
        [480, 175],
        [600, 175],
        [720, 175],
    ],
    dtype=float,
)


reordering_tests = [(polylines, polylines_bottom_up)]


@pytest.mark.parametrize("polylines, expected_polylines", reordering_tests)
def test_enforce_bottom_up_vertex_order(_clipper, polylines, expected_polylines):
    sess = tf.compat.v1.Session()
    bottom_up_ordered = sess.run(_clipper._enforce_bottom_up_vertex_order(polylines))
    np.testing.assert_equal(expected_polylines, bottom_up_ordered)


polylines_shortened = np.array(
    [
        [300, 500],
        [300, 250],
        [300, 4],
        [660, 4],
        [660, 125],
        [660, 250],
        [660, 375],
        [660, 500],
        [390, 350],
        [390, 300],
        [390, 250],
        [390, 200],
        [390, 150],
        [950, 150],
        [570, 200],
        [950, 400],
        [715, 400],
        [480, 400],
        [245, 400],
        [10, 400],
        [10, 100],
        [245, 100],
        [720, 325],
        [600, 325],
        [480, 325],
        [360, 325],
        [240, 325],
        [240, 175],
        [360, 175],
        [480, 175],
        [600, 175],
        [720, 175],
    ],
    dtype=float,
)

polylines_resampled = np.array(
    [
        [300, 500],
        [300, 250],
        [300, 4],
        [300, 4],
        [300, 4],
        [660, 4],
        [660, 125],
        [660, 250],
        [660, 375],
        [660, 500],
        [390, 350],
        [390, 300],
        [390, 250],
        [390, 200],
        [390, 150],
        [950, 150],
        [570, 200],
        [570, 200],
        [570, 200],
        [570, 200],
        [950, 400],
        [715, 400],
        [480, 400],
        [245, 400],
        [10, 400],
        [10, 100],
        [245, 100],
        [245, 100],
        [245, 100],
        [245, 100],
        [720, 325],
        [600, 325],
        [480, 325],
        [360, 325],
        [240, 325],
        [240, 175],
        [360, 175],
        [480, 175],
        [600, 175],
        [720, 175],
    ],
    dtype=float,
)


resampling_tests = [
    (
        polylines_shortened,
        np.array([3, 5, 5, 2, 5, 2, 5, 5], dtype=int),
        polylines_resampled,
        vertices_per_polyline,
        np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int),
    ),
    (
        polylines,
        vertices_per_polyline,
        polylines,
        vertices_per_polyline,
        np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int),
    ),
]


@pytest.mark.parametrize(
    "polylines, vertices, expected_polylines, expected_vertices, index_map",
    resampling_tests,
)
def test_resample_shortened_polylines(
    _clipper, polylines, vertices, expected_polylines, expected_vertices, index_map
):
    resampled_polylines, resampled_vertices = _clipper._resample_shortened_polylines(
        polylines, vertices, expected_vertices, index_map
    )
    np.testing.assert_almost_equal(resampled_polylines, expected_polylines)
    np.testing.assert_almost_equal(resampled_vertices, expected_vertices, 4)


attributes_per_polyline_with_path_attributes = np.array(
    [-1, 1, -1, 1, -1, 1, -1, 1, 2, 2, 3, 3, 2, 2, 3, 3], dtype=int
)

clipping_tests_with_path_attributes = [
    # all inside.
    (
        polylines,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline_with_path_attributes,
        0,
        0,
        1,
        1,
        False,
        polylines_all_inside,
        [0, 1, 2, 3, 4, 5, 6, 7],
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline_with_path_attributes,
    ),
    # all outside.
    (
        polylines,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline_with_path_attributes,
        0,
        504,
        1,
        1,
        False,
        np.empty((0, 2)),
        [],
        [],
        [],
        [],
    ),
    # straddle width.
    (
        polylines,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline_with_path_attributes,
        -720,
        0,
        2.5,
        1,
        False,
        polylines_width_test,
        [0, 1, 2, 3, 4, 5, 6, 7],
        [5, 5, 5, 5, 3, 3, 5, 5],
        class_ids_per_polyline,
        attributes_per_polyline_with_path_attributes,
    ),
    # straddle height.
    (
        polylines,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline_with_path_attributes,
        0,
        -375,
        1,
        2.5,
        False,
        polylines_height_test,
        [0, 1, 2, 3, 6, 7],
        [3, 3, 5, 5, 5, 5],
        [0, 0, 1, 1, 3, 3],
        [-1, 1, -1, 1, -1, 1, 2, 2, 3, 3, 3, 3],
    ),
]


@pytest.mark.parametrize(
    "polylines, vertices_per_polyline, class_ids_per_polyline, "
    "attributes_per_polyline_with_path_attributes, translate_x, translate_y, "
    "scale_x, scale_y, maintain_vertex_number, expected_polylines, "
    "expected_index_map, expected_vertices, expected_classes, "
    "expected_attributes",
    clipping_tests_with_path_attributes,
)
def test_polyline_clipping_with_path_attributes(
    _clipper,
    polylines,
    vertices_per_polyline,
    class_ids_per_polyline,
    attributes_per_polyline_with_path_attributes,
    translate_x,
    translate_y,
    scale_x,
    scale_y,
    maintain_vertex_number,
    expected_polylines,
    expected_index_map,
    expected_vertices,
    expected_classes,
    expected_attributes,
):
    """Test the polyline clipper."""
    # Adjust coordinates based on inputs.
    polylines_modified = polylines.copy()
    polylines_modified[:, 0] = (polylines[:, 0] * scale_x) + translate_x
    polylines_modified[:, 1] = (polylines[:, 1] * scale_y) + translate_y
    image_height = 504
    image_width = 960
    polygon_mask = tf.constant(
        [
            [0, 0],
            [0, image_height],
            [image_width, image_height],
            [image_width, 0],
            [0, 0],
        ],
        tf.float32,
    )
    sess = tf.compat.v1.Session()
    (
        clipped_polylines,
        clipped_polyline_index_map,
        clipped_vertices_per_polyline,
        clipped_class_ids_per_polyline,
        clipped_attributes_per_polyline,
    ) = sess.run(
        _clipper.clip(
            tf.constant(polylines_modified, dtype=tf.float32),
            tf.constant(vertices_per_polyline, dtype=tf.int32),
            tf.constant(class_ids_per_polyline, dtype=tf.int32),
            tf.constant(attributes_per_polyline_with_path_attributes, dtype=tf.int32),
            maintain_vertex_number=maintain_vertex_number,
            polygon_mask=polygon_mask,
        )
    )

    np.testing.assert_almost_equal(clipped_polylines, expected_polylines, decimal=4)
    np.testing.assert_almost_equal(
        clipped_polyline_index_map, expected_index_map, decimal=4
    )
    np.testing.assert_almost_equal(
        clipped_vertices_per_polyline, expected_vertices, decimal=4
    )
    np.testing.assert_almost_equal(
        clipped_class_ids_per_polyline, expected_classes, decimal=4
    )
    np.testing.assert_almost_equal(
        clipped_attributes_per_polyline, expected_attributes, decimal=4
    )
