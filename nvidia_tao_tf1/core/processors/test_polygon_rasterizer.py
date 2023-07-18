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
"""Test the polygon rasterizer processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict, namedtuple
import errno
import os

import numpy as np
from PIL import Image
import pytest
import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import deserialize_tao_object
from nvidia_tao_tf1.core.processors import PolygonRasterizer
from nvidia_tao_tf1.core.processors import PolygonTransform
from nvidia_tao_tf1.core.processors import SparsePolygonRasterizer
from nvidia_tao_tf1.core.types import DataFormat
from nvidia_tao_tf1.core.utils import test_session

# Debug mode for saving test_polygon_rasterizer generated images to disk.
# This can be used for visually comparing test_polygon_rasterizer generated images to references,
# and regenerating the reference images in case the test or the rasterizer changes. In the latter
# case, run the test, visually check the images in test_polygon_rasterizer folder, and if ok, copy
# them to test_polygon_rasterizer_ref folder and commit to git.
debug_save_images = False


def _connected_components(edges):
    """
    Generate its connected components as sets of nodes.

    Time complexity is linear with respect to the number of edges.

    Adapted from: https://stackoverflow.com/questions/12321899
    """
    neighbors = defaultdict(set)
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    seen = set()

    def _component(
        node, neighbors=neighbors, seen=seen, see=seen.add
    ):  # pylint: disable=W0102
        unseen = set([node])
        next_unseen = unseen.pop
        while unseen:
            node = next_unseen()
            see(node)
            unseen |= neighbors[node] - seen
            yield node

    return (set(_component(node)) for node in neighbors if node not in seen)


def _matching_pixels(image, test):
    """Generate all pixel coordinates where pixel satisfies test."""
    width, height = image.size
    pixels = image.load()
    for x in range(width):
        for y in range(height):
            if test(pixels[x, y]):
                yield x, y


def _make_edges(coordinates):
    """Generate all pairs of neighboring pixel coordinates."""
    coordinates = set(coordinates)
    for x, y in coordinates:
        if (x - 1, y - 1) in coordinates:
            yield (x, y), (x - 1, y - 1)
        if (x, y - 1) in coordinates:
            yield (x, y), (x, y - 1)
        if (x + 1, y - 1) in coordinates:
            yield (x, y), (x + 1, y - 1)
        if (x - 1, y) in coordinates:
            yield (x, y), (x - 1, y)
        yield (x, y), (x, y)


def _boundingbox(coordinates):
    """Return the bounding box of all coordinates."""
    xs, ys = zip(*coordinates)
    return min(xs), min(ys), max(xs), max(ys)


def _is_black_enough(pixel):
    return pixel > 0


def disjoint_areas(image, test=_is_black_enough):
    """Return the bounding boxes of all non-consecutive areas who's pixels satisfy test."""
    for each in _connected_components(_make_edges(_matching_pixels(image, test))):
        yield _boundingbox(each)


Polygon = namedtuple(
    "Polygon",
    ["polygons", "vertices_per_polygon", "class_ids_per_polygon", "polygons_per_image"],
)


def _make_triangle(width, height, class_id=0):
    """Make a triangle."""
    polygons = tf.constant(
        [[width / 2.0, 0.0], [0.0, height], [width, height]], dtype=tf.float32
    )
    vertices_per_polygon = tf.constant([3], dtype=tf.int32)
    class_ids_per_polygon = tf.constant([class_id], dtype=tf.int32)
    polygons_per_image = tf.constant([1], dtype=tf.int32)
    return Polygon(
        polygons, vertices_per_polygon, class_ids_per_polygon, polygons_per_image
    )


def _make_line(length, thickness, offset_y, offset_x, angle=0.0):
    """Draw a line (thin rectangle) at an angle."""

    width = length
    height = thickness

    # Draw a line (rectangle) around the origin
    x = np.array(
        [-width / 2.0, width / 2.0, width / 2.0, -width / 2.0], dtype=np.float32
    )
    y = np.array(
        [-height / 2.0, -height / 2.0, height / 2.0, height / 2.0], dtype=np.float32
    )

    # Rotate the rectangle, and zip the polygons to a tensor
    xr = x * np.cos(angle) - y * np.sin(angle)
    yr = y * np.cos(angle) + x * np.sin(angle)
    polygons = tf.stack([xr, yr], axis=1)

    # Translate the polygon to the center of the image
    polygons = polygons + tf.constant([offset_y, offset_x], dtype=tf.float32)

    vertices_per_polygon = tf.constant([4], dtype=tf.int32)
    class_ids_per_polygon = tf.constant([0], dtype=tf.int32)
    return polygons, vertices_per_polygon, class_ids_per_polygon


@pytest.fixture
def _triangle_polygon():
    polygons = tf.constant(
        [
            [0.1, 0.3],
            [0.7, 0.3],
            [0.5, 0.4],
            [0.2, 0.3],
            [0.7, 0.3],
            [0.5, 0.7],
            [0.3, 0.3],
            [0.7, 0.3],
            [0.5, 0.7],
            [0.4, 0.3],
            [0.7, 0.3],
            [0.5, 0.7],
            [0.5, 0.3],
            [0.7, 0.3],
            [0.5, 0.7],
        ],
        dtype=tf.float32,
    )
    vertices_per_polygon = tf.constant([3, 3, 3, 3, 3], dtype=tf.int32)
    class_ids_per_polygon = tf.constant([1, 0, 0, 0, 1], dtype=tf.int32)
    polygons_per_image = tf.constant([5], dtype=tf.int32)
    return Polygon(
        polygons, vertices_per_polygon, class_ids_per_polygon, polygons_per_image
    )


@pytest.fixture
def _lanenet_polygon():
    """A set of polygons that spells out "L A N E N E T"."""
    polygons = (
        tf.constant(
            [
                [5.183, 9.818],
                [3.298, 92.005],
                [24.787, 92.005],
                [28.086, 77.019],
                [17.907, 69.29],
                [29.5, 92.005],
                [34.306, 9.818],
                [43.731, 92.005],
                [46.465, 92.005],
                [43.731, 10.667],
                [57.021, 83.428],
                [56.078, 11.892],
                [60.603, 12.174],
                [61.262, 93.041],
                [50.706, 87.669],
                [64.467, 86.256],
                [64.467, 93.607],
                [74.834, 93.607],
                [74.08, 85.689],
                [67.671, 82.392],
                [67.954, 60.996],
                [73.42, 59.865],
                [73.608, 53.268],
                [68.802, 51.477],
                [68.236, 27.443],
                [71.818, 24.521],
                [72.007, 11.986],
                [64.467, 11.986],
                [76.907, 93.041],
                [75.776, 11.986],
                [80.583, 11.986],
                [84.824, 85.973],
                [84.824, 11.986],
                [87.746, 11.986],
                [87.746, 93.041],
                [80.395, 89.365],
                [89.632, 93.607],
                [94.438, 93.607],
                [94.438, 85.407],
                [92.035, 84.842],
                [92.035, 53.174],
                [94.438, 52.514],
                [94.438, 44.597],
                [92.035, 41.769],
                [92.459, 20.28],
                [95.192, 20.28],
                [94.627, 12.457],
                [90.386, 11.986],
                [96.041, 11.986],
                [96.041, 15.567],
                [97.549, 15.567],
                [96.041, 93.607],
                [98.585, 93.607],
                [98.585, 15.85],
                [99.527, 15.85],
                [99.527, 11.986],
            ],
            dtype=tf.float32,
        )
        * 0.01
    )
    vertices_per_polygon = tf.constant([5, 3, 7, 13, 8, 12, 8], dtype=tf.int32)
    class_ids_per_polygon = tf.constant([0, 1, 2, 0, 1, 2, 0], dtype=tf.int32)
    polygons_per_image = tf.constant([7], dtype=tf.int32)

    return Polygon(
        polygons, vertices_per_polygon, class_ids_per_polygon, polygons_per_image
    )


def test_one_hot_binarize_fail():
    """Test incompatibility of non-one_hot and non-binary output."""
    with pytest.raises(ValueError):
        PolygonRasterizer(
            width=256,
            height=128,
            nclasses=3,
            one_hot=False,
            binarize=False,
            data_format=DataFormat.CHANNELS_FIRST,
        )


@pytest.mark.parametrize(
    "data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize("one_hot", [True, False])
def test_single_input(one_hot, data_format, width=60, height=30, nclasses=2):
    """Test the input of a single image, as opposed to a batched input."""
    sess = test_session()
    polygons = tf.zeros([0, 2], dtype=tf.float32)
    vertices_per_polygon = tf.constant([], dtype=tf.int32)
    class_ids_per_polygon = tf.constant([], dtype=tf.int32)
    op = PolygonRasterizer(
        width=width,
        height=height,
        nclasses=nclasses,
        one_hot=one_hot,
        data_format=data_format,
    )
    fetches = op(polygons, vertices_per_polygon, class_ids_per_polygon)
    sess.run(tf.compat.v1.global_variables_initializer())
    out = sess.run(fetches)
    if data_format == DataFormat.CHANNELS_LAST:
        out = np.transpose(out, [2, 0, 1])
    expected_shape = (nclasses + 1, height, width) if one_hot else (1, height, width)
    np.testing.assert_array_equal(out.shape, expected_shape)


dim_tests = [
    # One image
    (1, 1, 60, 30, True, (1, 2, 30, 60)),
    # Three images
    (3, 1, 60, 30, True, (3, 2, 30, 60)),
    # Three images, three classes
    (3, 3, 60, 30, True, (3, 4, 30, 60)),
    # Without one_hot
    # Three images, one class
    (3, 1, 60, 30, False, (3, 1, 30, 60)),
    # Three images, three classes
    (3, 3, 60, 30, False, (3, 1, 30, 60)),
]


@pytest.mark.parametrize(
    "nimages,nclasses,width,height,one_hot,expected_shape", dim_tests
)
def test_shapes_dims_no_polygons(
    nimages, nclasses, width, height, one_hot, expected_shape
):
    """
    Test for dimension inputs, but without any polygons. Checks if the shapes are as expected
    and the contents of the output tensor is entirely blank.
    """
    sess = test_session()
    polygons = tf.zeros([0, 2], dtype=tf.float32)
    vertices_per_polygon = tf.constant([], dtype=tf.int32)
    class_ids_per_polygon = tf.constant([], dtype=tf.int32)
    polygons_per_image = tf.zeros([nimages], dtype=tf.int32)
    op = PolygonRasterizer(
        width=width,
        height=height,
        nclasses=nclasses,
        one_hot=one_hot,
        data_format=DataFormat.CHANNELS_FIRST,
    )
    fetches = op(
        polygons, vertices_per_polygon, class_ids_per_polygon, polygons_per_image
    )

    sess.run(tf.compat.v1.global_variables_initializer())
    out = sess.run(fetches)

    # Check shape
    assert out.shape == expected_shape

    if one_hot:
        expected = np.concatenate(
            [
                np.ones([nimages, 1, height, width]),  # background
                np.zeros([nimages, nclasses, height, width]),
            ],  # class map
            axis=1,
        )
        np.testing.assert_array_equal(out, expected)
    else:
        np.testing.assert_array_equal(out, np.zeros([nimages, 1, height, width]))


dim_tests = [
    # One image
    (1, 1, 60, 30, True, True, (1, 2, 30, 60)),
    # Three images, exclude background.
    (3, 1, 60, 30, True, False, (3, 1, 30, 60)),
    # Without one_hot
    # Three images, one class, exclude background
    (3, 1, 60, 30, False, False, (3, 1, 30, 60)),
    # Three images, three classes
    (3, 3, 60, 30, False, True, (3, 1, 30, 60)),
]


@pytest.mark.parametrize(
    "nimages,nclasses,width,height,one_hot,include_background,expected_shape", dim_tests
)
def test_shapes_dims_include_background(
    nimages, nclasses, width, height, one_hot, include_background, expected_shape
):
    """
    Test for dimension inputs, but without any polygons. Checks if the shapes are as expected
    and the contents of the output tensor is entirely blank.
    """
    sess = test_session()
    polygons = tf.zeros([0, 2], dtype=tf.float32)
    vertices_per_polygon = tf.constant([], dtype=tf.int32)
    class_ids_per_polygon = tf.constant([], dtype=tf.int32)
    polygons_per_image = tf.zeros([nimages], dtype=tf.int32)
    op = PolygonRasterizer(
        width=width,
        height=height,
        nclasses=nclasses,
        one_hot=one_hot,
        data_format=DataFormat.CHANNELS_FIRST,
        include_background=include_background,
    )
    fetches = op(
        polygons, vertices_per_polygon, class_ids_per_polygon, polygons_per_image
    )

    sess.run(tf.compat.v1.global_variables_initializer())
    out = sess.run(fetches)

    # Check shape
    assert out.shape == expected_shape

    if one_hot:
        if include_background:
            expected = np.concatenate(
                [
                    np.ones([nimages, 1, height, width]),  # background
                    np.zeros([nimages, nclasses, height, width]),
                ],  # class map
                axis=1,
            )
        else:
            expected = np.zeros([nimages, nclasses, height, width])
        np.testing.assert_array_equal(out, expected)
    else:
        np.testing.assert_array_equal(out, np.zeros([nimages, 1, height, width]))


@pytest.mark.parametrize("class_id,nclasses", [(0, -1), (1, 1), (0, 0)])
def test_class_id_fails(class_id, nclasses, width=64, height=32):
    """
    Test fail cases where the class_id is out of bounds with respect to the total amount of
    classes.
    """
    op = PolygonRasterizer(
        width=width,
        height=height,
        nclasses=nclasses,
        one_hot=True,
        data_format=DataFormat.CHANNELS_FIRST,
    )
    inputs = _make_triangle(width=width, height=height, class_id=class_id)
    fetches = op(*inputs)
    sess = test_session()
    sess.run(tf.compat.v1.global_variables_initializer())
    with pytest.raises(Exception):
        sess.run(fetches)


@pytest.mark.parametrize(
    "data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize(
    "one_hot,binarize", [(True, True), (True, False), (False, True)]
)
def test_triangle_coverage(one_hot, binarize, data_format):
    """Draw a triangle in the center of the image, and test if it covers half of the image.

    When `one_hot` is `True`, we also check that the background map inversely contains the inverse
    coverage.
    """
    height, width, nclasses = 128, 256, 1
    op = PolygonRasterizer(
        width=width,
        height=height,
        nclasses=nclasses,
        one_hot=one_hot,
        binarize=binarize,
        data_format=data_format,
    )

    inputs = _make_triangle(width=width, height=height, class_id=0)
    fetches = op(*inputs)

    sess = test_session()
    sess.run(tf.compat.v1.global_variables_initializer())
    out = sess.run(fetches)

    # Cancel the transpose from output for 'channels_last' format to continue testing.
    if data_format == DataFormat.CHANNELS_LAST:
        out = np.transpose(out, [0, 3, 1, 2])

    mean_per_class_map = np.mean(out, axis=(0, 2, 3))

    expected_mean, rtol = 0.5, 1e-2
    if one_hot:
        np.testing.assert_allclose(
            mean_per_class_map[0],
            expected_mean,
            rtol=rtol,
            err_msg="unexpected background coverage.",
        )
        np.testing.assert_allclose(
            mean_per_class_map[1],
            expected_mean,
            rtol=rtol,
            err_msg="unexpected foreground coverage.",
        )
        if binarize:
            # Test of an exact result where each pixel's sum over the class map results in 1
            # Combine the background and foreground, and check each pixel adds up to exactly 1
            combined_coverage = np.sum(out, axis=1, keepdims=True)
            np.testing.assert_array_equal(
                combined_coverage,
                np.ones([1, 1, height, width]),
                err_msg="not every pixel sums up to a value of exactly 1.",
            )
    else:
        np.testing.assert_allclose(
            mean_per_class_map,
            expected_mean,
            rtol=rtol,
            err_msg="unexpected coverage area while drawing a triangle.",
        )


@pytest.mark.parametrize(
    "data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize("one_hot", (True, False))
def test_draw_outside_canvas(one_hot, data_format, width=64, height=32, nclasses=1):
    """Test drawing outside of the canvas visible range, and assert the output is blank."""
    polygons = tf.constant(
        [
            [0.0, 0.0],
            [-1.0, 0.0],
            [0.0, -1.0],  # < (0, 0)
            [width, height],
            [2 * width, height],
            [width, 2 * height],
        ],  # > (width, height)
        dtype=tf.float32,
    )
    vertices_per_polygon = tf.constant([3, 3], dtype=tf.int32)
    class_ids_per_polygon = tf.constant([0, 0], dtype=tf.int32)
    polygons_per_image = tf.constant([2], dtype=tf.int32)

    op = PolygonRasterizer(
        width=width,
        height=height,
        nclasses=nclasses,
        one_hot=one_hot,
        data_format=data_format,
    )
    fetches = op(
        polygons, vertices_per_polygon, class_ids_per_polygon, polygons_per_image
    )

    sess = test_session()
    sess.run(tf.compat.v1.global_variables_initializer())
    out = sess.run(fetches)

    # Cancel the transpose from output for 'channels_last' format to continue testing.
    if data_format == DataFormat.CHANNELS_LAST:
        out = np.transpose(out, [0, 3, 1, 2])

    if one_hot:
        expected = np.concatenate(
            [
                np.ones([1, 1, height, width]),  # background
                np.zeros([1, nclasses, height, width]),
            ],  # class map
            axis=1,
        )
        np.testing.assert_array_equal(out, expected)
    else:
        np.testing.assert_array_equal(out, np.zeros([1, nclasses, height, width]))


@pytest.mark.parametrize(
    "data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize(
    "one_hot, binarize", [(True, True), (True, False), (False, True)]
)
def test_draw_repeated_lanenet_text(
    one_hot,
    binarize,
    data_format,
    _lanenet_polygon,
    nclasses=3,
    width=128,
    height=64,
    nrepeats=5,
):
    """
    Test a complicated set of polygons that spells out "L A N E N E T" in for a few option
    combinations and loop those a few times, and compare the output to the previous one.
    This is run in a new session each time.
    Also tests the binarize setting for actually result in binary outputs, and visa versa.
    """
    polygons_abs = tf.matmul(
        _lanenet_polygon.polygons,
        tf.constant([[width, 0], [0, height]], dtype=tf.float32),
    )
    vertices_per_polygon = _lanenet_polygon.vertices_per_polygon
    class_ids_per_polygon = _lanenet_polygon.class_ids_per_polygon
    polygons_per_image = _lanenet_polygon.polygons_per_image

    op = PolygonRasterizer(
        width=width,
        height=height,
        nclasses=nclasses,
        one_hot=one_hot,
        binarize=binarize,
        data_format=data_format,
    )
    fetches = op(
        polygons_abs, vertices_per_polygon, class_ids_per_polygon, polygons_per_image
    )
    last_output = None
    for _ in range(nrepeats):
        sess = test_session()
        sess.run(tf.compat.v1.global_variables_initializer())
        out = sess.run(fetches)
        # Cancel the transpose from output for 'channels_last' format to continue testing.
        if data_format == DataFormat.CHANNELS_LAST:
            out = np.transpose(out, [0, 3, 1, 2])

        if one_hot:
            if binarize:
                np.testing.assert_array_equal(
                    np.unique(out),
                    [0.0, 1.0],
                    "binarize and one_hot did not result in exclusively 0 and 1 values.",
                )
            else:
                assert (
                    len(np.unique(out)) > 2
                ), "one_hot and non-binarized output resulted in too few unique values."
            # Check each class contains something
            np.testing.assert_array_equal(
                np.amax(out, axis=(0, 2, 3)),
                np.ones(1 + nclasses),
                "not every class map contains output while using one_hot.",
            )
        else:  # not one hot
            expected_unique_range = np.arange(nclasses + 1)
            np.testing.assert_array_equal(
                np.unique(out),
                expected_unique_range,
                "binarize and not one_hot did not result in exclusively values: %s."
                % expected_unique_range,
            )

        if last_output is not None:
            np.testing.assert_array_equal(
                out,
                last_output,
                "drawing of the same polygon set repeatedly results in a non-deterministic "
                "rasterized map",
            )
        last_output = out


@pytest.mark.parametrize(
    "data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize("device", ["/gpu:0", "/cpu"])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_draw_batched_copies_are_identical(
    batch_size, device, data_format, _lanenet_polygon, width=128, height=64
):
    """Test drawing batched copies of the same polygon produce identical results"""
    with tf.device(device):
        op = PolygonRasterizer(
            width=width,
            height=height,
            nclasses=3,
            one_hot=True,
            binarize=True,
            verbose=False,
            data_format=data_format,
        )

        polygons_abs = tf.matmul(
            _lanenet_polygon.polygons,
            tf.constant([[width, 0], [0, height]], dtype=tf.float32),
        )
        vertices_per_polygon = _lanenet_polygon.vertices_per_polygon
        class_ids_per_polygon = _lanenet_polygon.class_ids_per_polygon
        polygons_per_image = _lanenet_polygon.polygons_per_image

        single_rasterized = op(
            polygons_abs,
            vertices_per_polygon,
            class_ids_per_polygon,
            polygons_per_image,
        )

        polygons_abs = tf.tile(
            tf.matmul(
                _lanenet_polygon.polygons,
                tf.constant([[width, 0], [0, height]], dtype=tf.float32),
            ),
            [batch_size, 1],
        )
        vertices_per_polygon = tf.tile(
            _lanenet_polygon.vertices_per_polygon, [batch_size]
        )
        class_ids_per_polygon = tf.tile(
            _lanenet_polygon.class_ids_per_polygon, [batch_size]
        )
        polygons_per_image = tf.tile(_lanenet_polygon.polygons_per_image, [batch_size])

        batch_rasterized = op(
            polygons_abs,
            vertices_per_polygon,
            class_ids_per_polygon,
            polygons_per_image,
        )

        sess = test_session()
        sess.run(tf.compat.v1.global_variables_initializer())
        single_out = sess.run(single_rasterized)
        batch_out = sess.run(batch_rasterized)

        # Cancel the transpose from output for 'channels_last' format to continue testing.
        if data_format == DataFormat.CHANNELS_LAST:
            single_out = np.transpose(single_out, [0, 3, 1, 2])
            batch_out = np.transpose(batch_out, [0, 3, 1, 2])

        for raster_index in range(batch_size):
            np.testing.assert_array_equal(single_out[0], batch_out[raster_index])


@pytest.mark.parametrize(
    "data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize("thickness,expect_fragmented", [(0.01, True), (0.26, False)])
def test_line_non_fragmentation(thickness, expect_fragmented, data_format):
    """
    Rasterize subpixel-thick lines at different angles, and test the lines are continuous

    Note that polygon thicknesses above 0.25 should not fragment because of the subsampler in the
    rasterizer.

    Args:
        thickness (float): the absolute thickness of the line we will draw.
        expect_fragmented (bool): Whether or not we expect fragmentation
    """
    width, height = 256, 128
    op = PolygonRasterizer(
        width=width,
        height=height,
        nclasses=1,
        one_hot=False,
        binarize=True,
        data_format=data_format,
    )

    ps, vs, cs = [], [], []
    nlines = 16
    angles = np.linspace(0.0, 3.14, nlines, endpoint=False)
    for angle in angles:
        p, v, c = _make_line(
            length=width,
            thickness=thickness,
            offset_y=width / 2,
            offset_x=height / 2,
            angle=angle,
        )
        ps.append(p)
        vs.append(v)
        cs.append(c)

    ps = tf.concat(ps, axis=0)
    vs = tf.concat(vs, axis=0)
    cs = tf.concat(cs, axis=0)
    fetches = op(ps, vs, cs)
    sess = test_session()
    sess.run(tf.compat.v1.global_variables_initializer())
    out = sess.run(fetches)

    # Cancel the transpose from output for 'channels_last' format to continue testing.
    if data_format == DataFormat.CHANNELS_LAST:
        out = np.transpose(out, [2, 0, 1])

    image = Image.fromarray(np.uint8(out[0] * 255))

    n_fragments = len(list(disjoint_areas(image)))

    is_fragmented = n_fragments > 1

    assert is_fragmented == expect_fragmented


def test_identity_polygon_transform():
    """Test an identity polygon transform leaves the polygon unchanged."""
    polygons_in = _make_triangle(width=64, height=32, class_id=0)[0]

    # Define a spatial transformation matrix.
    stm = tf.eye(3, dtype=tf.float32)

    # Set up the polygon transformer, and transform the polygons.
    polygon_transform_op = PolygonTransform()
    polygons_out = polygon_transform_op(polygons_in, stm)

    # Set up the rasterizer and run the graph.
    sess = test_session()
    sess.run(tf.compat.v1.global_variables_initializer())
    p_in_np, p_out_np = sess.run([polygons_in, polygons_out])
    np.testing.assert_array_equal(p_out_np, p_in_np)


def _draw_dense(
    num_classes,
    one_hot,
    binarize,
    num_samples,
    data_format,
    polygons_per_image,
    vertices_per_polygon,
    class_ids_per_polygon,
    vertices,
):
    """Execute PolygonRasterizer op."""
    sess = test_session()
    op = PolygonRasterizer(
        width=100,
        height=90,
        nclasses=num_classes,
        one_hot=one_hot,
        binarize=binarize,
        verbose=True,
        data_format=data_format,
        num_samples=num_samples,
    )
    fetches = op(
        polygon_vertices=tf.constant(vertices, dtype=tf.float32),
        vertex_counts_per_polygon=vertices_per_polygon,
        class_ids_per_polygon=class_ids_per_polygon,
        polygons_per_image=polygons_per_image,
    )
    sess.run(tf.compat.v1.global_variables_initializer())
    output = sess.run(fetches)

    # Check shape inference.
    assert fetches.shape == output.shape

    return output


def _draw_sparse(
    num_classes,
    one_hot,
    binarize,
    num_samples,
    data_format,
    polygons,
    class_ids_per_polygon,
):
    """Execute SparsePolygonRasterizer op."""
    sess = test_session()
    op = SparsePolygonRasterizer(
        width=100,
        height=90,
        nclasses=num_classes,
        one_hot=one_hot,
        binarize=binarize,
        verbose=True,
        data_format=data_format,
        num_samples=num_samples,
    )
    fetches = op(polygons=polygons, class_ids_per_polygon=class_ids_per_polygon)

    sess.run(tf.compat.v1.global_variables_initializer())
    output = sess.run(fetches)

    # Check shape inference.
    assert fetches.shape == output.shape

    return output


def _run_sparse_rasterizer_error_check(data):
    """Run sparse rasterizer error check."""
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        polygons = tf.SparseTensor(
            values=tf.constant(data["vertices"], dtype=tf.float32),
            indices=tf.constant(data["indices"], dtype=tf.int64),
            dense_shape=data["dense_shape"],
        )

        class_ids_per_polygon = tf.SparseTensor(
            values=tf.constant(data["class_ids_per_polygon"], dtype=tf.int64),
            indices=tf.constant(data["class_indices"], dtype=tf.int64),
            dense_shape=data["dense_shape"],
        )

        _draw_sparse(
            num_classes=2,
            one_hot=False,
            binarize=True,
            num_samples=1,
            data_format=DataFormat.CHANNELS_FIRST,
            polygons=polygons,
            class_ids_per_polygon=class_ids_per_polygon,
        )


@pytest.fixture()
def sparse_rasterizer_error_check_data():
    """Return data dict for sparse rasterizer error checks."""
    vertices = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    indices = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 2, 0],
        [0, 0, 2, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 1, 2, 0],
        [0, 1, 2, 1],
    ]
    class_ids_per_polygon = [0, 1]
    class_indices = [[0, 0, 0, 0], [0, 1, 0, 0]]
    dense_shape = [1, 2, 6, 2]
    class_ids_dense_shape = [1, 2, 6]
    return {
        "vertices": vertices,
        "indices": indices,
        "dense_shape": dense_shape,
        "class_ids_per_polygon": class_ids_per_polygon,
        "class_indices": class_indices,
        "class_ids_dense_shape": class_ids_dense_shape,
    }


@pytest.mark.parametrize(
    "dense_shape",
    [
        [[1], [2]],  # Dense shape is not 1D.
        [1, 2],  # Dense shape does not have 3 or 4 elements.
        [0, 2, 6, 2],  # Dense shape batch size is <= 0.
        [1, -1, 6, 2],  # Dense shape polygons dimension size < 0.
        [1, 2, -1, 2],  # Dense shape vertices dimension size < 0.
        [1, 2, 6, 1],  # Dense shape coordinates dimension size != 2.
    ],
)
def test_sparse_rasterizer_error_check_dense_shape(
    sparse_rasterizer_error_check_data, dense_shape
):
    """Test sparse rasterizer dense shape error checks."""
    sparse_rasterizer_error_check_data["dense_shape"] = dense_shape
    sparse_rasterizer_error_check_data["class_ids_dense_shape"] = dense_shape[:-1]
    _run_sparse_rasterizer_error_check(sparse_rasterizer_error_check_data)


@pytest.mark.parametrize(
    "vertices",
    [
        [[0.0], [0.0]],  # Vertices is not 1D.
        [0.0, 0.0, 0.0],  # Vertices size is not even.
        [0.0, 0.0],  # Vertices size does not match indices size.
    ],
)
def test_sparse_rasterizer_error_check_vertices(
    sparse_rasterizer_error_check_data, vertices
):
    """Test sparse rasterizer vertices array error checks."""
    sparse_rasterizer_error_check_data["vertices"] = vertices
    sparse_rasterizer_error_check_data["indices"] = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]
    _run_sparse_rasterizer_error_check(sparse_rasterizer_error_check_data)


@pytest.mark.parametrize(
    "indices",
    [
        [0, 0, 0, 0],  # Indices is not 2D.
        [[0, 0, 0, 0]],  # Indices dim0 size != values size.
        [[0, 0, 0], [0, 0, 1]],  # Indices dim1 size != dense_shape num elements.
        [[2, 0, 0, 0], [2, 0, 0, 1]],  # Image index >= batch_size or < 0.
        [
            [0, 2, 0, 0],
            [0, 2, 0, 1],
        ],  # Polygon index < 0 or >= dense_shape polygons dimension size.
        [
            [0, 0, 2, 0],
            [0, 0, 2, 1],
        ],  # Vertex index < 0 or >= dense_shape vertices dimension size.
        [[0, 0, 0, 2], [0, 0, 0, 3]],  # Coordinate index not 0 or 1.
        [[1, 0, 0, 0], [0, 0, 0, 1]],  # Image index not in order.
        [[0, 1, 0, 0], [0, 0, 0, 1]],  # Polygon index not in order.
        [[0, 0, 1, 0], [0, 0, 0, 1]],  # Vertex index not in order.
        [[0, 0, 0, 1], [0, 0, 0, 0]],  # Coordate index not in order.
    ],
)
def test_sparse_rasterizer_error_check_indices(
    sparse_rasterizer_error_check_data, indices
):
    """Test sparse rasterizer indices error checks."""
    sparse_rasterizer_error_check_data["dense_shape"] = [2, 2, 2, 2]
    sparse_rasterizer_error_check_data["vertices"] = [0.0, 0.0]
    sparse_rasterizer_error_check_data["indices"] = indices
    _run_sparse_rasterizer_error_check(sparse_rasterizer_error_check_data)


@pytest.mark.parametrize(
    "class_ids",
    [
        [0],  # Class IDs size does not match the number of polygons.
        [2, 2],  # Class ID >= num classes.
    ],
)
def test_sparse_rasterizer_error_check_class_ids(
    sparse_rasterizer_error_check_data, class_ids
):
    """Test sparse rasterizer class ids error checks."""
    sparse_rasterizer_error_check_data["class_ids_per_polygon"] = class_ids
    _run_sparse_rasterizer_error_check(sparse_rasterizer_error_check_data)


@pytest.mark.parametrize(
    "values, indices, dense_shape, class_ids_per_polygon, class_indices, class_dense_shape, "
    "expected_shape",
    [
        # No vertices, single image.
        (None, None, [0, 0, 2], None, None, [0, 0], [1, 90, 100]),
        # No vertices, batch size 1.
        (None, None, [1, 0, 0, 2], None, None, [1, 0, 0], [1, 1, 90, 100]),
        # 1 vertex, batch size 1.
        (
            [0.0, 0.0],
            [[0, 0, 0, 0], [0, 0, 0, 1]],
            [1, 1, 1, 2],
            [0],
            [[0, 0, 0]],
            [1, 1, 1],
            [1, 1, 90, 100],
        ),
        # 2 vertices, batch size 1.
        (
            [0.0, 0.0, 1.0, 1.0],
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]],
            [1, 1, 2, 2],
            [0],
            [[0, 0, 0]],
            [1, 1, 2],
            [1, 1, 90, 100],
        ),
        # 3 images, 1 polygon per image, 1 vertex per polygon. batch size 3.
        (
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 1],
                [2, 0, 0, 0],
                [2, 0, 0, 1],
            ],
            [3, 1, 1, 2],
            [0, 0, 0],
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            [3, 1, 1],
            [3, 1, 90, 100],
        ),
        # 2 polygons, batch size 3, the first image is empty.
        (
            [1.0, 1.0, 1.0, 1.0],
            [[1, 0, 0, 0], [1, 0, 0, 1], [2, 0, 0, 0], [2, 0, 0, 1]],
            [3, 1, 1, 2],
            [0, 0],
            [[1, 0, 0], [2, 0, 0]],
            [3, 1, 1],
            [3, 1, 90, 100],
        ),
        # 2 polygons, batch size 3, the second image is empty.
        (
            [1.0, 1.0, 1.0, 1.0],
            [[0, 0, 0, 0], [0, 0, 0, 1], [2, 0, 0, 0], [2, 0, 0, 1]],
            [3, 1, 1, 2],
            [0, 0],
            [[0, 0, 0], [2, 0, 0]],
            [3, 1, 1],
            [3, 1, 90, 100],
        ),
        # 2 polygons, batch size 3, the third image is empty.
        (
            [1.0, 1.0, 1.0, 1.0],
            [[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 1]],
            [3, 1, 1, 2],
            [0, 0],
            [[0, 0, 0], [1, 0, 0]],
            [3, 1, 1],
            [3, 1, 90, 100],
        ),
    ],
)
def test_sparse_polygon_rasterizer_degenerate_input(
    values,
    indices,
    dense_shape,
    class_ids_per_polygon,
    class_indices,
    class_dense_shape,
    expected_shape,
):
    """Test valid but degenerate input."""
    if values is None:
        values = tf.zeros(shape=[0], dtype=tf.float32)
    if indices is None:
        indices = tf.zeros(shape=[0, len(dense_shape)], dtype=tf.int64)
    if class_ids_per_polygon is None:
        class_ids_per_polygon = tf.zeros(shape=[0], dtype=tf.int64)
    if class_indices is None:
        class_indices = tf.zeros(shape=[0, len(class_dense_shape)], dtype=tf.int64)

    polygons = tf.SparseTensor(values=values, indices=indices, dense_shape=dense_shape)

    class_ids_per_polygon = tf.SparseTensor(
        values=class_ids_per_polygon,
        indices=class_indices,
        dense_shape=class_dense_shape,
    )
    output = _draw_sparse(
        num_classes=1,
        one_hot=False,
        binarize=True,
        num_samples=1,
        data_format=DataFormat.CHANNELS_FIRST,
        polygons=polygons,
        class_ids_per_polygon=class_ids_per_polygon,
    )
    np.testing.assert_array_equal(
        output, np.zeros(expected_shape), err_msg="output is not all zeros."
    )


def _check_sparse_shape_inference(
    dense_shape, height, width, one_hot, data_format, expected_shape
):
    """Run shape inference test for sparse rasterizer."""
    vertices = tf.compat.v1.placeholder(dtype=tf.float32)
    class_ids_per_polygon = tf.SparseTensor(
        values=tf.compat.v1.placeholder(dtype=tf.int64),
        indices=tf.compat.v1.placeholder(dtype=tf.int64),
        dense_shape=dense_shape[:-1],
    )
    indices = tf.compat.v1.placeholder(dtype=tf.int64)
    polygons = tf.SparseTensor(
        values=vertices, indices=indices, dense_shape=dense_shape
    )

    op = SparsePolygonRasterizer(
        width=width,
        height=height,
        nclasses=1,
        one_hot=one_hot,
        binarize=True,
        verbose=True,
        data_format=data_format,
    )
    output = op(polygons=polygons, class_ids_per_polygon=class_ids_per_polygon)

    assert expected_shape == output.shape.as_list()


@pytest.mark.parametrize(
    "dense_shape,single_image,expected_batch_size",
    [
        ([1, 1, 2], True, None),
        ([1, 1, 1, 2], False, 1),
        ([-1, 1, 1, 2], False, None),
        # Shape is unknown, the op assumes no batch dimension.
        (tf.compat.v1.placeholder(dtype=tf.int64), True, None),
        (tf.compat.v1.placeholder(shape=[3], dtype=tf.int64), True, None),
        (tf.compat.v1.placeholder(shape=[4], dtype=tf.int64), False, None),
        (tf.constant([-1, 2, 6, 2], dtype=tf.int64), False, None),
        (tf.Variable([-1, 2, 6, 2], dtype=tf.int64), False, None),
        # Variable value is unavailable for shape inference.
        (tf.Variable([1, 2, 6, 2], dtype=tf.int64), False, None),
    ],
)
def test_sparse_polygon_rasterizer_shape_inference_batch_size(
    dense_shape, single_image, expected_batch_size
):
    """Test different ways of specifying batch size."""
    if single_image:
        expected_shape = [1, 90, 100]
    else:
        expected_shape = [expected_batch_size, 1, 90, 100]
    _check_sparse_shape_inference(
        dense_shape,
        height=90,
        width=100,
        one_hot=False,
        data_format=DataFormat.CHANNELS_FIRST,
        expected_shape=expected_shape,
    )


@pytest.mark.parametrize("one_hot,expected_nclasses", [(False, 1), (True, 2)])
def test_sparse_polygon_rasterizer_shape_inference_classes(one_hot, expected_nclasses):
    """Test shape inference for the number of classes."""
    dense_shape = [2, 1, 1, 2]
    expected_shape = [2, expected_nclasses, 90, 100]
    _check_sparse_shape_inference(
        dense_shape,
        height=90,
        width=100,
        one_hot=one_hot,
        data_format=DataFormat.CHANNELS_FIRST,
        expected_shape=expected_shape,
    )


@pytest.mark.parametrize(
    "width, height, expected_width, expected_height",
    [
        (100, 90, 100, 90),
        (
            tf.compat.v1.placeholder(dtype=tf.int32),
            tf.compat.v1.placeholder(dtype=tf.int32),
            None,
            None,
        ),
        (
            tf.Variable(100, dtype=tf.int32),
            tf.Variable(100, dtype=tf.int32),
            None,
            None,
        ),
    ],
)
def test_sparse_polygon_rasterizer_shape_inference_width_height(
    width, height, expected_width, expected_height
):
    """Test shape inference for the different ways of specifying width and height."""
    dense_shape = [1, 1, 1, 2]
    expected_shape = [1, 1, expected_height, expected_width]
    _check_sparse_shape_inference(
        dense_shape,
        height=height,
        width=width,
        one_hot=False,
        data_format=DataFormat.CHANNELS_FIRST,
        expected_shape=expected_shape,
    )


def _check_dense_shape_inference(
    vertices_per_polygon,
    polygons_per_image,
    height,
    width,
    one_hot,
    data_format,
    expected_shape,
):
    """Run shape inference test for dense rasterizer."""
    vertices = tf.compat.v1.placeholder(dtype=tf.float32)
    class_ids_per_polygon = tf.compat.v1.placeholder(dtype=tf.int64)
    op = PolygonRasterizer(
        width=width,
        height=height,
        nclasses=1,
        one_hot=one_hot,
        binarize=True,
        verbose=True,
        data_format=data_format,
    )
    output = op(
        polygon_vertices=vertices,
        vertex_counts_per_polygon=vertices_per_polygon,
        class_ids_per_polygon=class_ids_per_polygon,
        polygons_per_image=polygons_per_image,
    )

    assert expected_shape == output.shape.as_list()


@pytest.mark.parametrize(
    "vertices_per_polygon,polygons_per_image,single_image,\
                          expected_batch_size",
    [
        # Polygons_per_image == None implies single image.
        ([1], None, True, None),
        (tf.compat.v1.placeholder(dtype=tf.int64), None, True, None),
        # Empty polygons_per_image implies single image.
        ([], [], True, None),
        # Polygons_per_image of size 1 implies 1 image.
        ([1], [3], False, 1),
        # Polygons_per_image placeholder of unknown shape implies batch dimension of unknown size.
        (
            tf.compat.v1.placeholder(dtype=tf.int64),
            tf.compat.v1.placeholder(dtype=tf.int64),
            False,
            None,
        ),
        # When placeholder size if specified, batch dimension size is known.
        (
            tf.compat.v1.placeholder(shape=[1], dtype=tf.int64),
            tf.compat.v1.placeholder(shape=[4], dtype=tf.int64),
            False,
            4,
        ),
        # Batch dimension size is computed from variable shape.
        (
            tf.compat.v1.placeholder(shape=[1], dtype=tf.int64),
            tf.Variable([1, 2], dtype=tf.int64),
            False,
            2,
        ),
    ],
)
def test_dense_polygon_rasterizer_shape_inference_batch_size(
    vertices_per_polygon, polygons_per_image, single_image, expected_batch_size
):
    """Test different ways of specifying batch size."""
    if single_image:
        expected_shape = [1, 90, 100]
    else:
        expected_shape = [expected_batch_size, 1, 90, 100]
    _check_dense_shape_inference(
        vertices_per_polygon,
        polygons_per_image,
        height=90,
        width=100,
        one_hot=False,
        data_format=DataFormat.CHANNELS_FIRST,
        expected_shape=expected_shape,
    )


@pytest.mark.parametrize("one_hot,expected_nclasses", [(False, 1), (True, 2)])
def test_dense_polygon_rasterizer_shape_inference_classes(one_hot, expected_nclasses):
    """Test shape inference for the number of classes."""
    vertices_per_polygon = [1]
    polygons_per_image = [1]
    expected_shape = [1, expected_nclasses, 90, 100]
    _check_dense_shape_inference(
        vertices_per_polygon,
        polygons_per_image,
        height=90,
        width=100,
        one_hot=one_hot,
        data_format=DataFormat.CHANNELS_FIRST,
        expected_shape=expected_shape,
    )


@pytest.mark.parametrize(
    "width,height,expected_width,expected_height",
    [
        (100, 90, 100, 90),
        (
            tf.compat.v1.placeholder(dtype=tf.int32),
            tf.compat.v1.placeholder(dtype=tf.int32),
            None,
            None,
        ),
        (
            tf.Variable(100, dtype=tf.int32),
            tf.Variable(100, dtype=tf.int32),
            None,
            None,
        ),
    ],
)
def test_dense_polygon_rasterizer_shape_inference_width_height(
    width, height, expected_width, expected_height
):
    """Test shape inference for the different ways of specifying width and height."""
    vertices_per_polygon = [1]
    polygons_per_image = [1]
    expected_shape = [1, 1, expected_height, expected_width]
    _check_dense_shape_inference(
        vertices_per_polygon,
        polygons_per_image,
        height=height,
        width=width,
        one_hot=False,
        data_format=DataFormat.CHANNELS_FIRST,
        expected_shape=expected_shape,
    )


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("one_hot", [False, True])
@pytest.mark.parametrize("binarize", [False, True])
@pytest.mark.parametrize("num_samples", [1, 5])
@pytest.mark.parametrize("single_image", [False, True])
@pytest.mark.parametrize(
    "data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize("cpu", [False, True])
def test_polygon_rasterizer(
    sparse, one_hot, binarize, num_samples, single_image, data_format, cpu
):
    """Test polygon rasterizer output."""
    class_id_sparse = True

    test_name = "sparse" if sparse else "dense"
    test_name += "_onehot" if one_hot else "_noonehot"
    test_name += "_bin" if binarize else "_nobin"
    test_name += "_onesample" if num_samples == 1 else "_multisample"
    test_name += "_single" if single_image else "_nosingle"
    test_name += (
        "_channelslast" if data_format == DataFormat.CHANNELS_LAST else "_channelsfirst"
    )
    test_name += "_cpu" if cpu else "_gpu"

    device = "cpu:0" if cpu else "gpu:0"

    should_raise = binarize is False and one_hot is False

    vertices = [
        # L
        [5.183, 9.818],
        [3.298, 92.005],
        [24.787, 92.005],
        [28.086, 77.019],
        [17.907, 69.29],
        # A
        [29.5, 92.005],
        [34.306, 9.818],
        [43.731, 92.005],
        # N
        [46.465, 92.005],
        [43.731, 10.667],
        [57.021, 83.428],
        [56.078, 11.892],
        [60.603, 12.174],
        [61.262, 93.041],
        [50.706, 87.669],
        # E
        [64.467, 86.256],
        [64.467, 93.607],
        [74.834, 93.607],
        [74.08, 85.689],
        [67.671, 82.392],
        [67.954, 60.996],
        [73.42, 59.865],
        [73.608, 53.268],
        [68.802, 51.477],
        [68.236, 27.443],
        [71.818, 24.521],
        [72.007, 11.986],
        [64.467, 11.986],
        # N
        [76.907, 93.041],
        [75.776, 11.986],
        [80.583, 11.986],
        [84.824, 85.973],
        [84.824, 11.986],
        [87.746, 11.986],
        [87.746, 93.041],
        [80.395, 89.365],
        # E
        [89.632, 93.607],
        [94.438, 93.607],
        [94.438, 85.407],
        [92.035, 84.842],
        [92.035, 53.174],
        [94.438, 52.514],
        [94.438, 44.597],
        [92.035, 41.769],
        [92.459, 20.28],
        [95.192, 20.28],
        [94.627, 12.457],
        [90.386, 11.986],
        # T
        [96.041, 11.986],
        [96.041, 15.567],
        [97.549, 15.567],
        [96.041, 93.607],
        [98.585, 93.607],
        [98.585, 15.85],
        [99.527, 15.85],
        [99.527, 11.986],
        # triangle
        [0.0, 30.0],
        [50.0, 40.0],
        [100.0, 20.0],
    ]

    vertices_per_polygon = [5, 3, 7, 13, 8, 12, 8, 3]
    class_ids_per_polygon = [0, 1, 2, 0, 1, 2, 0, 1]
    if single_image:
        polygons_per_image = [8]
    else:
        polygons_per_image = [0, 3, 0, 5, 0]

    num_classes = max(class_ids_per_polygon) + 1
    batch_size = len(polygons_per_image)

    with tf.device(device):
        if sparse is False:
            if single_image:
                polygons_per_image = None
            try:
                output = _draw_dense(
                    num_classes,
                    one_hot,
                    binarize,
                    num_samples,
                    data_format,
                    polygons_per_image,
                    vertices_per_polygon,
                    class_ids_per_polygon,
                    vertices,
                )
            except ValueError:
                assert should_raise
                return
        else:
            curr_polygon = 0
            max_polygons_per_image = 0
            max_vertices_per_polygon = 0
            indices = []
            class_id_indices = []
            for b in range(batch_size):
                num_polygons = polygons_per_image[b]
                max_polygons_per_image = max(num_polygons, max_polygons_per_image)
                for p in range(num_polygons):
                    num_vertices = vertices_per_polygon[curr_polygon]
                    if single_image:
                        class_id_indices.append([p, 0])
                    else:
                        class_id_indices.append([b, p, 0])
                    curr_polygon += 1
                    max_vertices_per_polygon = max(
                        num_vertices, max_vertices_per_polygon
                    )
                    for v in range(num_vertices):
                        if single_image:
                            indices.append([p, v, 0])
                            indices.append([p, v, 1])
                        else:
                            indices.append([b, p, v, 0])
                            indices.append([b, p, v, 1])

            dense_shape = [max_polygons_per_image, max_vertices_per_polygon, 2]
            if single_image is False:
                dense_shape = [batch_size] + dense_shape

            flat_vertices = tf.reshape(
                tf.constant(vertices, dtype=tf.float32), shape=[len(indices)]
            )

            polygons = tf.SparseTensor(
                values=flat_vertices,
                indices=tf.constant(indices, dtype=tf.int64),
                dense_shape=dense_shape,
            )

            class_ids = tf.constant(class_ids_per_polygon, dtype=tf.int64)

            if class_id_sparse:
                class_id_dense_shape = [max_polygons_per_image, 1]
                if single_image is False:
                    class_id_dense_shape = [batch_size] + class_id_dense_shape
                class_ids = tf.SparseTensor(
                    values=class_ids,
                    indices=tf.constant(class_id_indices, dtype=tf.int64),
                    dense_shape=class_id_dense_shape,
                )

            try:
                output = _draw_sparse(
                    num_classes,
                    one_hot,
                    binarize,
                    num_samples,
                    data_format,
                    polygons,
                    class_ids,
                )
            except ValueError:
                assert should_raise
                return

        # Check output image shape.
        num_output_channels = num_classes + 1 if one_hot else 1
        if data_format == DataFormat.CHANNELS_LAST:
            expected_shape = (90, 100, num_output_channels)
        else:
            expected_shape = (num_output_channels, 90, 100)
        if single_image is False:
            expected_shape = (batch_size,) + expected_shape
        assert output.shape == expected_shape

        # Cancel the transpose effect, convert the op's final output from NHWC to NCHW for testing
        if data_format == DataFormat.CHANNELS_LAST:
            if single_image:
                output = np.transpose(output, [2, 0, 1])
            else:
                output = np.transpose(output, [0, 3, 1, 2])

        if single_image:
            # Tile classes horizontally.
            output = np.transpose(output, [1, 0, 2])
            output = output.reshape(
                (output.shape[0], output.shape[1] * output.shape[2])
            )
        else:
            # Tile images vertically, classes horizontally.
            output = np.transpose(output, [0, 2, 1, 3])
            output = output.reshape(
                (output.shape[0] * output.shape[1], output.shape[2] * output.shape[3])
            )

        test_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test_polygon_rasterizer"
        )

        if debug_save_images:
            try:
                os.mkdir(test_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

        if one_hot:
            # Scale 1.0 to full 8b white.
            output *= 255.0
        else:
            # Map class ids to gray scale.
            output *= 255.0 / num_classes

        # Construct an image out of each output slice.
        channel = output.astype(np.uint8)
        image = np.stack([channel, channel, channel], axis=-1)

        # Optionally save test images to disk for visual comparison.
        if debug_save_images:
            debug_im = Image.fromarray(image)
            debug_im.save("%s/test_%s.png" % (test_dir, test_name))

        # Load reference image.
        ref_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test_polygon_rasterizer_ref"
        )
        ref_image = Image.open("%s/test_%s.png" % (ref_dir, test_name))
        ref_image = np.array(ref_image).astype(np.float32)

        # Compare and assert that test images match reference.
        # Note that there might be slight differences depending on whether the code
        # is run on CPU or GPU, or between different GPUs, CUDA versions, TF versions,
        # etc. We may need to change this assertion to allow some tolerance. Before
        # doing that, please check the generated images to distinguish bugs from
        # small variations.
        squared_diff = np.square(np.subtract(ref_image, image.astype(np.float)))
        assert np.sum(squared_diff) < 0.0001


def test_serialization_and_deserialization():
    """Test that it is a MaglevObject that can be serialized and deserialized."""
    op = PolygonRasterizer(
        width=3,
        height=3,
        nclasses=1,
        one_hot=True,
        binarize=True,
        verbose=True,
        data_format=DataFormat.CHANNELS_FIRST,
    )
    op_dict = op.serialize()
    deserialized_op = deserialize_tao_object(op_dict)
    assert op.width == deserialized_op.width
    assert op.height == deserialized_op.height
    assert op.nclasses == deserialized_op.nclasses
    assert op.one_hot == deserialized_op.one_hot
    assert op.binarize == deserialized_op.binarize
    assert op.verbose == deserialized_op.verbose
    assert op.data_format == deserialized_op.data_format
