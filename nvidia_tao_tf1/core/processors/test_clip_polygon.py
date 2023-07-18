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
"""Test polygon clipping."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import tensorflow as tf

import nvidia_tao_tf1.core
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object

# Relative and absolute tolerances for comparing polygon coordinates.
_RTOL = 1e-6
_ATOL = 1e-6


def _flatten_polygon_list(polygons):
    """Convert from a list of polygons (list of (x, y) tuples) to a dense list of coordinates."""
    if polygons == []:
        return np.ndarray(shape=(0, 2), dtype=np.float32), []
    polygons_flattened = np.concatenate(polygons)
    points_per_polygon = np.array([len(a) for a in polygons])
    return polygons_flattened, points_per_polygon


def _to_polygon_list(polygons, points_per_polygon):
    """Convert from a dense list of coordinates to a list of polygons (list of (x, y) tuples)."""
    polygon_list = []
    start_index = 0
    for npoints in points_per_polygon:
        polygon_list.append(polygons[start_index : start_index + npoints, :])
        start_index += npoints
    return polygon_list


helper_tests = [
    # Test some random polygons.
    (
        [
            np.array([(1.0, 2.0), (3.0, 4.0)]),
            np.ndarray(shape=(0, 2), dtype=np.float32),
            np.array([(5.0, 6.0)]),
            np.array([(7.0, 8.0), (9.0, 10.0)]),
        ],
        np.array([2, 0, 1, 2]),
    ),
    # Test empty polygons.
    (
        [
            np.ndarray(shape=(0, 2), dtype=np.float32),
            np.ndarray(shape=(0, 2), dtype=np.float32),
        ],
        np.array([0, 0]),
    ),
]


@pytest.mark.parametrize("polygons,ppp_expected", helper_tests)
def test_polygon_test_helpers(polygons, ppp_expected):
    """Test the polygon helper functions that convert between a dense and seperate polygons."""
    pf, ppp = _flatten_polygon_list(polygons)
    np.testing.assert_array_equal(ppp, ppp_expected)
    polygons_roundtrip = _to_polygon_list(pf, ppp)
    for p, p_expected in zip(polygons, polygons_roundtrip):
        np.testing.assert_array_equal(p, p_expected)


def _clip_and_test(
    polygon_list,
    polygon_mask,
    expected_polygon_list,
    expected_polygon_index_mapping,
    closed,
):
    """Run the numpy arrays through our TensorFlow Op and compare."""
    polygons, points_per_polygon = _flatten_polygon_list(polygon_list)
    # expected_polygons, expected_points_per_polygon = _flatten_polygon_list(expected_polygon_list)

    clipper = nvidia_tao_tf1.core.processors.ClipPolygon(closed=closed)
    clipped_polygons, clipped_points_per_polygon, clipped_polygon_index_mapping = clipper(
        polygons=polygons,
        points_per_polygon=points_per_polygon,
        polygon_mask=polygon_mask,
    )
    sess = nvidia_tao_tf1.core.utils.test_session()
    np_clipped_polygons, np_clipped_points_per_polygon, np_clipped_polygon_index_mapping = sess.run(
        [clipped_polygons, clipped_points_per_polygon, clipped_polygon_index_mapping]
    )

    clipped_polygon_list = _to_polygon_list(
        np_clipped_polygons, np_clipped_points_per_polygon
    )
    np.testing.assert_array_equal(
        np_clipped_polygon_index_mapping, expected_polygon_index_mapping
    )

    assert len(expected_polygon_list) == len(clipped_polygon_list)
    for p, p_clipped in zip(expected_polygon_list, clipped_polygon_list):
        assert _polygon_equal(p, p_clipped) is True


@pytest.mark.parametrize("closed", (True, False))
def test_no_polygons(closed):
    """Test that no polygons as input returns no input."""
    _clip_and_test(
        polygon_list=[],
        polygon_mask=np.array([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]),
        expected_polygon_list=[],
        expected_polygon_index_mapping=[],
        closed=closed,
    )


@pytest.mark.parametrize("closed", (True, False))
def test_equal_coordinate_polygons(closed):
    """Test that polygons with identical coordinates are filtered out."""
    _clip_and_test(
        polygon_list=[np.array([(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])],
        polygon_mask=np.array([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]),
        expected_polygon_list=[],
        expected_polygon_index_mapping=[],
        closed=closed,
    )


@pytest.mark.parametrize("closed", (True, False))
def test_polygon_outside_mask(closed):
    """Test that polygons outside the mask are not drawn at all."""
    _clip_and_test(
        polygon_list=[
            np.array([(-2.0, -2.0), (-2.0, -1.0), (-1.0, -1.0), (-1.0, -2.0)])
        ],
        polygon_mask=np.array([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]),
        expected_polygon_list=[],
        expected_polygon_index_mapping=[],
        closed=closed,
    )


@pytest.mark.parametrize(
    "polygon_mask", (np.array([]), np.array([(0.0, 0.0), (0.0, 1.0)]))
)
@pytest.mark.parametrize("closed", (True, False))
def test_invalid_mask(closed, polygon_mask):
    """Test that an empty or non-polygonal mask raises an error."""
    with pytest.raises(tf.errors.InvalidArgumentError):
        _clip_and_test(
            polygon_list=[],
            polygon_mask=polygon_mask,
            expected_polygon_list=[],
            expected_polygon_index_mapping=[],
            closed=closed,
        )


def _polygon_equal(p1, p2):
    """Compare two polygons or polylines.

    Note the coordinate sequence (list) is allowed to be inverted and shifted.
    """
    np.testing.assert_array_equal(p1.shape, p2.shape)
    nvertices = p1.shape[0]
    p2p2 = np.concatenate((p2, p2))
    rp2p2 = np.flip(p2p2, axis=0)
    # Slide p1 over p2p2
    for i in range(nvertices):
        if np.allclose(p1, p2p2[i : i + nvertices, :], rtol=_RTOL, atol=_ATOL):
            return True
        if np.allclose(p1, rp2p2[i : i + nvertices, :], rtol=_RTOL, atol=_ATOL):
            return True
    return False


polygon_equal_tests = [
    # Exactly equal.
    (
        np.array([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]),
        np.array([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]),
    ),
    # Offset.
    (
        np.array([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]),
        np.array([(3.0, 4.0), (5.0, 6.0), (1.0, 2.0)]),
    ),
    # Inverted and offset.
    (
        np.array([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]),
        np.array([(5.0, 6.0), (3.0, 4.0), (1.0, 2.0)]),
    ),
]


@pytest.mark.parametrize("p1, p2", polygon_equal_tests)
def test_polygon_equal(p1, p2):
    """Test for the polygon equality helper function."""
    assert _polygon_equal(p1, p2) is True


encompass_tests = [
    (np.array([(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)])),
    (np.array([(1.0e6, 1.0e6), (0.0, 1.0e6), (0.0, 0.0), (1.0e6, 0.0)])),
]


@pytest.mark.parametrize("polygon_mask", encompass_tests)
def test_polygon_encompass_mask(polygon_mask):
    """Test that polygons encompassing the mask entirely yield the same as the mask."""
    polygon = polygon_mask * 2  # Simply inflate the polygon.
    _clip_and_test(
        polygon_list=[polygon],
        polygon_mask=polygon_mask,
        expected_polygon_list=[polygon_mask],
        expected_polygon_index_mapping=[0],
        closed=True,
    )


@pytest.mark.parametrize("polygon_mask", encompass_tests)
def test_polyline_encompass_mask(polygon_mask):
    """Test that polylines circumscribing the mask are entirely removed."""
    polyline = polygon_mask * 2  # Simply inflate the polygon.
    _clip_and_test(
        polygon_list=[polyline],
        polygon_mask=polygon_mask,
        expected_polygon_list=[],
        expected_polygon_index_mapping=[],
        closed=False,
    )


@pytest.mark.parametrize("polygon", encompass_tests)
@pytest.mark.parametrize("closed", (True, False))
def test_mask_encompass_polygon(polygon, closed):
    """Test that a mask encompassing the polygons will leave the polygons as-is."""
    polygon_mask = polygon * 2  # Simply inflate the polygon.
    _clip_and_test(
        polygon_list=[polygon],
        polygon_mask=polygon_mask,
        expected_polygon_list=[polygon],
        expected_polygon_index_mapping=[0],
        closed=closed,
    )


@pytest.mark.parametrize("xy_range", (1.0, 1.0e6))
def test_many_polylines_in_encompassing_mask(
    xy_range, npolylines=1000, max_nvertices=100
):
    """Test that many polygons inside one encompassing mask stay the same."""
    polygon_mask = np.array(
        [
            (-xy_range, -xy_range),
            (xy_range, -xy_range),
            (xy_range, xy_range),
            (-xy_range, xy_range),
        ]
    )
    # Create random polygons
    polylines = []
    np.random.seed(42)
    for _ in range(npolylines):
        # Create random vertices centered around the axis origin, to stay within mask bounds.
        nvertices = np.random.randint(low=3, high=max_nvertices)
        polylines.append((np.random.rand(nvertices, 2) - 0.5) * xy_range)
    _clip_and_test(
        polygon_list=polylines,
        polygon_mask=polygon_mask,
        expected_polygon_list=polylines,
        expected_polygon_index_mapping=list(range(npolylines)),
        closed=False,
    )


def test_self_intersecting_polygon():
    r"""Draw an intersecting polygon (8-shape) that is encompassed by the mask.

    1mmm2    +---+
    m\ /m     \1/
    m X m ->   X
    m/ \m     /2\
    3mmm4    +---+

    Note: The polygon clipper will reduce self-intersecting polygons into multiple
    non-intersecting polygons.
    """
    polygon_list = [np.array([(-2.0, -2.0), (2.0, 2.0), (2.0, -2.0), (-2.0, 2.0)])]
    polygon_mask = np.array([(2.0, 2.0), (-2.0, 2.0), (-2.0, -2.0), (2.0, -2.0)])
    expected_polygon_list = [
        np.array([(-2.0, -2.0), (0.0, 0.0), (-2.0, 2.0)]),
        np.array([(2.0, 2.0), (0.0, 0.0), (2.0, -2.0)]),
    ]
    expected_polygon_index_mapping = [0, 0]
    _clip_and_test(
        polygon_list=polygon_list,
        polygon_mask=polygon_mask,
        expected_polygon_list=expected_polygon_list,
        expected_polygon_index_mapping=expected_polygon_index_mapping,
        closed=True,
    )


def test_self_intersecting_polygon_clipping():
    r"""Draw an intersecting polygon (8-shape) that is cropped by a smaller mask.

    1-------2
     \     /
      mmmmm     +---+
      m\ /m      \1/
      m x m  ->   X
      m/ \m      /2\
      mmmmm     +---+
     /     \
    3-------4


    Note that the polygon clipper will reduce self-intersecting polygons into multiple
    non-intersecting polygons.
    """
    polygon_list = [np.array([(-2.0, -2.0), (2.0, 2.0), (2.0, -2.0), (-2.0, 2.0)])]
    polygon_mask = np.array([(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)])
    expected_polygon_list = [
        np.array([(-1.0, -1.0), (0.0, 0.0), (-1.0, 1.0)]),
        np.array([(1.0, 1.0), (0.0, 0.0), (1.0, -1.0)]),
    ]
    expected_polygon_index_mapping = [0, 0]
    _clip_and_test(
        polygon_list=polygon_list,
        polygon_mask=polygon_mask,
        expected_polygon_list=expected_polygon_list,
        expected_polygon_index_mapping=expected_polygon_index_mapping,
        closed=True,
    )


def test_convex_polygon_clipping():
    """Test simple convex polygon clipping.

      +-+
    mm|m|mm      +-+
    m | | m  ->  |1|
    mm|m|mm      +-+
      +-+
    """
    polygon_list = [np.array([(1.0, 0.0), (2.0, 0.0), (2.0, 3.0), (1.0, 3.0)])]
    polygon_mask = np.array([(0.0, 1.0), (3.0, 1.0), (3.0, 2.0), (0.0, 2.0)])
    expected_polygon_list = [np.array([(1.0, 1.0), (1, 2.0), (2.0, 2.0), (2.0, 1.0)])]
    expected_polygon_index_mapping = [0]
    _clip_and_test(
        polygon_list=polygon_list,
        polygon_mask=polygon_mask,
        expected_polygon_list=expected_polygon_list,
        expected_polygon_index_mapping=expected_polygon_index_mapping,
        closed=True,
    )


def test_polyline_clipping():
    """Test simple 'convex' polyline clipping.

      2-3        1 2
    mm|m|mm      + +
    m | | m  ->  | |
    mm|m|mm      + +
      1 4
    """
    polyline_list = [np.array([(1.0, 0.0), (1.0, 3.0), (2.0, 3.0), (2.0, 0.0)])]
    polygon_mask = np.array([(0.0, 1.0), (3.0, 1.0), (3.0, 2.0), (0.0, 2.0)])
    expected_polyline_list = [
        np.array([(1.0, 1.0), (1.0, 2.0)]),
        np.array([(2.0, 1.0), (2.0, 2.0)]),
    ]
    expected_polyline_index_mapping = [0, 0]
    _clip_and_test(
        polygon_list=polyline_list,
        polygon_mask=polygon_mask,
        expected_polygon_list=expected_polyline_list,
        expected_polygon_index_mapping=expected_polyline_index_mapping,
        closed=False,
    )


def test_concave_polygon_clipping():
    r"""Test that drawing a 'V' masked with a thin horizontal mask results in two polygons.
      _       _
      1\     /3
    mm\m\mmm/m/mm      _     _
    m  \ \2/ /  m  -> \1\   /2/
    mmmm\mmm/mmmm      \_\ /_/
         \4/

    """
    polygon_list = [np.array([(-3.0, 3.0), (0.0, 0.0), (3.0, 3.0), (0.0, -3.0)])]
    polygon_mask = np.array([(-3.0, 2.0), (-3.0, 1.0), (3.0, 1.0), (3.0, 2.0)])
    expected_polygon_list = [
        np.array([(-2.0, 2.0), (-2.5, 2.0), (-2.0, 1.0), (-1.0, 1.0)]),
        np.array([(2.5, 2.0), (2.0, 2.0), (1.0, 1.0), (2.0, 1.0)]),
    ]
    expected_polygon_index_mapping = [0, 0]
    _clip_and_test(
        polygon_list=polygon_list,
        polygon_mask=polygon_mask,
        expected_polygon_list=expected_polygon_list,
        expected_polygon_index_mapping=expected_polygon_index_mapping,
        closed=True,
    )


@pytest.mark.parametrize("swap_polygon_with_mask", (True, False))
def test_corner_polygon_clipping(swap_polygon_with_mask):
    r"""Test that drawing a 'V' masked with a thin horizontal mask results in two polygons.

    mmmmm
    m   m
    m 1-m-2
    m | m |
    mmmmm |
      |   |
      4---3
    """
    polygons = np.array([(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)])
    polygon_mask = np.array([(0.0, 0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0)])
    if swap_polygon_with_mask:
        polygon_mask, polygons = polygons, polygon_mask
    expected_polygon_list = [np.array([(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)])]
    expected_polygon_index_mapping = [0]
    _clip_and_test(
        polygon_list=[polygons],
        polygon_mask=polygon_mask,
        expected_polygon_list=expected_polygon_list,
        expected_polygon_index_mapping=expected_polygon_index_mapping,
        closed=True,
    )


@pytest.mark.parametrize("swap_polygon_with_mask", (True, False))
def test_side_polygon_clipping(swap_polygon_with_mask):
    r"""Test that drawing a 'V' masked with a thin horizontal mask results in two polygons.

    Args:
        swap_polygon_with_mask: swaps the polygon with the mask. Should give the same result.

    mmmmm
    m 1-m-2
    m | m |
    m 3-m-4
    mmmmm
    """
    polygons = np.array([(1.0, 1.0), (4.0, 1.0), (4.0, 2.0), (1.0, 2.0)])
    polygon_mask = np.array([(0.0, 0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)])
    if swap_polygon_with_mask:
        polygon_mask, polygons = polygons, polygon_mask
    expected_polygon_list = [np.array([(1.0, 1.0), (3.0, 1.0), (3.0, 2.0), (1.0, 2.0)])]
    expected_polygon_index_mapping = [0]
    _clip_and_test(
        polygon_list=[polygons],
        polygon_mask=polygon_mask,
        expected_polygon_list=expected_polygon_list,
        expected_polygon_index_mapping=expected_polygon_index_mapping,
        closed=True,
    )


polyline_corner_tests = [
    (
        np.array([(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]),
        np.array([(1.0, 1.0), (2.0, 1.0)]),
    ),
    (
        np.array([(3.0, 1.0), (3.0, 3.0), (1.0, 3.0), (1.0, 1.0)]),
        np.array([(1.0, 2.0), (1.0, 1.0)]),
    ),
    (
        np.array([(3.0, 3.0), (1.0, 3.0), (1.0, 1.0), (3.0, 1.0)]),
        np.array([(1.0, 2.0), (1.0, 1.0), (2.0, 1.0)]),
    ),
    (
        np.array([(1.0, 3.0), (1.0, 1.0), (3.0, 1.0), (3.0, 3.0)]),
        np.array([(1.0, 2.0), (1.0, 1.0), (2.0, 1.0)]),
    ),
]


@pytest.mark.parametrize("reverse_path", (True, False))
@pytest.mark.parametrize("polylines,expected_polylines", polyline_corner_tests)
def test_corner_polyline_clipping(polylines, expected_polylines, reverse_path):
    r"""Test corner polyline cropping for differently organized paths.

    Args:
        reverse_path: reversing the path should yield the same result.

    mmmmm    mmmmm    mmmmm    mmmmm
    m   m    m   m    m   m    m   m
    m 1-m-2  m 4 m 1  m 3-m-4  m 2-m-3
    m   m |  m | m |  m | m    m | m |
    mmmmm |  mmmmm |  mmmmm    mmmmm |
          |    |   |    |        |   |
      4---3    3---2    2---1    1   4
    """
    if reverse_path:
        polylines = np.flip(polylines, axis=0)
    polygon_mask = np.array([(0.0, 0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0)])
    expected_polyline_index_mapping = [0]
    _clip_and_test(
        polygon_list=[polylines],
        polygon_mask=polygon_mask,
        expected_polygon_list=[expected_polylines],
        expected_polygon_index_mapping=expected_polyline_index_mapping,
        closed=False,
    )


def test_serialization_and_deserialization():
    """Test that it is a MaglevObject that can be serialized and deserialized."""
    clipper = nvidia_tao_tf1.core.processors.ClipPolygon(closed=False)
    clipper_dict = clipper.serialize()
    deserialized_clipper = deserialize_tao_object(clipper_dict)
    assert clipper.closed == deserialized_clipper.closed
