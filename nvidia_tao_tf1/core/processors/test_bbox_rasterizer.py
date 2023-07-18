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
"""Test the bbox rasterizer processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import math
import os

import numpy as np
from PIL import Image
import pytest
import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import deserialize_maglev_object
from nvidia_tao_tf1.core.processors import BboxRasterizer
from nvidia_tao_tf1.core.types import DataFormat

# Shorten these for convenience.
ELLIPSE = BboxRasterizer.DRAW_MODE_ELLIPSE
RECTANGLE = BboxRasterizer.DRAW_MODE_RECTANGLE

PASS = BboxRasterizer.GRADIENT_MODE_PASSTHROUGH
COV = BboxRasterizer.GRADIENT_MODE_MULTIPLY_BY_COVERAGE

# Debug mode for saving generated images to disk.
debug_save_shape_images = False


# Special case 3x3 matrix multiply where the third column of inputs and output is [0,0,1].
def mul3x2(ml, mr):
    return [
        [
            ml[0][0] * mr[0][0] + ml[0][1] * mr[1][0],
            ml[0][0] * mr[0][1] + ml[0][1] * mr[1][1],
            0.0,
        ],
        [
            ml[1][0] * mr[0][0] + ml[1][1] * mr[1][0],
            ml[1][0] * mr[0][1] + ml[1][1] * mr[1][1],
            0.0,
        ],
        [
            ml[2][0] * mr[0][0] + ml[2][1] * mr[1][0] + mr[2][0],
            ml[2][0] * mr[0][1] + ml[2][1] * mr[1][1] + mr[2][1],
            1.0,
        ],
    ]


# Special case 3x3 matrix inverse where the third column of input and output is [0,0,1].
def inv3x2(mat):
    det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
    oodet = 1.0 / det
    return [
        [mat[1][1] * oodet, -mat[0][1] * oodet, 0.0],
        [-mat[1][0] * oodet, mat[0][0] * oodet, 0.0],
        [
            -(mat[2][0] * mat[1][1] + mat[2][1] * -mat[1][0]) * oodet,
            -(mat[2][0] * -mat[0][1] + mat[2][1] * mat[0][0]) * oodet,
            1.0,
        ],
    ]


def matrix_from_bbox(xmin, ymin, xmax, ymax):
    # Compute a matrix that transforms bbox from canonical [-1,1] space to image coordinates.
    half_width = (xmax - xmin) * 0.5
    half_height = (ymax - ymin) * 0.5
    smat = [[half_width, 0.0, 0.0], [0.0, half_height, 0.0], [0.0, 0.0, 1.0]]

    tmat = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [half_width + xmin, half_height + ymin, 1.0],
    ]

    mat = mul3x2(smat, tmat)
    # [-1,-1] -> [-hw+hw+xmin, -hh+hh+ymin] = [xmin, ymin]
    # [1,1]   -> [hw+hw+xmin, hh+hh+ymin] = [xmax-xmin+xmin, ymax-ymin+ymin] = [xmax, ymax]

    # Inverse matrix transforms from image coordinates to canonical space.
    return inv3x2(mat)


def matrix_from_center(centerx, centery, half_width, half_height, angle):
    # Compute a matrix that transforms bbox from canonical [-1,1] space to image coordinates.
    smat = [[half_width, 0.0, 0.0], [0.0, half_height, 0.0], [0.0, 0.0, 1.0]]

    tmat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [centerx, centery, 1.0]]

    a = angle * math.pi / 180.0
    rot = [[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)], [0.0, 0.0]]
    mat = mul3x2(smat, rot)
    mat = mul3x2(mat, tmat)

    # Inverse matrix transforms from image coordinates to canonical space.
    # c * smat * rot * tmat = p
    # c = p * tmat^-1 * rot^-1 * smat^-1
    return inv3x2(mat)


def gradient_from_endpoints(sx, sy, svalue, ex, ey, evalue):
    # edge = [ex - sx, ey - sy]
    # p = [px - sx, py - sy]
    # ratio = dot(p, edge) / |edge|^2
    # value = (1-ratio) * svalue + ratio * evalue
    # ->
    # l = 1 / |edge|^2
    # ratio = ((ex - sx) * (px - sx) + (ey - sy) * (py - sy)) * l
    # ->
    # dvalue = (evalue - svalue), dx = (ex - sx), dy = (ey - sy)
    # value = dvalue * dx * l * px +
    #         dvalue * dy * l * py +
    #         svalue - dvalue * dx * l * sx - dvalue * dy * l * sy
    # ->
    # A = dvalue * dx * l
    # B = dvalue * dy * l
    # C = svalue - dvalue * dx * l * sx - dvalue * dy * l * sy

    dx = ex - sx
    dy = ey - sy
    le = 0.0
    if dx != 0.0 or dy != 0.0:
        le = 1.0 / (dx * dx + dy * dy)
    dvalue = (evalue - svalue) * le
    dvx = dvalue * dx
    dvy = dvalue * dy
    offset = svalue - (dvx * sx + dvy * sy)
    vec = [dvx, dvy, offset]
    return vec


bbox_tests = [
    # Test multiple images and classes.
    (
        "0",  # Test name
        4,  # num_images
        2,  # num_classes
        1,  # num_gradients
        [2, 2, 0, 1],  # num bboxes per image
        [0, 1, 1, 0, 1],  # bbox class IDs
        [
            matrix_from_bbox(0.0, 0.0, 80.0, 60.0),  # bbox matrices (3D)
            matrix_from_bbox(80.0, 0.0, 160.0, 60.0),
            matrix_from_bbox(80.0, 60.0, 160.0, 120.0),
            matrix_from_bbox(0.0, 60.0, 80.0, 120.0),
            matrix_from_bbox(40.0, 30.0, 120.0, 90.0),
        ],
        [
            [[0.0, 0.0, 32.0]],
            [[0.0, 0.0, 64.0]],
            [[0.0, 0.0, 128.0]],
            [[0.0, 0.0, 192.0]],
            [[0.0, 0.0, 224.0]],
        ],  # bbox gradients (3D)
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],  # bbox coverage radii
        [ELLIPSE, RECTANGLE, ELLIPSE, ELLIPSE, ELLIPSE],  # bbox flags
        [0.5, 1.5, 0.5, 1.5, 1.5],  # bbox sort values
        [COV],
    ),  # gradient flags
    # Test empty images.
    (
        "1",  # Test name
        6,  # num_images
        1,  # num_classes
        1,  # num_gradients
        [0, 1, 0, 0, 1, 0],  # num bboxes per image
        [0, 0],  # bbox class IDs
        [
            matrix_from_bbox(0.0, 0.0, 160.0, 120.0),  # bbox matrices (3D)
            matrix_from_bbox(00.0, 0.0, 160.0, 120.0),
        ],
        [[[0.0, 0.0, 32.0]], [[0.0, 0.0, 64.0]]],  # bbox gradients (3D)
        [[1.0, 1.0], [1.0, 1.0]],  # bbox coverage radii
        [ELLIPSE, RECTANGLE],  # bbox flags
        [0.0, 0.0],  # bbox sort values
        [COV],
    ),  # gradient flags
    # Test basic shapes with only one constant gradient.
    (
        "2",
        1,
        1,
        1,
        [3],
        [0, 0, 0],
        [
            matrix_from_bbox(0.0, 0.0, 80.0, 60.0),
            matrix_from_bbox(40.0, 30.0, 120.0, 90.0),
            matrix_from_bbox(80.0, 60.0, 160.0, 120.0),
        ],
        [[[0.0, 0.0, 255.0]], [[0.0, 0.0, 255.0]], [[0.0, 0.0, 255.0]]],
        [[0.8, 0.6], [0.8, 0.9], [0.5, 0.9]],
        [RECTANGLE, ELLIPSE, RECTANGLE],
        [1.0, 2.0, 3.0],
        [COV],
    ),
    # Reverse sort order.
    (
        "3",
        1,
        1,
        1,
        [3],
        [0, 0, 0],
        [
            matrix_from_bbox(0.0, 0.0, 80.0, 60.0),
            matrix_from_bbox(40.0, 30.0, 120.0, 90.0),
            matrix_from_bbox(80.0, 60.0, 160.0, 120.0),
        ],
        [[[0.0, 0.0, 255.0]], [[0.0, 0.0, 255.0]], [[0.0, 0.0, 255.0]]],
        [[0.8, 0.6], [0.8, 0.9], [0.5, 0.9]],
        [RECTANGLE, ELLIPSE, RECTANGLE],
        [3.0, 2.0, 1.0],
        [COV],
    ),
    # Zero sort values should draw in the order bboxes are specified.
    (
        "4",
        1,
        1,
        1,
        [3],
        [0, 0, 0],
        [
            matrix_from_bbox(40.0, 30.0, 160.0, 90.0),
            matrix_from_bbox(0.0, 0.0, 120.0, 90.0),
            matrix_from_bbox(90.0, 60.0, 160.0, 120.0),
        ],
        [[[0.0, 0.0, 255.0]], [[0.0, 0.0, 255.0]], [[0.0, 0.0, 255.0]]],
        [[0.9, 0.9], [0.9, 0.9], [0.9, 0.9]],
        [RECTANGLE, ELLIPSE, RECTANGLE],
        [0.0, 0.0, 0.0],
        [COV],
    ),
    # Test affine transformations.
    (
        "5",
        1,
        1,
        1,
        [3],
        [0, 0, 0],
        [
            matrix_from_center(40.0, 30.0, 30.0, 22.5, 15.0),
            matrix_from_center(80.0, 60.0, 30.0, 22.5, 22.5),
            matrix_from_center(120.0, 90.0, 30.0, 22.5, 30.0),
        ],
        [[[0.0, 0.0, 255.0]], [[0.0, 0.0, 255.0]], [[0.0, 0.0, 255.0]]],
        [[0.8, 0.6], [0.8, 0.9], [0.5, 0.9]],
        [RECTANGLE, ELLIPSE, RECTANGLE],
        [1.0, 2.0, 3.0],
        [COV],
    ),
    # Test one constant and one interpolated gradient.
    (
        "6",
        1,
        1,
        2,
        [1],
        [0],
        [matrix_from_bbox(20, 15, 140, 105)],
        [[[0.0, 0.0, 255.0], gradient_from_endpoints(20, 15, 32, 140, 105, 224)]],
        [[1.0, 1.0]],
        [RECTANGLE],
        [1.0],
        [COV, PASS],
    ),
    # Test one constant and two interpolated gradients.
    (
        "7",
        1,
        1,
        3,
        [1],
        [0],
        [matrix_from_bbox(20, 15, 140, 105)],
        [
            [
                [0.0, 0.0, 255.0],
                gradient_from_endpoints(20, 15, 0, 140, 15, 255),
                gradient_from_endpoints(20, 15, 64, 20, 105, 192),
            ]
        ],
        [[1.0, 1.0]],
        [RECTANGLE],
        [2.0],
        [COV, PASS, PASS],
    ),
    # Empty image.
    ("8", 1, 1, 1, [0], [], [], [], [], [], [], [COV]),
]


@pytest.mark.parametrize(
    "data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize(
    "test_name,num_images,num_classes,num_gradients,bboxes_per_image,\
                          bbox_class_ids,bbox_matrices,bbox_gradients,bbox_coverage_radii,\
                          bbox_flags, bbox_sort_values, gradient_flags",
    bbox_tests,
)
@pytest.mark.parametrize("cpu", [False, True])
def test_bbox_rasterizer(
    test_name,
    num_images,
    num_classes,
    num_gradients,
    bboxes_per_image,
    bbox_class_ids,
    bbox_matrices,
    bbox_gradients,
    bbox_coverage_radii,
    bbox_flags,
    bbox_sort_values,
    gradient_flags,
    data_format,
    cpu,
):
    """Test the ground-truth generator for different shapes, sizes and deadzones."""

    device = "cpu:0" if cpu else "gpu:0"
    data_format_string = (
        "chlast" if data_format is DataFormat.CHANNELS_LAST else "chfirst"
    )
    device_string = "cpu" if cpu is True else "gpu"
    file_name = "test_%s_%s_%s.png" % (test_name, data_format_string, device_string)

    image_height = 120
    image_width = 160

    with tf.device(device):
        sess = tf.compat.v1.Session()
        op = BboxRasterizer(verbose=True, data_format=data_format)
        fetches = op(
            num_images=num_images,
            num_classes=num_classes,
            num_gradients=num_gradients,
            image_height=image_height,
            image_width=image_width,
            bboxes_per_image=bboxes_per_image,
            bbox_class_ids=bbox_class_ids,
            bbox_matrices=bbox_matrices,
            bbox_gradients=bbox_gradients,
            bbox_coverage_radii=bbox_coverage_radii,
            bbox_flags=bbox_flags,
            bbox_sort_values=bbox_sort_values,
            gradient_flags=gradient_flags,
        )

        sess.run(tf.compat.v1.global_variables_initializer())
        output = sess.run(fetches)

        # Cancel the transpose effect, convert the op's final output from NHWCG to NCGHW for
        # testing.
        if data_format == DataFormat.CHANNELS_LAST:
            output = np.transpose(output, [0, 3, 4, 1, 2])

        ref_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test_bbox_rasterizer_ref"
        )
        test_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test_bbox_rasterizer"
        )

        channel = np.reshape(
            output,
            [num_images * num_classes * num_gradients * image_height, image_width],
        ).astype(np.uint8)
        image = np.stack([channel, channel, channel, channel], axis=-1)

        if debug_save_shape_images:
            try:
                os.mkdir(test_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

            debug_im = Image.fromarray(image)
            debug_im.save("%s/%s" % (test_dir, file_name))

        # Load reference image.
        ref_image = Image.open("%s/%s" % (ref_dir, file_name))
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
    op = BboxRasterizer(verbose=True, data_format=DataFormat.CHANNELS_LAST)
    op_dict = op.serialize()
    deserialized_op = deserialize_maglev_object(op_dict)
    assert op.verbose == deserialized_op.verbose
    assert op.data_format == deserialized_op.data_format
