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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import os

import numpy as np
from PIL import Image
import pytest
import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import deserialize_tao_object
from nvidia_tao_tf1.core.processors import SpatialTransform
from nvidia_tao_tf1.core.processors.augment.spatial import PolygonTransform
from nvidia_tao_tf1.core.types import DataFormat


# Debug mode for saving generated images to disk.
debug_save_shape_images = False

device_list = ["/cpu", "/gpu"]


def _get_image_batch(batch_size, data_format=DataFormat.CHANNELS_LAST):
    """
    Generate batch of real images.

    The image used is a real image, because spatial translations (mainly rotations) are not
    lossless, especially for pixels with random values. Real images are quite smooth and therefore
    do not suffer much from this loss at all.
    """
    test_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "pekka_256.png"
    )
    image = Image.open(test_file)
    images_np = np.array(image).astype(np.float32) / 255.0
    images_np = np.repeat([images_np], batch_size, axis=0)
    if data_format == DataFormat.CHANNELS_FIRST:
        images_np = np.transpose(images_np, (0, 3, 1, 2))
    return images_np


def _apply_stm_on_images(
    device,
    images,
    stms,
    method,
    shape=None,
    background_value=0.0,
    input_data_format=DataFormat.CHANNELS_LAST,
    output_data_format=DataFormat.CHANNELS_LAST,
    input_dtype=np.float32,
    output_dtype=None,
):
    with tf.device(device):
        stm_op = SpatialTransform(
            method=method,
            background_value=background_value,
            input_data_format=input_data_format,
            output_data_format=output_data_format,
        )
        # Note: placeholders are needed to make TensorFlow respect device placement.
        placeholder_images = tf.compat.v1.placeholder(dtype=input_dtype)
        placeholder_stms = tf.compat.v1.placeholder(dtype=tf.float32)
        fetches = stm_op(placeholder_images, placeholder_stms, shape)
        sess = tf.compat.v1.Session()
        return sess.run(
            fetches, feed_dict={placeholder_images: images, placeholder_stms: stms}
        )


def _apply_stm_on_default_image(device, stms, method, background_value=0.0):
    input_np = _get_image_batch(stms.shape[0])
    return _apply_stm_on_images(
        device=device,
        images=input_np,
        stms=stms,
        method=method,
        background_value=background_value,
    )


def _empty_canvas_checker(device, stms, background_value=0.42):
    output_np = _apply_stm_on_default_image(
        device=device, stms=stms, method="bilinear", background_value=background_value
    )
    blank_canvas = np.full(
        output_np.shape, fill_value=background_value, dtype=np.float32
    )
    np.testing.assert_array_equal(
        blank_canvas,
        output_np,
        err_msg="input array changed after "
        "application of an identity spatial transformation matrix.",
    )


@pytest.mark.parametrize("device", device_list)
@pytest.mark.parametrize(
    "input_data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize(
    "output_data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("method", ["nearest", "bilinear", "bicubic"])
def test_identity_spatial_transform(
    device, method, input_data_format, output_data_format, dtype, batch_size=2
):
    """Test an identity spatial transformation, which should not significantly augment the image."""
    input_np = _get_image_batch(batch_size, data_format=input_data_format)
    # For uint8, convert to [0,255] range.
    if dtype == np.uint8:
        input_np *= 255.0
    stms = np.repeat([np.eye(3)], batch_size, axis=0)
    output_np = _apply_stm_on_images(
        device=device,
        images=input_np,
        stms=stms,
        method=method,
        input_data_format=input_data_format,
        output_data_format=output_data_format,
        output_dtype=dtype,
    )

    if input_data_format != output_data_format:
        if input_data_format == DataFormat.CHANNELS_FIRST:
            axes = [0, 2, 3, 1]
        else:
            axes = [0, 3, 1, 2]
        input_np = np.transpose(input_np, axes=axes)
    input_np = input_np.astype(dtype=dtype)

    if method in ("nearest", "bilinear"):
        # Nearest and bilinear should result in exact values when input and output are fp32.
        if dtype == np.float32:
            np.testing.assert_array_equal(
                input_np,
                output_np,
                err_msg="input array changed after "
                "application of an identity spatial transformation matrix.",
            )
        else:
            # When input and output dtypes don't match, allow more tolerance.
            np.testing.assert_almost_equal(
                input_np,
                output_np,
                decimal=3,
                err_msg="input array changed after "
                "application of an identity spatial transformation matrix.",
            )
    elif method == "bicubic":
        # Bicubic does not result in exact values upon identity transform (expected).
        mean_abs_diff = np.mean(np.abs(output_np - input_np))
        if dtype == np.uint8:
            mean_abs_diff /= 255.0
        assert (
            mean_abs_diff < 0.02
        ), "input array changed significantly after identity transform"
    else:
        raise ValueError("Unknown method %s" % method)


@pytest.mark.parametrize("device", device_list)
@pytest.mark.parametrize(
    "input_data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize(
    "output_data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("method", ["nearest", "bilinear", "bicubic"])
@pytest.mark.parametrize("zoom", ["1x", "2x"])
def test_zoom_transform(
    device, method, input_data_format, output_data_format, dtype, zoom
):
    """Test zoom transformation against reference images."""
    batch_size = 1
    input_np = _get_image_batch(batch_size, data_format=input_data_format)
    input_np *= 255.0
    if zoom == "1x":
        scale = 1.0
    elif zoom == "2x":
        scale = (
            0.5
        )  # The matrix specifies an inverse mapping from output to input image.
    stms = np.repeat(
        [[[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]]], batch_size, axis=0
    )
    image = _apply_stm_on_images(
        device=device,
        images=input_np,
        stms=stms,
        method=method,
        input_data_format=input_data_format,
        output_data_format=output_data_format,
        output_dtype=dtype,
    )
    # Drop batch dimension.
    image = image[0]

    # Convert to HWC.
    if output_data_format == DataFormat.CHANNELS_FIRST:
        image = np.transpose(image, [1, 2, 0])

    ref_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_spatial_transform_ref"
    )
    test_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_spatial_transform"
    )

    if debug_save_shape_images:
        try:
            os.mkdir(test_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        file_name = (
            "test_zoom_transform_"
            + str(input_data_format)
            + "_"
            + str(output_data_format)
        )
        file_name += "_cpu" if device == "/cpu" else "_gpu"
        if dtype == np.uint8:
            file_name += "_uint8_"
        elif dtype == np.float16:
            file_name += "_fp16_"
        else:
            file_name += "_fp32_"
        file_name += zoom + "_" + method + ".png"
        debug_im = Image.fromarray(image.astype(np.uint8))
        debug_im.save("%s/%s" % (test_dir, file_name))

    # Load reference image.
    file_name = "test_zoom_transform_" + zoom + "_" + method + ".png"
    ref_image = Image.open("%s/%s" % (ref_dir, file_name))
    ref_image = np.array(ref_image)

    # Compare and assert that test images match reference.
    # Note that there might be slight differences depending on whether the code
    # is run on CPU or GPU, or between different GPUs, CUDA versions, TF versions,
    # etc. We may need to change this assertion to allow some tolerance. Before
    # doing that, please check the generated images to distinguish bugs from
    # small variations.
    squared_diff = np.square(np.subtract(ref_image, image.astype(np.uint8)))
    assert np.sum(squared_diff) < 200


_xy_offsets = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


@pytest.mark.parametrize("device", device_list)
@pytest.mark.parametrize("x, y", _xy_offsets)
def test_translate_out_of_canvas(device, x, y, batch_size=2):
    """Test translating the image outside of the canvas visible area."""
    width, height = 256, 256
    stms = np.repeat(
        [[[1, 0, 0], [0, 1, 0], [x * width, y * height, 1]]], batch_size, axis=0
    )
    _empty_canvas_checker(device, stms)


@pytest.mark.parametrize("device", device_list)
@pytest.mark.parametrize("x, y", [(-1, -1), (1, -1), (-1, 1)])
def test_flip_out_of_canvas(device, x, y, batch_size=2):
    """Test flipping the image outside of the canvas visible area."""
    stms = np.repeat([np.diag([x, y, 1])], batch_size, axis=0)
    _empty_canvas_checker(device, stms)


@pytest.mark.parametrize("device", device_list)
@pytest.mark.parametrize(
    "input_data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize(
    "output_data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize("roundtrip", ["flip", "transpose"])
@pytest.mark.parametrize("method", ["nearest", "bilinear", "bicubic"])
def test_roundtrip(
    device, roundtrip, method, input_data_format, output_data_format, batch_size=2
):
    """Test a that a xy-flip-move roundtrip results in the same image."""
    input_np = _get_image_batch(batch_size, data_format=input_data_format)

    if input_data_format == DataFormat.CHANNELS_LAST:
        width, height = input_np.shape[2], input_np.shape[1]
    else:
        width, height = input_np.shape[3], input_np.shape[2]

    if roundtrip == "flip":
        stms = np.repeat(
            [[[-1, 0, 0], [0, -1, 0], [width, height, 1]]], batch_size, axis=0
        )
    elif roundtrip == "transpose":
        stms = np.repeat([[[0, 1, 0], [1, 0, 0], [0, 0, 1]]], batch_size, axis=0)
    else:
        raise ValueError("Unknown roundtrip %s" % roundtrip)

    # First full flip and move
    output_np = _apply_stm_on_images(
        device=device,
        images=input_np,
        stms=stms,
        method="bilinear",
        input_data_format=input_data_format,
        output_data_format=output_data_format,
    )

    # Check this one is significantly different
    if input_data_format != output_data_format:
        if input_data_format == DataFormat.CHANNELS_FIRST:
            axes = [0, 2, 3, 1]
        else:
            axes = [0, 3, 1, 2]
        check_input_np = np.transpose(input_np, axes=axes)
    else:
        check_input_np = input_np
    mean_abs_diff = np.mean(np.abs(output_np - check_input_np))
    assert mean_abs_diff > 0.1, (
        "input array it not significantly different after %s." % roundtrip
    )

    # Final full flip and move should result in the same as initally
    output_np = _apply_stm_on_images(
        device=device,
        images=output_np,
        stms=stms,
        method="bilinear",
        input_data_format=output_data_format,
        output_data_format=input_data_format,
    )

    if method in ("nearest", "bilinear"):
        # Nearest and bilinear should result in exact values
        np.testing.assert_array_equal(
            input_np,
            output_np,
            err_msg="input array changed after " "%s roundtrip." % roundtrip,
        )
    elif method == "bicubic":
        # bicubic does not result in exact values upon identity transform (expected)
        mean_abs_diff = np.mean(np.abs(output_np - input_np))
        assert mean_abs_diff < 0.02, (
            "input array changed after %s roundtrip." % roundtrip
        )
    else:
        raise ValueError("Unknown method %s" % method)


@pytest.mark.parametrize("device", device_list)
@pytest.mark.parametrize(
    "data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
# Note: bicubic blurs the image so it's not tested here.
@pytest.mark.parametrize("method", ["nearest", "bilinear"])
@pytest.mark.parametrize("num_channels", [3, 1, 5])
def test_4x90_identity(device, data_format, method, num_channels, batch_size=2):
    """Test that 4 rotations of 90 degrees yields the same image."""
    input_np = _get_image_batch(batch_size, data_format=data_format)
    # Duplicate number of channels from three to six.
    channels_dim = 1 if data_format == DataFormat.CHANNELS_FIRST else 3
    input_np = np.concatenate([input_np, input_np * 0.5], axis=channels_dim)
    # Slice to desired number of channels.
    if data_format == DataFormat.CHANNELS_FIRST:
        input_np = input_np[:, 0:num_channels, :, :]
        input_height = input_np.shape[2]
    else:
        input_np = input_np[:, :, :, 0:num_channels]
        input_height = input_np.shape[1]

    theta = np.pi / 2

    stms = np.repeat(
        [
            [
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [input_height, 0, 1],
            ]
        ],
        batch_size,
        axis=0,
    )
    output_np = input_np
    for _ in range(4):
        output_np = _apply_stm_on_images(
            device=device,
            images=output_np,
            stms=stms,
            method=method,
            input_data_format=data_format,
            output_data_format=data_format,
        )
    np.testing.assert_array_equal(
        input_np,
        output_np,
        err_msg="input array changed after " "four 90 degree rotations.",
    )


def test_bad_method(method="foo"):
    """Test whether unsupported interpolation methods will raise errors as expected."""
    stms = np.repeat([np.eye(3)], 2, axis=0)
    with pytest.raises(NotImplementedError):
        _apply_stm_on_default_image("/cpu", stms, method)


shapes_test = [
    ((32, 32), (32, 32)),  # same size
    ((32, 32), (64, 32)),  # larger h
    ((32, 32), (32, 64)),  # larger w
    ((32, 32), (64, 64)),  # larger w, h
    ((32, 32), (16, 16)),
]  # smaller


@pytest.mark.parametrize("device", device_list)
@pytest.mark.parametrize(
    "data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize("shape, new_shape", shapes_test)
def test_different_shape(device, shape, new_shape, data_format):
    """Test that the canvas can be expanded through the provided shape argument."""
    batch_size = 1
    nchannels = 3
    images = np.ones((batch_size,) + shape + (nchannels,), dtype=np.float32)
    # Use NCHW images for channels_first tests.
    if data_format == DataFormat.CHANNELS_FIRST:
        images = np.transpose(images, (0, 3, 1, 2))
    stms = np.repeat([np.eye(3, dtype=np.float32)], batch_size, axis=0)
    out = _apply_stm_on_images(
        device=device,
        images=images,
        stms=stms,
        method="bilinear",
        shape=new_shape,
        input_data_format=data_format,
        output_data_format=data_format,
    )

    # Expected shapes designed for NHWC, perform a transpose for channels_first cases.
    if data_format == DataFormat.CHANNELS_FIRST:
        out = np.transpose(out, (0, 2, 3, 1))

    # Check the output shape is as intended
    np.testing.assert_array_equal(new_shape, out.shape[1:3])

    # Check that the input image sum is equal to its number of pixels * channels
    assert images.sum() == (shape[0] * shape[1] * nchannels)

    # Check that the minimum shape dicatates the amount of activates pixels * channels
    # this works becaues our background is zeroed out so has no influence when the
    # canvas is larger.
    min_shape = np.min([new_shape, shape], axis=0)
    assert out.sum() == (min_shape[0] * min_shape[1] * nchannels)


@pytest.mark.parametrize("device", device_list)
@pytest.mark.parametrize(
    "data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
def test_perspective_transform(device, data_format):
    """Test the perspective transform.

    This is quite hard to test, so currently just testing the shapes are correct, and a
    perspective 'roundtrip' can be performed.
    """
    MAX_MABS_DIFF = 0.02
    batch_size = 2
    nchannels = 3
    new_shape = (384, 384)
    full_shape = (
        (batch_size,) + new_shape + (nchannels,)
        if data_format == DataFormat.CHANNELS_LAST
        else (batch_size, nchannels) + new_shape
    )

    stm1 = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [128, 128, 1.0]], dtype=np.float32
    )

    stm2 = np.array(
        [[1.0, 0.0, 0.001], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )

    stm3 = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-164, -164, 1.0]], dtype=np.float32
    )

    stm = np.matmul(stm2, stm1)
    stm = np.matmul(stm3, stm)

    stms = np.repeat([stm], batch_size, axis=0)

    images = _get_image_batch(batch_size, data_format=data_format)

    input_dtype = np.float32

    with tf.device(device):
        stm_op = SpatialTransform(
            method="bilinear", background_value=0.0, data_format=data_format
        )
        # Note: placeholders are needed to make TensorFlow respect device placement.
        placeholder_images = tf.compat.v1.placeholder(
            shape=images.shape, dtype=input_dtype
        )
        placeholder_stms = tf.compat.v1.placeholder(dtype=tf.float32)
        images_transformed = stm_op(
            placeholder_images, placeholder_stms, shape=new_shape
        )

        # Test the new shape is correct.
        np.testing.assert_array_equal(full_shape, images_transformed.shape)

        # Perform the inverted stm, that should result in the original images.
        # Note that such a roundtrip is (obviously) a lossy; it loses some fidelity.
        inv_stm = np.linalg.inv(stms)
        shape = (
            images.shape[1:3]
            if data_format == DataFormat.CHANNELS_LAST
            else images.shape[2:]
        )

        placeholder_inv_stms = tf.compat.v1.placeholder(dtype=tf.float32)
        fetches = stm_op(images_transformed, placeholder_inv_stms, shape=shape)

        sess = tf.compat.v1.Session()
        images_roundtrip = sess.run(
            fetches,
            feed_dict={
                placeholder_images: images,
                placeholder_stms: stms,
                placeholder_inv_stms: inv_stm,
            },
        )

    # Check that the roundtrip is the same as the input image.
    mean_abs_diff = np.mean(np.abs(images_roundtrip - images))
    assert mean_abs_diff < MAX_MABS_DIFF


def _make_dense_coordinates(batch_size, num_time_steps, num_shapes, num_vertices):
    """Construct a dense coordinate array."""
    shape = [num_vertices, 2]
    if num_shapes is not None:
        shape = [num_shapes] + shape
    if num_time_steps is not None:
        shape = [num_time_steps] + shape
    if batch_size is not None:
        shape = [batch_size] + shape
    return np.random.random_integers(low=0, high=9, size=shape).astype(np.float32)


def _make_sparse_coordinates(batch_size, num_time_steps, shape_dim):
    """Construct a sparse coordinate array."""
    indices = []

    # Construct 2D-5D indices depending on whether batch, time, and shape dimensions are
    # present. Batch and time dimension sizes are fixed, but shape and vertices sizes
    # vary. Some of the sizes are zero.
    def make_vertices(prefix, num_vertices):
        for v in range(num_vertices):
            indices.append(prefix + [v, 0])
            indices.append(prefix + [v, 1])

    def make_shapes(prefix, num_shapes):
        if shape_dim:
            num_vertices_per_shape = np.random.permutation(num_shapes) * 2
            for s, num_vertices in enumerate(num_vertices_per_shape):
                make_vertices(prefix + [s], num_vertices)
        else:
            make_vertices(prefix, 4)

    def make_time_steps(prefix, num_time_steps):
        if num_time_steps is not None:
            num_shapes_per_time_step = np.random.permutation(num_time_steps) * 2
            for t, num_shapes in enumerate(num_shapes_per_time_step):
                make_shapes(prefix + [t], num_shapes)
        else:
            make_shapes(prefix, 6)

    if batch_size is not None:
        for b in range(batch_size):
            make_time_steps([b], num_time_steps)
    else:
        make_time_steps([], num_time_steps)

    # Compute dense shape.
    dense_shape = np.amax(np.array(indices), axis=0)  # Max index along each axis.
    dense_shape += np.ones_like(
        dense_shape
    )  # + 1 = Max number of indices along each axis.

    # Values are random integers between 0 and 9.
    values = np.random.random_integers(low=0, high=9, size=len(indices)).astype(
        np.float32
    )

    return values, indices, dense_shape


@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("num_time_steps", [None, 3])
@pytest.mark.parametrize("shape_dim", [False, True])
@pytest.mark.parametrize("sparse", [True, False])
def test_identity_coordinate_transform(batch_size, num_time_steps, shape_dim, sparse):
    """Identity coordinate transform test."""
    np.random.seed(42)

    # Dimensions = [Batch, Time, Shape, Vertex, Coordinate]
    if sparse is False:
        num_shapes = 6 if shape_dim else None
        num_vertices = 4
        coordinates = _make_dense_coordinates(
            batch_size, num_time_steps, num_shapes, num_vertices
        )
        coordinates_tf = tf.constant(coordinates, dtype=tf.float32)
    else:
        values, indices, dense_shape = _make_sparse_coordinates(
            batch_size, num_time_steps, shape_dim
        )
        coordinates_tf = tf.SparseTensor(
            values=tf.constant(values),
            indices=tf.constant(indices, dtype=tf.int64),
            dense_shape=dense_shape,
        )

    # Construct processor.
    op = PolygonTransform(invert_stm=True)

    # Construct transformation matrix.
    stm = np.eye(3, dtype=np.float32)
    if batch_size is not None:
        stm = np.tile(stm, [batch_size, 1, 1])

    # Do the operation.
    coordinates_transformed = op(polygons=coordinates_tf, stm=stm)
    coordinates_transformed_np = tf.compat.v1.Session().run(coordinates_transformed)

    if sparse:
        np.testing.assert_array_equal(values, coordinates_transformed_np.values)
        np.testing.assert_array_equal(indices, coordinates_transformed_np.indices)
        np.testing.assert_array_equal(
            dense_shape, coordinates_transformed_np.dense_shape
        )
    else:
        np.testing.assert_array_equal(coordinates, coordinates_transformed_np)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("perspective", [False, True])
def test_batch_transform(sparse, perspective):
    """Batch coordinate transform test."""
    batch_size = 8

    np.random.seed(42)

    # Dimensions = [Batch, Vertex, Coordinate]
    if sparse is False:
        coordinates = _make_dense_coordinates(
            batch_size, num_time_steps=None, num_shapes=None, num_vertices=4
        )
        coordinates_tf = tf.constant(coordinates, dtype=tf.float32)
    else:
        values, indices, dense_shape = _make_sparse_coordinates(
            batch_size, num_time_steps=None, shape_dim=False
        )
        coordinates_tf = tf.SparseTensor(
            values=tf.constant(values),
            indices=tf.constant(indices, dtype=tf.int64),
            dense_shape=dense_shape,
        )
    # Construct processor.
    op = PolygonTransform(invert_stm=False)

    # Construct transformation matrix.
    stms = []
    for b in range(batch_size):
        stm = np.eye(3, dtype=np.float32)
        stm[0][0] = float(b + 1)  # Scale x coordinates by b+1.
        if perspective:
            stm[2][2] = 0.5  # Set z coordinate to 0.5.
        stms.append(stm)
    stms = tf.constant(np.stack(stms))

    # Do the operation.
    coordinates_transformed = op(polygons=coordinates_tf, stm=stms)
    coordinates_transformed_np = tf.compat.v1.Session().run(coordinates_transformed)

    if sparse:
        np.testing.assert_array_equal(indices, coordinates_transformed_np.indices)
        np.testing.assert_array_equal(
            dense_shape, coordinates_transformed_np.dense_shape
        )
        assert values.shape == coordinates_transformed_np.values.shape
        for i, index in enumerate(indices):
            b = index[0]
            is_x_coordinate = index[2] == 0
            expected = values[i]
            if is_x_coordinate:
                expected *= float(b + 1)
            if perspective:
                expected *= 2.0  # Due to perspective divide by z coord (0.5).
            assert coordinates_transformed_np.values[i] == expected
    else:
        assert coordinates.shape == coordinates_transformed_np.shape
        for b in range(coordinates.shape[0]):
            for v in range(coordinates.shape[1]):
                expected_x = float(b + 1) * coordinates[b][v][0]
                expected_y = coordinates[b][v][1]
                if perspective:
                    expected_x *= 2.0
                    expected_y *= 2.0
                assert coordinates_transformed_np[b][v][0] == expected_x
                assert coordinates_transformed_np[b][v][1] == expected_y


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("batch_size", [None, 4])
@pytest.mark.parametrize("dense_shape", [(0, 0), (0, 2), (3, 0)])
def test_empty_coordinates(sparse, batch_size, dense_shape):
    """Test that empty coordinates tensor works."""
    # Dimensions = [Batch, Vertices, Coordinates].
    # SparseTensors can have pretty funky degeneracies, so test that any combination of
    # zero length vertices and coords dimensions works.
    if batch_size is not None:
        dense_shape = (batch_size,) + dense_shape

    if sparse:
        # Indices has zero rows of data.
        indices_shape = (0, len(dense_shape))
        coordinates_tf = tf.SparseTensor(
            values=tf.zeros(shape=[0], dtype=tf.float32),
            indices=tf.zeros(shape=indices_shape, dtype=tf.int64),
            dense_shape=dense_shape,
        )
    else:
        coordinates_tf = tf.zeros(shape=dense_shape, dtype=tf.float32)

    # Construct processor.
    op = PolygonTransform(invert_stm=True)

    # Construct transformation matrix.
    stm = np.eye(3, dtype=np.float32)
    if batch_size is not None:
        stm = np.tile(stm, [batch_size, 1, 1])

    # Do the operation.
    coordinates_transformed = op(polygons=coordinates_tf, stm=stm)
    coordinates_transformed_np = tf.compat.v1.Session().run(coordinates_transformed)

    if sparse:
        assert coordinates_transformed_np.values.shape == (0,)
        assert coordinates_transformed_np.indices.shape == indices_shape
        np.testing.assert_array_equal(
            coordinates_transformed_np.dense_shape, dense_shape
        )
    else:
        np.testing.assert_array_equal(coordinates_transformed_np.shape, dense_shape)


def test_error_checks():
    """Test error checks."""
    # Both data_format and either input_data_format or output_data_format given.
    with pytest.raises(ValueError):
        SpatialTransform(
            data_format=DataFormat.CHANNELS_FIRST,
            input_data_format=DataFormat.CHANNELS_FIRST,
        )

    with pytest.raises(ValueError):
        SpatialTransform(
            data_format=DataFormat.CHANNELS_FIRST,
            output_data_format=DataFormat.CHANNELS_FIRST,
        )

    # Input_data_format given, but output_data_format is missing.
    with pytest.raises(ValueError):
        SpatialTransform(input_data_format=DataFormat.CHANNELS_FIRST)

    # Output_data_format given, but input_data_format is missing.
    with pytest.raises(ValueError):
        SpatialTransform(output_data_format=DataFormat.CHANNELS_FIRST)

    # Invalid data format.
    with pytest.raises(NotImplementedError):
        SpatialTransform(data_format="weird_data_format")

    # Unknown filtering method.
    with pytest.raises(NotImplementedError):
        SpatialTransform(method="unknown")


@pytest.mark.parametrize(
    "input_np",
    [
        np.zeros([2, 1, 16, 12], np.int16),  # Unsupported data type.
        np.zeros([16, 12, 3], np.float32),  # Too few dimensions.
        np.zeros([2, 3, 16, 12, 3], np.float32),  # Too many dimensions.
        np.zeros(
            [1, 3, 16, 12], np.float32
        ),  # Number of images does not match number of stms.
    ],
)
def test_invalid_input_tensors(input_np):
    """Test invalid input tensors."""
    with pytest.raises(Exception):
        stm_op = SpatialTransform(data_format=DataFormat.CHANNELS_FIRST)
        batch_size = 2
        stms = np.repeat([np.eye(3)], batch_size, axis=0)
        fetches = stm_op(tf.constant(input_np), tf.constant(stms, dtype=tf.float32))
        sess = tf.compat.v1.Session()
        sess.run(fetches)


@pytest.mark.parametrize(
    "shape",
    [
        np.zeros([1], np.float32),  # Too few entries.
        np.zeros([3], np.float32),  # Too many entries.
        np.zeros([16, 12], np.float32),  # Wrong rank.
    ],
)
def test_invalid_shape_tensors(shape):
    """Test invalid shape tensors."""
    with pytest.raises(Exception):
        stm_op = SpatialTransform(data_format=DataFormat.CHANNELS_FIRST)
        batch_size = 2
        stms = np.repeat([np.eye(3)], batch_size, axis=0)
        fetches = stm_op(
            tf.zeros([2, 3, 16, 12], dtype=tf.float32),
            tf.constant(stms, dtype=tf.float32),
            shape,
        )
        sess = tf.compat.v1.Session()
        sess.run(fetches)


@pytest.mark.parametrize(
    "stms",
    [
        np.zeros([2, 2, 3], np.float32),  # Wrong number of rows.
        np.zeros([2, 3, 4], np.float32),  # Wrong number of columns.
        np.zeros([2, 9], np.float32),  # Wrong dimensionality.
        np.zeros([1, 3, 3], np.float32),  # Wrong batch size.
    ],
)
def test_invalid_stms(stms):
    """Test invalid stms."""
    with pytest.raises(Exception):
        input_np = np.zeros([2, 16, 12, 3], np.float32)
        stm_op = SpatialTransform(data_format=DataFormat.CHANNELS_FIRST)
        fetches = stm_op(tf.constant(input_np), tf.constant(stms, dtype=tf.float32))
        sess = tf.compat.v1.Session()
        sess.run(fetches)


@pytest.mark.parametrize(
    "images, input_data_format, output_data_format, shape, expected_shape",
    [
        (
            np.zeros([2, 3, 16, 12], np.float32),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_FIRST,
            None,
            [2, 3, 16, 12],
        ),
        (
            np.zeros([2, 4, 16, 12], np.float32),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_LAST,
            None,
            [2, 16, 12, 4],
        ),
        (
            np.zeros([2, 16, 12, 3], np.float32),
            DataFormat.CHANNELS_LAST,
            DataFormat.CHANNELS_LAST,
            None,
            [2, 16, 12, 3],
        ),
        (
            np.zeros([2, 16, 12, 1], np.float32),
            DataFormat.CHANNELS_LAST,
            DataFormat.CHANNELS_FIRST,
            None,
            [2, 1, 16, 12],
        ),
        (
            tf.zeros(shape=[2, 3, 16, 12], dtype=tf.uint8),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_LAST,
            None,
            [2, 16, 12, 3],
        ),
        (
            tf.compat.v1.placeholder(shape=[2, 3, 16, 12], dtype=tf.float16),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_FIRST,
            None,
            [2, 3, 16, 12],
        ),
        # Cases where the shape is completely or partially unknown.
        (
            tf.compat.v1.placeholder(dtype=tf.float32),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_FIRST,
            None,
            [None, None, None, None],
        ),
        (
            tf.compat.v1.placeholder(shape=[2, 5, 16, None], dtype=tf.float16),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_FIRST,
            None,
            [2, 5, 16, None],
        ),
        (
            tf.compat.v1.placeholder(shape=[2, None, 16, 12], dtype=tf.uint8),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_LAST,
            None,
            [2, 16, 12, None],
        ),
        (
            tf.compat.v1.placeholder(shape=[None, 3, 16, 12], dtype=tf.uint8),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_LAST,
            None,
            [None, 16, 12, 3],
        ),
        (
            tf.compat.v1.placeholder(shape=[2, None, None, 3], dtype=tf.float32),
            DataFormat.CHANNELS_LAST,
            DataFormat.CHANNELS_FIRST,
            None,
            [2, 3, None, None],
        ),
        (
            tf.compat.v1.placeholder(shape=[None, None, None, None], dtype=tf.uint8),
            DataFormat.CHANNELS_LAST,
            DataFormat.CHANNELS_FIRST,
            None,
            [None, None, None, None],
        ),
        # Cases where output shape is given.
        (
            np.zeros([3, 2, 16, 12], np.float32),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_FIRST,
            [24, 18],
            [3, 2, 24, 18],
        ),
        (
            tf.compat.v1.placeholder(shape=[2, 3, None, None], dtype=tf.float16),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_FIRST,
            [24, 18],
            [2, 3, 24, 18],
        ),
        (
            tf.compat.v1.placeholder(shape=[2, None, None, None], dtype=tf.float16),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_FIRST,
            [24, 18],
            [2, None, 24, 18],
        ),
        (
            tf.compat.v1.placeholder(shape=[None, None, None, None], dtype=tf.float16),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_FIRST,
            [24, 18],
            [None, None, 24, 18],
        ),
        (
            tf.compat.v1.placeholder(dtype=tf.float16),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_FIRST,
            [24, 18],
            [None, None, 24, 18],
        ),
        (
            tf.compat.v1.placeholder(dtype=tf.float16),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_FIRST,
            [24, -1],
            [None, None, 24, None],
        ),
        (
            np.zeros([2, 3, 16, 12], np.float32),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_FIRST,
            tf.compat.v1.placeholder(shape=[1, 1], dtype=tf.int32),
            [2, 3, None, None],
        ),
    ],
)
def test_spatial_transform_shape_inference(
    images, input_data_format, output_data_format, shape, expected_shape
):
    """Test shape inference."""
    stm_op = SpatialTransform(
        input_data_format=input_data_format, output_data_format=output_data_format
    )
    output = stm_op(images, tf.constant(0.0, tf.float32), shape)
    assert expected_shape == output.shape.as_list()


def test_serialization_and_deserialization():
    """Test that it is a MaglevObject that can be serialized and deserialized."""
    stm_op = SpatialTransform(
        method="bilinear",
        background_value=0.0,
        input_data_format=DataFormat.CHANNELS_LAST,
        output_data_format=DataFormat.CHANNELS_LAST,
    )
    stm_op_dict = stm_op.serialize()
    deserialized_stm_op = deserialize_tao_object(stm_op_dict)
    assert stm_op.method == deserialized_stm_op.method
    assert stm_op.background_value == deserialized_stm_op.background_value
    assert stm_op.output_data_format == deserialized_stm_op.output_data_format
    assert stm_op.input_data_format == deserialized_stm_op.input_data_format
    assert stm_op.output_dtype == deserialized_stm_op.output_dtype
