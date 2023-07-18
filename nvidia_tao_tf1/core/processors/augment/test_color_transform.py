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

import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import deserialize_tao_object
from nvidia_tao_tf1.core.processors import ColorTransform
from nvidia_tao_tf1.core.types import DataFormat


device_list = ["/cpu", "/gpu"]


def get_random_image(
    batch_size,
    start=0.0,
    stop=1.0,
    data_format=DataFormat.CHANNELS_LAST,
    dtype=np.float32,
    num_channels=3,
):
    """Create a batch of images, with values within a linespace, that are then randomly shuffled."""
    shape = (batch_size, 16, 64, num_channels)
    if data_format == DataFormat.CHANNELS_FIRST:
        shape = (batch_size, num_channels, 16, 64)

    images = np.linspace(
        start, stop, batch_size * 16 * 64 * num_channels, dtype=dtype
    ).reshape(shape)
    return np.random.permutation(images)


def _apply_ctm_on_images(
    device,
    images,
    ctms,
    min_clip=0.0,
    max_clip=1.0,
    data_format=DataFormat.CHANNELS_LAST,
    input_dtype=np.float32,
    output_dtype=None,
):
    """Run ColorTransform op."""
    with tf.device(device):
        ctm_op = ColorTransform(
            min_clip=min_clip,
            max_clip=max_clip,
            data_format=data_format,
            output_dtype=output_dtype,
        )
        # Note: placeholders are needed to make TensorFlow respect device placement.
        placeholder_images = tf.compat.v1.placeholder(dtype=input_dtype)
        placeholder_ctms = tf.compat.v1.placeholder(dtype=tf.float32)
        fetches = ctm_op(placeholder_images, placeholder_ctms)
        sess = tf.compat.v1.Session()
        return sess.run(
            fetches, feed_dict={placeholder_images: images, placeholder_ctms: ctms}
        )


@pytest.mark.parametrize("device", device_list)
@pytest.mark.parametrize("min_clip", [0.0, 1.0])
@pytest.mark.parametrize("max_clip", [1.0, 8.0, 255.0])
@pytest.mark.parametrize("input_dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("output_dtype", [np.uint8, np.float16, np.float32, None])
def test_clipping_and_type_cast(
    device, min_clip, max_clip, input_dtype, output_dtype, batch_size=2
):
    """Test color clipping and type casting."""
    ctms = np.repeat([np.eye(4)], batch_size, axis=0)
    input_np = get_random_image(batch_size, start=-512, stop=512, dtype=input_dtype)
    output_np = _apply_ctm_on_images(
        device,
        input_np,
        ctms,
        min_clip,
        max_clip,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )
    assert (
        output_np.min() >= min_clip
    ), "Minimal value lower than specified 'min_clip' value."
    assert (
        output_np.max() <= max_clip
    ), "Maximum value lower than specified 'max_clip' value."
    if output_dtype is None:
        assert output_np.dtype == input_dtype
    else:
        assert output_np.dtype == output_dtype


@pytest.mark.parametrize("device", device_list)
@pytest.mark.parametrize("max_value", [1.0, 255.0])
@pytest.mark.parametrize("identity_scale", [1.0, 255.0])
def test_identity_and_scale(device, max_value, identity_scale, batch_size=2):
    """Test elements over the diagonal leaving the image unchanged or only linearly scale them."""
    ctms = np.repeat([np.eye(4)], batch_size, axis=0) * identity_scale
    input_np = get_random_image(batch_size, start=0.0, stop=max_value)
    output_np = _apply_ctm_on_images(
        device, input_np, ctms, min_clip=0.0, max_clip=max_value * identity_scale
    )
    np.testing.assert_allclose(
        input_np * identity_scale,
        output_np,
        rtol=1e-6,
        err_msg="input array changed after application of an identity color "
        " transformation matrix.",
    )


@pytest.mark.parametrize("device", device_list)
@pytest.mark.parametrize("c1", [0, 0.1])
@pytest.mark.parametrize("c2", [0, 0.2])
@pytest.mark.parametrize("c3", [0, 0.3])
@pytest.mark.parametrize(
    "data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
def test_value_offset(device, c1, c2, c3, data_format, batch_size=2):
    """Test adding offsets to either of the channels, and assert that they shift those channels."""
    ctm = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [c1, c2, c3, 1]], dtype=np.float32
    )
    ctms = np.repeat([ctm], batch_size, axis=0)
    input_np = get_random_image(
        batch_size, start=0.0, stop=1.0, data_format=data_format
    )
    output_np = _apply_ctm_on_images(
        device, input_np, ctms, min_clip=0.0, max_clip=2.0, data_format=data_format
    )
    axis = (0, 1, 2) if data_format == DataFormat.CHANNELS_LAST else (0, 2, 3)
    mean_diff = np.mean(output_np - input_np, axis=axis)
    expected_mean_diff = (c1, c2, c3)
    np.testing.assert_allclose(mean_diff, expected_mean_diff, rtol=1e-4)


@pytest.mark.parametrize("device", device_list)
@pytest.mark.parametrize(
    "data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_value_roundtrip(device, data_format, dtype, batch_size=2):
    """Shift channels by 1 channel, 3 times; then assert that it results in the same tensor."""
    ctm = np.array(
        [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    ctms = np.repeat([ctm], batch_size, axis=0)
    input_np = get_random_image(batch_size, data_format=data_format, dtype=dtype)

    # Test that the first 2 channels shifts (out of 3) result in a different tensor
    output_np = input_np
    for _ in range(2):
        output_np = _apply_ctm_on_images(
            device, output_np, ctms, data_format=data_format, input_dtype=dtype
        )
        np.testing.assert_equal(np.any(np.not_equal(output_np, input_np)), True)

    # The 3rd out of 3 channel shifts should result in the same image as the input image
    output_np = _apply_ctm_on_images(
        device, output_np, ctms, data_format=data_format, input_dtype=dtype
    )
    assert output_np.dtype == input_np.dtype

    np.testing.assert_array_equal(
        input_np,
        output_np,
        err_msg="input array changed after application of shifting 3 "
        "color channels 3 times.",
    )


@pytest.mark.parametrize("device", device_list)
@pytest.mark.parametrize(
    "input_data_format", [DataFormat.CHANNELS_FIRST, DataFormat.CHANNELS_LAST]
)
def test_transpose(device, input_data_format, batch_size=2):
    """Test transposing between channels first and channels last."""
    ctms = np.repeat([np.eye(4)], batch_size, axis=0)
    input_np = get_random_image(
        batch_size, start=0.0, stop=255.0, data_format=input_data_format
    )
    if input_data_format == DataFormat.CHANNELS_FIRST:
        output_data_format = DataFormat.CHANNELS_LAST
        axes = [0, 2, 3, 1]
    else:
        output_data_format = DataFormat.CHANNELS_FIRST
        axes = [0, 3, 1, 2]

    input_shape_transposed = []
    for i in range(4):
        input_shape_transposed.append(input_np.shape[axes[i]])

    with tf.device(device):
        ctm_op = ColorTransform(
            min_clip=0.0,
            max_clip=255.0,
            input_data_format=input_data_format,
            output_data_format=output_data_format,
        )
        images = tf.constant(input_np, dtype=tf.float32)
        fetches = ctm_op(images, tf.constant(ctms, dtype=tf.float32))
        # Check shape inference.
        np.testing.assert_array_equal(fetches.shape, input_shape_transposed)
        sess = tf.compat.v1.Session()
        output_np = sess.run(fetches)

    # Check output shape.
    np.testing.assert_array_equal(output_np.shape, input_shape_transposed)

    transposed_input_np = np.transpose(input_np, axes=axes)
    np.testing.assert_array_equal(
        transposed_input_np,
        output_np,
        err_msg="input array changed after application of conversion from CHANNELS_LAST to "
        "CHANNELS_FIRST.",
    )


def test_error_checks():
    """Test error checks."""
    # min_clip > max_clip.
    with pytest.raises(ValueError):
        ColorTransform(min_clip=1.0, max_clip=0.0)

    # Both data_format and either input_data_format or output_data_format given.
    with pytest.raises(ValueError):
        ColorTransform(
            data_format=DataFormat.CHANNELS_FIRST,
            input_data_format=DataFormat.CHANNELS_FIRST,
        )

    with pytest.raises(ValueError):
        ColorTransform(
            data_format=DataFormat.CHANNELS_FIRST,
            output_data_format=DataFormat.CHANNELS_FIRST,
        )

    # Input_data_format given, but output_data_format is missing.
    with pytest.raises(ValueError):
        ColorTransform(input_data_format=DataFormat.CHANNELS_FIRST)

    # Output_data_format given, but input_data_format is missing.
    with pytest.raises(ValueError):
        ColorTransform(output_data_format=DataFormat.CHANNELS_FIRST)

    # Invalid data format.
    with pytest.raises(NotImplementedError):
        ColorTransform(data_format="weird_data_format")


@pytest.mark.parametrize(
    "input_np",
    [
        np.zeros([2, 16, 12, 3], np.float32),  # Wrong channel order.
        np.zeros([2, 4, 16, 12], np.float32),  # Too many color channels.
        np.zeros([2, 1, 16, 12], np.int16),  # Unsupported data type.
        np.zeros([16, 12, 3], np.float32),  # Too few dimensions.
        np.zeros([2, 3, 16, 12, 3], np.float32),  # Too many dimensions.
        np.zeros(
            [1, 3, 16, 12], np.float32
        ),  # Number of images does not match number of ctms.
    ],
)
def test_invalid_input_tensors(input_np):
    """Test invalid input tensors."""
    with pytest.raises(Exception):
        ctm_op = ColorTransform(data_format=DataFormat.CHANNELS_FIRST)
        batch_size = 2
        ctms = np.repeat([np.eye(4)], batch_size, axis=0)
        fetches = ctm_op(tf.constant(input_np), tf.constant(ctms, dtype=tf.float32))
        sess = tf.compat.v1.Session()
        sess.run(fetches)


@pytest.mark.parametrize(
    "ctms",
    [
        np.zeros([2, 3, 4], np.float32),  # Wrong number of rows.
        np.zeros([2, 5, 4], np.float32),  # Wrong number of columns.
        np.zeros([2, 16], np.float32),  # Wrong dimensionality.
        np.zeros([1, 4, 4], np.float32),  # Wrong batch size.
    ],
)
def test_invalid_ctms(ctms):
    """Test invalid ctms."""
    with pytest.raises(Exception):
        input_np = np.zeros([2, 16, 12, 3], np.float32)
        ctm_op = ColorTransform(data_format=DataFormat.CHANNELS_FIRST)
        fetches = ctm_op(tf.constant(input_np), tf.constant(ctms, dtype=tf.float32))
        sess = tf.compat.v1.Session()
        sess.run(fetches)


@pytest.mark.parametrize(
    "images, input_data_format, output_data_format, expected_shape",
    [
        (
            np.zeros([2, 3, 16, 12], np.float32),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_FIRST,
            [2, 3, 16, 12],
        ),
        (
            np.zeros([2, 3, 16, 12], np.float32),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_LAST,
            [2, 16, 12, 3],
        ),
        (
            np.zeros([2, 16, 12, 3], np.float32),
            DataFormat.CHANNELS_LAST,
            DataFormat.CHANNELS_LAST,
            [2, 16, 12, 3],
        ),
        (
            np.zeros([2, 16, 12, 3], np.float32),
            DataFormat.CHANNELS_LAST,
            DataFormat.CHANNELS_FIRST,
            [2, 3, 16, 12],
        ),
        (
            tf.zeros(shape=[2, 3, 16, 12], dtype=tf.uint8),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_LAST,
            [2, 16, 12, 3],
        ),
        (
            tf.compat.v1.placeholder(shape=[2, 3, 16, 12], dtype=tf.float16),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_FIRST,
            [2, 3, 16, 12],
        ),
        # Cases where the shape is completely or partially unknown.
        (
            tf.compat.v1.placeholder(dtype=tf.float32),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_FIRST,
            [None, None, None, None],
        ),
        (
            tf.compat.v1.placeholder(shape=[2, 3, 16, None], dtype=tf.float16),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_FIRST,
            [2, 3, 16, None],
        ),
        (
            tf.compat.v1.placeholder(shape=[2, None, 16, 12], dtype=tf.uint8),
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_LAST,
            [2, 16, 12, None],
        ),
        (
            tf.compat.v1.placeholder(shape=[2, None, None, 3], dtype=tf.float32),
            DataFormat.CHANNELS_LAST,
            DataFormat.CHANNELS_FIRST,
            [2, 3, None, None],
        ),
        (
            tf.compat.v1.placeholder(shape=[None, None, None, None], dtype=tf.uint8),
            DataFormat.CHANNELS_LAST,
            DataFormat.CHANNELS_FIRST,
            [None, None, None, None],
        ),
    ],
)
def test_color_transform_shape_inference(
    images, input_data_format, output_data_format, expected_shape
):
    """Test shape inference."""
    ctm_op = ColorTransform(
        input_data_format=input_data_format, output_data_format=output_data_format
    )
    output = ctm_op(images, tf.constant(0.0, tf.float32))
    assert expected_shape == output.shape.as_list()


def test_serialization_and_deserialization():
    """Test that it is a MaglevObject that can be serialized and deserialized."""
    ctm_op = ColorTransform(
        input_data_format=DataFormat.CHANNELS_FIRST,
        output_data_format=DataFormat.CHANNELS_FIRST,
    )
    ctm_op_dict = ctm_op.serialize()
    deserialized_ctm_op = deserialize_tao_object(ctm_op_dict)
    assert ctm_op.min_clip == deserialized_ctm_op.min_clip
    assert ctm_op.max_clip == deserialized_ctm_op.max_clip
    assert ctm_op.input_data_format == deserialized_ctm_op.input_data_format
    assert ctm_op.output_data_format == deserialized_ctm_op.output_data_format
    assert ctm_op.output_dtype == deserialized_ctm_op.output_dtype
