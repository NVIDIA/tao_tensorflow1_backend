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
"""Modulus Color Processors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.processors import load_custom_tf_op
from nvidia_tao_tf1.core.processors.processors import Processor
from nvidia_tao_tf1.core.types import data_format as tao_data_format, DataFormat


class ColorTransform(Processor):
    """
    ColorTransform class.

    Args:
        min_clip (float): Minimum color value after transformation.
        max_clip (float): Maximum color value after transformation.
        data_format (string): A string representing the dimension ordering of the input
            images, must be one of 'channels_last' (NHWC) or 'channels_first' (NCHW). If
            specified, input_data_format and output_data_format must be None.
        input_data_format (string): Data format for input. If specified, data_format must be None,
            and output_data_format must be given.
        output_data_format (string): Data format for output. If specified, data_format must be
            None, and input_data_format must be given.
        output_dtype (dtype): Valid values are tf.uint8, tf.float16, tf.float32, None. If None,
            image dtype is used. Note for uint8 output: Image data must be prescaled to [0,255]
            range, and min_clip set to at least 0 and max_clip set to at most 255.
        kwargs (dict): keyword arguments passed to parent class.
    """

    @save_args
    def __init__(
        self,
        min_clip=0.0,
        max_clip=255.0,
        data_format=None,
        input_data_format=None,
        output_data_format=None,
        output_dtype=None,
        **kwargs
    ):
        """__init__ method."""
        if min_clip > max_clip:
            raise ValueError(
                "Min_clip={} is greater than max_clip={}.".format(min_clip, max_clip)
            )

        self.min_clip = min_clip
        self.max_clip = max_clip

        if data_format is not None and (
            input_data_format is not None or output_data_format is not None
        ):
            raise ValueError(
                "When data_format is specified, input_data_format and "
                "output_data_format must be None."
            )

        if input_data_format is not None and output_data_format is None:
            raise ValueError(
                "When input_data_format is specified, output_data_format "
                "must be specified too."
            )

        if output_data_format is not None and input_data_format is None:
            raise ValueError(
                "When output_data_format is specified, input_data_format "
                "must be specified too."
            )

        if (
            data_format is None
            and input_data_format is None
            and output_data_format is None
        ):
            data_format = tao_data_format()

        if data_format is not None:
            input_data_format = data_format
            output_data_format = data_format

        if input_data_format not in [
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_LAST,
        ]:
            raise NotImplementedError(
                "Data format not supported, must be 'channels_first' or "
                "'channels_last', given {}.".format(input_data_format)
            )
        if output_data_format not in [
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_LAST,
        ]:
            raise NotImplementedError(
                "Data format not supported, must be 'channels_first' or "
                "'channels_last', given {}.".format(output_data_format)
            )

        self.output_data_format = output_data_format
        self.input_data_format = input_data_format

        self.output_dtype = output_dtype

        super(ColorTransform, self).__init__(**kwargs)

    def call(self, images, ctms):
        """
        Apply color transformation matrices on images.

        For each pixel, computes (r,g,b,_) = (r,g,b,1) * color_matrix, where _ denotes that
        the result is not used.

        Args:
            images: 4D tensor with shape `(batch_size, channels, height, width)`
                if input_data_format='channels_first', or 4D tensor with shape
                `(batch_size, height, width, channels)` if input_data_format='channels_last'.
                Number of channels must be 3.
            ctms: 3D tensor (batch_size, 4, 4) Color Transformation Matrix per image (N, 4, 4)

        Returns
            4D Tensor after ctm application, with shape `(batch_size, channels, height, width)`
                if output_data_format='channels_first', or 4D tensor with shape
                `(batch_size, height, width, channels)` if output_data_format='channels_last'.
        """
        op = load_custom_tf_op("op_colortransform.so")

        output_dtype = self.output_dtype
        if output_dtype is None:
            output_dtype = images.dtype

        data_formats = {
            DataFormat.CHANNELS_FIRST: "NCHW",
            DataFormat.CHANNELS_LAST: "NHWC",
        }
        input_data_format = data_formats[self.input_data_format]
        output_data_format = data_formats[self.output_data_format]

        transformed_images = op.colortransform(
            images,
            ctms,
            min_clip=self.min_clip,
            max_clip=self.max_clip,
            input_data_format=input_data_format,
            output_data_format=output_data_format,
            output_dtype=output_dtype,
        )
        return transformed_images


def brightness_offset_matrix(offset):
    """
    Form a per-channel brightness offset matrix for transforming RGB images.

    Args:
        offset: tensor(float32) offset per color channel (3,) or a batch of offsets (N, 3).

    Returns:
        fp32 tensor (4, 4), color transformation matrix if offset is not batched. If
        offset is batched, (N, 4, 4).
    """
    offset = tf.cast(tf.convert_to_tensor(value=offset), tf.float32)
    if offset.shape.ndims == 2:
        batch_shape = [tf.shape(input=offset)[0]]
        one = tf.ones(shape=batch_shape + [1], dtype=tf.float32)
    else:
        batch_shape = None
        one = tf.constant([1.0])

    # Attach fourth column to offset: [N, 3] + [N, 1] = [N, 4]
    offset = tf.concat([offset, one], axis=-1)

    # [N, 4] -> [N, 1, 4]
    offset = tf.expand_dims(offset, axis=-2)

    # Construct a [N, 3, 4] identity matrix.
    m = tf.eye(num_rows=3, num_columns=4, batch_shape=batch_shape)

    # Attach offset row: [N, 3, 4] + [N, 1, 4] = [N, 4, 4]
    return tf.concat([m, offset], axis=-2)


def contrast_matrix(contrast, center):
    """
    Form a contrast transformation matrix for RGB images.

    The contrast matrix introduces a scaling around a center point.

    Args:
        contrast: tensor(float32) contrast value (scalar or vector). A value of 0 will keep the
            scaling untouched.
        center: tensor(float32) center value. For 8-bit images this is commonly 127.5, and 0.5 for
            images within the [0,1] range. Scalar or vector.

    Returns:
        fp32 tensor (4, 4), color transformation matrix if contrast and center are scalars. If
        contrast and center are vectors, (len(contrast), 4, 4).
    """
    contrast = tf.cast(tf.convert_to_tensor(value=contrast), tf.float32)
    center = tf.cast(tf.convert_to_tensor(value=center), tf.float32)
    zero = tf.zeros_like(contrast)
    one = tf.ones_like(contrast)
    scale = one + contrast
    bias = -contrast * center
    m = tf.stack(
        [
            scale,
            zero,
            zero,
            zero,
            zero,
            scale,
            zero,
            zero,
            zero,
            zero,
            scale,
            zero,
            bias,
            bias,
            bias,
            one,
        ],
        axis=-1,
    )
    shape = [-1, 4, 4] if contrast.shape.ndims == 1 else [4, 4]
    return tf.reshape(m, shape)


def hue_saturation_matrix(hue, saturation):
    """
    Form a color saturation and hue transformation matrix for RGB images.

    Single matrix transform for both hue and saturation change. Matrix taken from [1].
    Derived by transforming first to HSV, then do the modification, and transform back to RGB.

    Note that perfect conversions between RGB and HSV are non-linear, but we can approximate it
    very well with these linear matrices. If one would truly care about color conversions, one
    would need calibrated images with a known color profile, white-point, etc.

    Args:
        hue: (float) hue rotation in degrees (scalar or vector). A value of 0.0 (modulo 360)
            leaves the hue unchanged.
        saturation: (float) saturation multiplier (scalar or vector). A value of 1.0 leaves the
            saturation unchanged. A value of 0 removes all saturation from the image and makes
            all channels equal in value.

    Returns:
        fp32 tensor (4, 4), color transformation matrix if hue and saturation are scalars. If
        hue and saturation are vectors, (len(hue), 4, 4).

    [1] See https://beesbuzz.biz/code/hsv_color_transforms.php, notice that our matrix convention
        is transposed compared to this reference.
    """
    hue = tf.cast(tf.convert_to_tensor(value=hue), tf.float32)
    saturation = tf.cast(tf.convert_to_tensor(value=saturation), tf.float32)

    const_mat = tf.constant(
        [
            [0.299, 0.299, 0.299, 0.0],
            [0.587, 0.587, 0.587, 0.0],
            [0.114, 0.114, 0.114, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=tf.float32,
    )
    sch_mat = tf.constant(
        [
            [0.701, -0.299, -0.299, 0.0],
            [-0.587, 0.413, -0.587, 0.0],
            [-0.114, -0.114, 0.886, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=tf.float32,
    )
    ssh_mat = tf.constant(
        [
            [0.168, -0.328, 1.25, 0.0],
            [0.330, 0.035, -1.05, 0.0],
            [-0.497, 0.292, -0.203, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=tf.float32,
    )

    angle = hue * (np.pi / 180.0)
    sch = saturation * tf.cos(angle)
    ssh = saturation * tf.sin(angle)

    if hue.shape.ndims == 1:
        # Tile constant matrices to batch size.
        batch_size = tf.shape(input=hue)[0]
        const_mat = tf.tile(tf.expand_dims(const_mat, 0), [batch_size, 1, 1])
        sch_mat = tf.tile(tf.expand_dims(sch_mat, 0), [batch_size, 1, 1])
        ssh_mat = tf.tile(tf.expand_dims(ssh_mat, 0), [batch_size, 1, 1])
        # Reshape to 3D for element-wise multiplication.
        sch = tf.reshape(sch, [batch_size, 1, 1])
        ssh = tf.reshape(ssh, [batch_size, 1, 1])

    return const_mat + sch * sch_mat + ssh * ssh_mat


def random_hue_saturation_matrix(
    hue_rotation_max,
    saturation_shift_max,
    batch_size=None,
    hue_center=0.0,
    saturation_shift_min=None,
):
    """Get random hue-saturation transformation matrix.

    Args:
        hue_rotation_max (float): The maximum rotation angle. This used in a truncated
            normal distribution, with a zero mean. This rotation angle is half of the
            standard deviation, because twice the standard deviation will be truncated.
            A value of 0 will not affect the matrix.
        saturation_shift_max (float): The random uniform shift that changes the
            saturation. This value gives is the positive extent of the
            augmentation, where a value of 0 leaves the matrix unchanged.
            For example, a value of 1 can result in a saturation values bounded
            between of 0 (entirely desaturated) and 2 (twice the saturation).
        batch_size (int): If None, return a single matrix, else return a batch of matrices.
        hue_center (float): The center of the distribution from which to select the hue.
        saturation_shift_min (float): The minimum of the uniform distribution from which to
            select the saturation shift. If unspecified, defaults to -saturation_shift_max.

    Returns:
        (tf.Tensor) If batch_size is None, a color transformation matrix of shape (4,4)
        and type tf.float32. If batch_size is not None, a tensor of shape (batch_size,4,4).
    """
    if saturation_shift_min is None:
        saturation_shift_min = -saturation_shift_max

    batch_shape = [] if batch_size is None else [batch_size]
    hue = tf.random.truncated_normal(
        batch_shape, mean=hue_center, stddev=hue_rotation_max / 2.0
    )
    mean_saturation = 1  # no saturation when saturation_shift_max=0
    saturation = mean_saturation + tf.random.uniform(
        batch_shape, minval=saturation_shift_min, maxval=saturation_shift_max
    )
    return hue_saturation_matrix(hue, saturation)


def random_contrast_matrix(scale_max, center, batch_size=None, scale_center=0.0):
    """Create random contrast transformation matrix.

    Args:
        scale_max (float): The scale (or slope) of the contrast, as rotated
            around the provided center point. This value is half of the standard
            deviation, where values of twice the standard deviation are truncated.
            A value of 0 will not affect the matrix.
        center (float): The center around which the contrast is 'tilted', this
            is generally equal to the middle of the pixel value range. This value is
            typically 0.5 with a maximum pixel value of 1, or 127.5 when the maximum
            value is 255.
        batch_size (int): If None, return a single matrix, else return a batch of matrices.
        scale_center (float): The center of the normal distribution from which to choose the
            contrast scale.

    Returns:
        (tf.Tensor) If batch_size is None, a color transformation matrix of shape (4,4)
        and type tf.float32. If batch_size is not None, a tensor of shape (batch_size,4,4).
    """
    batch_shape = [] if batch_size is None else [batch_size]
    contrast = tf.random.truncated_normal(
        batch_shape, mean=scale_center, stddev=scale_max / 2.0
    )
    return contrast_matrix(contrast, center)


def random_brightness_matrix(
    brightness_scale_max,
    brightness_uniform_across_channels=True,
    batch_size=None,
    brightness_center=0.0,
):
    """Create a random brightness transformation matrix.

    Args:
        brightness_scale_max (float): The range of the brightness offsets. This value
            is half of the standard deviation, where values of twice the standard
            deviation are truncated. A value of 0 will not affect the matrix.
        brightness_uniform_across_channels (bool): If true will apply the same brightness
            shift to all channels. If false, will apply a different brightness shift to each
            channel.
        batch_size (int): If None, return a single matrix, else return a batch of matrices.
        brightness_center (float): The center of the distribution of brightness offsets to
            sample from.

    Returns:
        (tf.Tensor) If batch_size is None, a color transformation matrix of shape (4,4)
        and type tf.float32. If batch_size is not None, a tensor of shape (batch_size,4,4).
    """
    if not brightness_uniform_across_channels:
        batch_shape = [3] if batch_size is None else [batch_size, 3]
        brightness_offset = tf.random.truncated_normal(
            batch_shape, mean=brightness_center, stddev=brightness_scale_max / 2.0
        )
    else:
        batch_shape = [1] if batch_size is None else [batch_size, 1]
        tile_shape = [3] if batch_size is None else [1, 3]
        randoms = tf.random.truncated_normal(
            batch_shape, mean=brightness_center, stddev=brightness_scale_max / 2.0
        )
        brightness_offset = tf.tile(randoms, tile_shape)
    return brightness_offset_matrix(offset=brightness_offset)


def get_color_transformation_matrix(
    ctm=None,
    hue_rotation=0.0,
    saturation_shift=0.0,
    contrast_scale=0.0,
    contrast_center=0.5,
    brightness_scale=0,
    batch_size=None,
):
    """
    The color transformation matrix (ctm) generator used for a specific set of values.

    This function creates a color transformation matrix (ctm) with the exact values given
    to augment 3-channel color images.

    Args:
        ctm ((4,4) fp32 Tensor or None): A color transformation matrix.
            If ``None`` (default), an identity matrix will be used.
        hue_rotation (float): The rotation angle for the hue.
        saturation_shift (float): The amound to shift the saturation of the image.
        contrast_scale (float): The scale (or slope of the contrast), as rotated
            around the provided center point.
        contrast_center (float): The center around which the contrast is 'tilted', this
            is generally equal to the middle of the pixel value range. This value is
            typically 0.5 with a maximum pixel value of 1, or 127.5 when the maximum
            value is 255.
        brightness_scale (float): The brightness offsets. A value of 0 (default)
            will not affect the matrix.
    """
    return get_random_color_transformation_matrix(
        ctm=ctm,
        hue_center=hue_rotation,
        saturation_shift_max=saturation_shift,
        saturation_shift_min=saturation_shift,
        contrast_scale_center=contrast_scale,
        contrast_center=contrast_center,
        brightness_center=brightness_scale,
    )


def get_random_color_transformation_matrix(
    ctm=None,
    hue_rotation_max=0.0,
    hue_center=0.0,
    saturation_shift_max=0.0,
    saturation_shift_min=None,
    contrast_scale_max=0.0,
    contrast_scale_center=0.0,
    contrast_center=0.5,
    brightness_scale_max=0,
    brightness_center=0.0,
    brightness_uniform_across_channels=True,
    batch_size=None,
):
    """
    The color transformation matrix (ctm) generator used for random augmentation.

    This function creates a random color transformation matrix (ctm) to augment 3-channel
    color images.

    Args:
        ctm ((4,4) fp32 Tensor or None): A random color transformation matrix.
            If ``None`` (default), an identity matrix will be used.
        hue_rotation_max (float): The maximum rotation angle. This used in a truncated
            normal distribution, with a zero mean. This rotation angle is half of the
            standard deviation, because twice the standard deviation will be truncated.
            A value of 0 will not affect the matrix.
        saturation_shift_max (float): The random uniform shift that changes the
            saturation. This value gives is the negative and positive extent of the
            augmentation, where a value of 0 leaves the matrix unchanged.
            For example, a value of 1 can result in a saturation values bounded
            between of 0 (entirely desaturated) and 2 (twice the saturation).
        contrast_scale_max (float): The scale (or slope) of the contrast, as rotated
            around the provided center point. This value is half of the standard
            deviation, where values of twice the standard deviation are truncated.
            A value of 0 (default) will not affect the matrix.
        contrast_scale_center (float): The center of the distribution from which to choose
            the contrast scale.
        contrast_center (float): The center around which the contrast is 'tilted', this
            is generally equal to the middle of the pixel value range. This value is
            typically 0.5 with a maximum pixel value of 1, or 127.5 when the maximum
            value is 255.
        brightness_scale_max (float): The range of the brightness offsets. This value
            is half of the standard deviation, where values of twice the standard
            deviation are truncated.
            A value of 0 (default) will not affect the matrix.
        brightness_uniform_across_channels (bool): If true will apply the same brightness
            shift to all channels. If false, will apply a different brightness shift to each
            channel.
        batch_size (int): If None, return a single matrix, else return a batch of matrices.
    Returns:
        (tf.Tensor) If batch_size is None, a color transformation matrix of shape (4,4)
        and type tf.float32. If batch_size is not None, a tensor of shape (batch_size,4,4).
    """
    # Initialize the spatial transform matrix as a 4x4 identity matrix
    if ctm is None:
        batch_shape = [] if batch_size is None else [batch_size]
        ctm = tf.eye(4, batch_shape=batch_shape, dtype=tf.float32)

    # Apply hue-saturation transformations.
    hue_saturation = random_hue_saturation_matrix(
        hue_rotation_max,
        saturation_shift_max,
        batch_size,
        hue_center,
        saturation_shift_min,
    )
    ctm = tf.matmul(ctm, hue_saturation)

    # Apply contrast transformations.
    contrast = random_contrast_matrix(
        contrast_scale_max, contrast_center, batch_size, contrast_scale_center
    )
    ctm = tf.matmul(ctm, contrast)

    # Apply brightness transformations.
    brightness = random_brightness_matrix(
        brightness_scale_max,
        brightness_uniform_across_channels,
        batch_size,
        brightness_center,
    )
    ctm = tf.matmul(ctm, brightness)
    return ctm
