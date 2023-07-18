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
"""Modulus Spatial Transformation Processors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework.sparse_tensor import is_sparse

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.processors import load_custom_tf_op
from nvidia_tao_tf1.core.processors.processors import Processor
from nvidia_tao_tf1.core.types import data_format as modulus_data_format, DataFormat


class PolygonTransform(Processor):
    """
    Processor that transforms polygons using a given spatial transformation matrix.

    Args:
        invert_stm (bool): Whether or not to invert the spatial transformation matrix. This
            is needed when the stm describes the change of new canvas, instead of the old
            canvas (matter of definition) or visa versa.
    """

    @save_args
    def __init__(self, invert_stm=True, **kwargs):
        """__init__ method."""
        # TODO(xiangbok): Inversion should not be default.
        self._invert_stm = invert_stm
        super(PolygonTransform, self).__init__(**kwargs)

    @staticmethod
    def _dense_transform(vertices, stm):
        """Dense transform.

        Args:
            vertices (Tensor): Vertices can be of any rank as long as they can be reshaped
                to [N, -1, 2] where N is the batch dimension.
            stm (Tensor): stm is must be either 3x3 (non-batched) or Nx3x3 (batched).
                Batched vs non-batched is infered from stm rank, which needs to be known
                statically. Batch dimension needs to be known only at runtime.
        Returns:
            Tensor of transformed vertices.
        """
        # Store the original vertices shape.
        vertices_shape = tf.shape(input=vertices)

        if stm.shape.ndims == 3:
            # Batched case. Take batch_size from stm dim 0. Use tf.shape to allow dynamic batch
            # size.
            batch_size = tf.shape(input=stm)[0]
            processing_shape = [batch_size, -1, 2]
        else:
            # Non-batched case. Convert both vertices and stm to batched so that we can handle both
            # batched and non-batched with the same code below.
            processing_shape = [1, -1, 2]
            stm = tf.expand_dims(stm, 0)

        # Reshape vertices to [N, n, 2] for processing.
        vertices_2D = tf.reshape(vertices, processing_shape)

        # Expand vertices into 3D: [x, y] -> [x, y, 1].
        num_vertices = tf.shape(input=vertices_2D)[1]
        one = tf.ones(
            shape=[processing_shape[0], num_vertices, 1], dtype=vertices.dtype
        )
        vertices_3D = tf.concat([vertices_2D, one], axis=-1)

        # Apply the transformation: (N, n, 3) x (N, 3, 3) = (N, n, 3).
        vertices_transformed = tf.matmul(vertices_3D, stm)

        # Normalize back from 3D homogeneous coordinates to regular 2D ones.
        xy = vertices_transformed[:, :, 0:2]
        z = vertices_transformed[:, :, 2:3]
        vertices_transformed_2D = xy / z

        # Restore the original vertices shape.
        return tf.reshape(vertices_transformed_2D, vertices_shape)

    @staticmethod
    def _sparse_transform(vertices, stm):
        """Sparse transform.

        Args:
            vertices (SparseTensor): Vertices can be of any rank as long as they can be reshaped
                to [N, -1, 2] where N is the batch dimension.
            stm (Tensor): stm is assumed to be either 3x3 (non-batched) or Nx3x3 (batched).
                Batched vs non-batched is infered from stm rank.
        Returns:
            SparseTensor of transformed vertices.
        """
        # Convert to dense for matmul. This is simpler, and likely to be faster than doing
        # a sparse matmul.
        dense_vertices = tf.sparse.to_dense(vertices, validate_indices=False)
        vertices_transformed = PolygonTransform._dense_transform(
            vertices=dense_vertices, stm=stm
        )

        # Sparsify.
        sparse_vertices = tf.gather_nd(
            params=vertices_transformed, indices=vertices.indices
        )

        # Rebuild a sparse tensor.
        return tf.SparseTensor(
            indices=vertices.indices,
            values=sparse_vertices,
            dense_shape=vertices.dense_shape,
        )

    def call(self, polygons, stm):
        """call method.

        Args:
            polygons (sparse or dense tensor (float32) of shape (n, 2))): A tensor with ``n``
                vertices with (x, y) coordinates. This tensor can contain multiple polygons
                concatenated together, as all coordinates will be transformed with the same
                transformation matrix.
            stm (tensor (float32) of shape (3, 3) ): spatial transformation matrix
        """
        if self._invert_stm:
            stm = tf.linalg.inv(stm)

        if is_sparse(polygons):
            return self._sparse_transform(vertices=polygons, stm=stm)

        return self._dense_transform(vertices=polygons, stm=stm)


class SpatialTransform(Processor):
    """
    Processor that transforms images using a given spatial transformation matrix.

    Args:
        method (string): Sampling method used. Can be 'nearest', 'bilinear', or 'bicubic'.
        background_value (float): The value the background canvas should have.
        verbose (bool): Toggle verbose output during processing.
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
    """

    SUPPORTED_METHODS = ["nearest", "bilinear", "bicubic"]

    @save_args
    def __init__(
        self,
        method="bilinear",
        background_value=0.0,
        verbose=False,
        data_format=None,
        input_data_format=None,
        output_data_format=None,
        output_dtype=None,
        **kwargs
    ):
        """__init__ method."""
        self.method = method
        self.background_value = background_value
        self.verbose = verbose

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
            data_format = modulus_data_format()

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

        if method not in self.SUPPORTED_METHODS:
            raise NotImplementedError(
                "Sampling method not supported: '{}'".format(method)
            )

        super(SpatialTransform, self).__init__(**kwargs)

    def call(self, images, stms, shape=None):
        """
        Apply spatial transformation (aka. image warp) to images.

        Args:
            images (4D tensor float32): 4D tensor with shape `(batch_size, channels, height, width)`
                if data_format='channels_first', or 4D tensor with shape
                `(batch_size, height, width, channels)` if data_format='channels_last'.
                Note that batch and channel dimensions must exist, even if their sizes are 1.
            stms (3D tensor float32): Spatial transformation matrices of
                shape (batch_size, 3, 3). Matrices are specified in row-major
                format. A matrix M transforms a destination image pixel
                coordinate vector P=[px, py, 1] into source image pixel Q:
                Q = P M. If Q is outside the source image, the
                sampled value is set to the background value, otherwise
                source image is sampled at location Q with a bilinear or
                bicubic filter kernel.
            shape (tuple): A tuple of size 2 containing height and width of
                the output images. If ``shape`` is ``None``, the canvas size
                is unchanged.
        Returns:
            4D tensor float32: Transformed images of shape `(batch_size, channels, height, width)`
                if data_format='channels_first', or 4D tensor with shape
                `(batch_size, height, width, channels)` if data_format='channels_last'.
        """
        images = tf.convert_to_tensor(value=images)
        # Shape inference needs to know whether shape == None. Unfortunately it is not possible
        # to just pass shape=None to the tensorflow side, so we need to create a boolean
        # tensor.
        use_input_image_shape = False
        if shape is None:
            use_input_image_shape = True
            if self.input_data_format == DataFormat.CHANNELS_FIRST:
                shape = tf.shape(input=images)[2:4]  # (height, width).
            else:
                shape = tf.shape(input=images)[1:3]  # (height, width).

        op = load_custom_tf_op("op_spatialtransform.so")

        output_dtype = self.output_dtype
        if output_dtype is None:
            output_dtype = images.dtype

        data_formats = {
            DataFormat.CHANNELS_FIRST: "NCHW",
            DataFormat.CHANNELS_LAST: "NHWC",
        }
        input_data_format = data_formats[self.input_data_format]
        output_data_format = data_formats[self.output_data_format]

        transformed_images = op.spatial_transform(
            images=images,
            transformation_matrices=stms,
            shape=shape,
            use_input_image_shape=use_input_image_shape,
            filter_mode=self.method,
            background_value=self.background_value,
            input_data_format=input_data_format,
            output_data_format=output_data_format,
            output_dtype=output_dtype,
            verbose=self.verbose,
        )
        return transformed_images


def flip_matrix(horizontal, vertical, width=None, height=None):
    """Construct a spatial transformation matrix that flips.

    Note that if width and height are supplied, it will move the object back into the canvas
    together with the flip.

    Args:
        horizontal (bool): If the flipping should be horizontal. Scalar or vector.
        vertical (bool): If the flipping should be vertical. Scalar or vector.
        width (int): the width of the canvas. Used for translating the coordinates into the canvas.
            Defaults to None (no added translation).
        height (int): the height of the canvas. Used for translating the coordinates back into the
            canvas. Defaults to None (no added translation).
    Returns:
        fp32 tensor (3, 3), spatial transformation matrix if horizontal and vertical are scalars.
        If horizontal and vertical are vectors, (len(horizontal), 3, 3).
    """
    # Casting bool to float converts False to 0.0 and True to 1.0.
    h = tf.cast(tf.convert_to_tensor(value=horizontal), tf.float32)
    v = tf.cast(tf.convert_to_tensor(value=vertical), tf.float32)
    zero = tf.zeros_like(h)
    one = tf.ones_like(h)

    if (width is None) ^ (height is None):
        raise ValueError(
            "Variables `width` and `height` should both be defined, or both `None`."
        )
    elif width is not None and height is not None:
        x_t = h * width
        y_t = v * height
    else:
        x_t = zero
        y_t = zero

    m = tf.stack(
        [one - 2.0 * h, zero, zero, zero, one - 2.0 * v, zero, x_t, y_t, one], axis=-1
    )

    shape = [-1, 3, 3] if h.shape.ndims == 1 else [3, 3]
    return tf.reshape(m, shape)


def rotation_matrix(theta, width=None, height=None):
    """Construct a rotation transformation matrix.

    Note that if width and height are supplied, it will rotate the coordinates around the canvas
    center-point, so there will be a translation added to the rotation matrix.

    Args:
        theta (float): the rotation radian. Scalar or vector.
        width (int): the width of the canvas. Used for center rotation. Defaults to None
            (no center rotation).
        height (int): the height of the canvas. Used for center rotation. Defaults to None
            (no center rotation).
    Returns:
        fp32 tensor (3, 3), spatial transformation matrix if theta is scalar. If theta is
        a vector, (len(theta), 3, 3).
    """
    theta = tf.cast(tf.convert_to_tensor(value=theta), tf.float32)
    cos_t = tf.cos(theta)
    sin_t = tf.sin(theta)
    zero = tf.zeros_like(theta)
    one = tf.ones_like(theta)
    if (width is None) ^ (height is None):
        raise ValueError(
            "Variables `width` and `height` should both be defined, or both `None`."
        )
    elif width is not None and height is not None:
        width = tf.cast(tf.convert_to_tensor(value=width), tf.float32)
        height = tf.cast(tf.convert_to_tensor(value=height), tf.float32)
        x_t = height * sin_t / 2.0 - width * cos_t / 2.0 + width / 2.0
        y_t = -1 * height * cos_t / 2.0 + height / 2.0 - width * sin_t / 2.0
    else:
        x_t = zero
        y_t = zero

    m = tf.stack([cos_t, sin_t, zero, -sin_t, cos_t, zero, x_t, y_t, one], axis=-1)

    shape = [-1, 3, 3] if theta.shape.ndims == 1 else [3, 3]
    return tf.reshape(m, shape)


def shear_matrix(ratio_x, ratio_y, width=None, height=None):
    """Construct a shear transformation matrix.

    Note that if width and height are supplied, it will shear the coordinates around
    the canvas center-point, so there will be a translation added to the shear matrix.
    It follows formula:
    [x_new, y_new, 1] = [x, y, 1] * [[1.,                 ratio_y,           0],
                                     [ratio_x,            1.,                0],
                                     [-height*ratio_x/2., -width*ratio_y/2., 1]]

    Args:
        ratio_x (float): the amount of horizontal shift per y row. Scalar or vector.
        ratio_y (float): the amount of vertical shift per x column. Scalar or vector.
        width (int): the width of the canvas. Used for center shearing. Defaults to None
            (no center shearing).
        height (int): the height of the canvas. Used for center shearing. Defaults to None
            (no center shearing).
    Returns:
        fp32 tensor (3, 3), spatial transformation matrix if ratio_{x,y} are scalars. If
        ratio_{x,y} are vectors, (len(ratio_x), 3, 3).
    """
    ratio_x = tf.cast(tf.convert_to_tensor(value=ratio_x), tf.float32)
    ratio_y = tf.cast(tf.convert_to_tensor(value=ratio_y), tf.float32)
    zero = tf.zeros_like(ratio_x)
    one = tf.ones_like(ratio_x)
    if (width is None) ^ (height is None):
        raise ValueError(
            "Variables `width` and `height` should both be defined, or both `None`."
        )
    elif width is not None and height is not None:
        x_t = -1 * height / 2.0 * ratio_x
        y_t = -1 * width / 2.0 * ratio_y
    else:
        x_t = zero
        y_t = zero
    m = tf.stack([one, ratio_y, zero, ratio_x, one, zero, x_t, y_t, one], axis=-1)

    shape = [-1, 3, 3] if ratio_x.shape.ndims == 1 else [3, 3]
    return tf.reshape(m, shape)


def translation_matrix(x, y):
    """Construct a spatial transformation matrix for translation.

    Args:
        x (float): the horizontal translation. Scalar or vector.
        y (float): the vertical translation. Scalar or vector.
    Returns:
        fp32 tensor (3, 3), spatial transformation matrix if x and y are scalars. If
        x and y are vectors, (len(x), 3, 3).
    """
    x = tf.cast(tf.convert_to_tensor(value=x), tf.float32)
    y = tf.cast(tf.convert_to_tensor(value=y), tf.float32)
    zero = tf.zeros_like(x)
    one = tf.ones_like(x)
    m = tf.stack([one, zero, zero, zero, one, zero, x, y, one], axis=-1)

    shape = [-1, 3, 3] if x.shape.ndims == 1 else [3, 3]
    return tf.reshape(m, shape)


def zoom_matrix(ratio, width=None, height=None):
    """Construct a spatial transformation matrix for zooming.

    Note that if width and height are supplied, it will perform a center-zoom by translation.

    Args:
        ratio (float or tuple(2) of float): the zoom ratio. If a tuple of length 2 is supplied,
            they distinguish between the horizontal and vertical zooming. Scalar or vector, or
            a tuple of scalars or vectors.
        width (int): the width of the canvas. Used for center-zooming. Defaults to None (no added
            translation).
        height (int): the height of the canvas. Used for center-zooming. Defaults to None (no added
            translation).
    Returns:
        fp32 tensor (3, 3), spatial transformation matrix if ratio is scalar. If
        ratio is a vector, (len(ratio), 3, 3).
    """
    if type(ratio) == tuple and len(ratio) == 2:
        r_x, r_y = ratio
    else:
        r_x, r_y = ratio, ratio
    r_x = tf.cast(tf.convert_to_tensor(value=r_x), tf.float32)
    r_y = tf.cast(tf.convert_to_tensor(value=r_y), tf.float32)
    zero = tf.zeros_like(r_x)
    one = tf.ones_like(r_x)

    if (width is None) ^ (height is None):
        raise ValueError(
            "Variables `width` and `height` should both be defined, or both `None`."
        )
    elif width is not None and height is not None:
        x_t = (width - width * r_x) * 0.5
        y_t = (height - height * r_y) * 0.5
    else:
        x_t = zero
        y_t = zero

    m = tf.stack([r_x, zero, zero, zero, r_y, zero, x_t, y_t, one], axis=-1)

    shape = [-1, 3, 3] if r_x.shape.ndims == 1 else [3, 3]
    return tf.reshape(m, shape)


def random_flip_matrix(
    horizontal_probability, vertical_probability, width, height, batch_size=None
):
    """Create random horizontal and vertical flip transformation matrix.

    Args:
        horizontal_probability (float): The probability that a left-right flip will occur.
        vertical_probability (float): The probability that a top-bottom flip will occur.
        width (int): the width of the image canvas.
        height (int): the height of the image canvas.
        batch_size (int): If None, return a single matrix, else return a batch of matrices.

    Returns:
        (tf.Tensor) If batch_size is None, a spatial transformation matrix of shape (3,3)
        and type tf.float32. If batch_size is not None, a tensor of shape (batch_size,3,3).
    """
    batch_shape = [] if batch_size is None else [batch_size]
    flip_lr_flag = tf.less(
        tf.random.uniform(batch_shape, 0.0, 1.0), horizontal_probability
    )
    flip_tb_flag = tf.less(
        tf.random.uniform(batch_shape, 0.0, 1.0), vertical_probability
    )
    return flip_matrix(
        horizontal=flip_lr_flag, vertical=flip_tb_flag, width=width, height=height
    )


def random_shear_matrix(
    max_ratio_x,
    max_ratio_y,
    width,
    height,
    batch_size=None,
    min_ratio_x=None,
    min_ratio_y=None,
):
    """Create random shear transformation matrix.

    Args:
        max_ratio_x (float): The higher bound for the uniform distribution from which a
             float will be picked to shear horizontally.
        max_ratio_y (float): The higher bound for the uniform distribution from which a
             float will be picked to shear vertically.
        width (int): The width of the image canvas.
        height (int): The height of the image canvas.
        batch_size (int): If None, return a single matrix, else return a batch of matrices.
        min_ratio_x (float): The lower bound for the uniform distribution from which a
             float will be picked to shear horizontally. If unspecified, defaults to
             -max_ratio_x.
        min_ratio_y (float): The lower bound for the uniform distribution from which a
            float will be picked to shear vertically. If unspecified, defaults to
            -max_ratio_y.

    Returns:
        (tf.Tensor) If batch_size is None, a spatial transformation matrix of shape (3,3)
        and type tf.float32. If batch_size is not None, a tensor of shape (batch_size,3,3).
    """
    if min_ratio_x is None:
        min_ratio_x = -max_ratio_x
    if min_ratio_y is None:
        min_ratio_y = -max_ratio_y
    batch_shape = [] if batch_size is None else [batch_size]
    s_x = tf.random.uniform(
        batch_shape, minval=min_ratio_x, maxval=max_ratio_x, dtype=tf.float32
    )
    s_y = tf.random.uniform(
        batch_shape, minval=min_ratio_y, maxval=max_ratio_y, dtype=tf.float32
    )
    return shear_matrix(s_x, s_y, width, height)


def random_translation_matrix(max_x, max_y, batch_size=None, min_x=None, min_y=None):
    """Create random translation transformation matrix.

    Args:
        max_x (int): The higher bound for the uniform distribution from which an integer will
            be picked to translate horizontally.
        max_y (int): The higher bound for the uniform distribution from which an integer will
            be picked to translate vertically.
        batch_size (int): If None, return a single matrix, else return a batch of matrices.
        min_x (int): The lower bound for the uniform distribution from which an integer will be
            picked to translate horizontally. If unspecified, defaults to -max_x.
        min_y (int): The lower bound for the uniform distribution from which an integer will be
            picked to translate vertically. If unspecified, defaults to -max_y.

    Returns:
        (tf.Tensor) If batch_size is None, a spatial transformation matrix of shape (3,3)
        and type tf.float32. If batch_size is not None, a tensor of shape (batch_size,3,3).
    """
    if min_x is None:
        min_x = -max_x
    if min_y is None:
        min_y = -max_y
    batch_shape = [] if batch_size is None else [batch_size]
    t_x = tf.random.uniform(batch_shape, minval=min_x, maxval=max_x + 1, dtype=tf.int32)
    t_y = tf.random.uniform(batch_shape, minval=min_y, maxval=max_y + 1, dtype=tf.int32)
    return translation_matrix(x=t_x, y=t_y)


def random_zoom_matrix(ratio_min, ratio_max, width, height, batch_size=None):
    """Create random zoom transformation matrix.

    Args:
        ratio_min (float): The lower bound of the zooming ratio's uniform distribution.
            A zooming ratio of 1.0 will not affect the image, while values higher than 1 will
            result in 'zooming out' (image gets rendered smaller than the canvas), and vice versa
            for values below 1.0.
        ratio_max (float): The upper bound of the zooming ratio's uniform distribution.
            A zooming ratio of 1.0 will not affect the image, while values higher than 1 will
            result in 'zooming out' (image gets rendered smaller than the canvas), and vice versa
            for values below 1.0.
        width (int): The width of the image canvas.
        height (int): The height of the image canvas.
        batch_size (int): If None, return a single matrix, else return a batch of matrices.

    Returns:
        (tf.Tensor) If batch_size is None, a spatial transformation matrix of shape (3,3)
        and type tf.float32. If batch_size is not None, a tensor of shape (batch_size,3,3).
    """
    batch_shape = [] if batch_size is None else [batch_size]
    ratio = tf.random.uniform(
        batch_shape, minval=ratio_min, maxval=ratio_max, dtype=tf.float32
    )
    t_x = tf.random.uniform(
        batch_shape, minval=0, maxval=(width - (width / ratio)), dtype=tf.float32
    )
    t_y = tf.random.uniform(
        batch_shape, minval=0, maxval=(height - (height / ratio)), dtype=tf.float32
    )
    scale_stm = zoom_matrix(ratio=ratio)
    translate_stm = translation_matrix(x=-t_x, y=-t_y)
    return tf.matmul(translate_stm, scale_stm)


def random_rotation_matrix(
    rotate_rad_max, width, height, batch_size=None, rotate_rad_min=None
):
    """Create random rotation transformation matrix.

    Args:
        rotate_rad_max (float): Maximum rotation angle. The final rotation angle will be bounded
        by [-rotate_rad_min, rotate_rad_max], following an uniform distribution.
        width (int): The width of the image canvas.
        height (int): The height of the image canvas.
        batch_size (int): If None, return a single matrix, else return a batch of matrices.
        rotate_rad_min (float): Minimum rotation angle. If unspecified, defaults to -rotate_rad_max.

    Returns:
        (tf.Tensor) If batch_size is None, a spatial transformation matrix of shape (3,3)
        and type tf.float32. If batch_size is not None, a tensor of shape (batch_size,3,3).
    """
    if rotate_rad_min is None:
        rotate_rad_min = -rotate_rad_max
    batch_shape = [] if batch_size is None else [batch_size]
    angle = tf.random.uniform(batch_shape, minval=rotate_rad_min, maxval=rotate_rad_max)
    return rotation_matrix(angle, width, height)


def get_spatial_transformation_matrix(
    width,
    height,
    stm=None,
    flip_lr=False,
    translate_x=0,
    translate_y=0,
    zoom_ratio=1.0,
    rotate_rad=0.0,
    shear_ratio_x=0.0,
    shear_ratio_y=0.0,
    batch_size=None,
):
    """
    The spatial transformation matrix (stm) generator used for augmentation.

    This function creates a spatial transformation matrix (stm) that can be used for
    generic data augmentation, usually images or coordinates.
    The order of spatial transform: flip, rotation, zoom and translation.

    Args:
        width (int): the width of the image canvas.
        height (int): the height of the image canvas.
        stm ((3,3) fp32 Tensor or None): A spatial transformation matrix produced in this
            function and will be used to transform images and coordinates spatiallly.
            If ``None`` (default), an identity matrix will be generated.
        flip_lr (bool): Flag to indicate whether to flip the image or not.
        translate_x (int): The amount by which to translate the image horizontally.
        translate_y (int): The amount by which to translate the image vertically.
        zoom_ratio (float): The ratio by which to zoom into the image. A zooming ratio of 1.0
            will not affect the image, while values higher than 1 will result in 'zooming out'
            (image gets rendered smaller than the canvas), and vice versa for values below 1.0.
        rotate_rad (float): The rotation in radians.
        shear_ratio_x (float): The amount to shear the horizontal direction per y row.
        shear_ratio_y (float): The amount to shear the vertical direction per x column.
        batch_size (int): If None, return a single matrix, else return a batch of matrices.
    Returns:
        (tf.Tensor) If batch_size is None, a spatial transformation matrix of shape (3,3)
        and type tf.float32. If batch_size is not None, a tensor of shape (batch_size,3,3).
    """
    return get_random_spatial_transformation_matrix(
        width=width,
        height=height,
        stm=stm,
        flip_lr_prob=1.0 if flip_lr else 0.0,
        translate_max_x=translate_x,
        translate_min_x=translate_x,
        translate_max_y=translate_y,
        translate_min_y=translate_y,
        zoom_ratio_min=zoom_ratio,
        zoom_ratio_max=zoom_ratio,
        rotate_rad_max=rotate_rad,
        rotate_rad_min=rotate_rad,
        shear_max_ratio_x=shear_ratio_x,
        shear_min_ratio_x=shear_ratio_x,
        shear_max_ratio_y=shear_ratio_y,
        shear_min_ratio_y=shear_ratio_y,
        batch_size=batch_size,
    )


def get_random_spatial_transformation_matrix(
    width,
    height,
    stm=None,
    flip_lr_prob=0.0,
    flip_tb_prob=0.0,
    translate_max_x=0,
    translate_min_x=None,
    translate_max_y=0,
    translate_min_y=None,
    zoom_ratio_min=1.0,
    zoom_ratio_max=1.0,
    rotate_rad_max=0.0,
    rotate_rad_min=None,
    shear_max_ratio_x=0.0,
    shear_min_ratio_x=None,
    shear_max_ratio_y=0.0,
    shear_min_ratio_y=None,
    batch_size=None,
):
    """
    The spatial transformation matrix (stm) generator used for random augmentation.

    This function creates a random spatial transformation matrix (stm) that can be used for
    generic data augmentation, usually images or coordinates. The flipping, rotation, translation
    and zooming all have independent probabilities. The RNG used is always of uniform distribution.
    Translation is lossless, as it picks discrete integers.

    The order of spatial transform: flip, rotation, zoom and translation.

    Args:
        width (int): the width of the image canvas.
        height (int): the height of the image canvas.
        stm ((3,3) fp32 Tensor or None): A random spatial transformation matrix produced in this
            function and will be used to transform images and coordinates spatiallly.
            If ``None`` (default), an identity matrix will be generated.
        flip_lr_prob (float): The probability that a left-right (horizontal) flip will occur.
        flip_tb_prob (float): The probability that a top-bottom (vertical) flip will occur.
        translate_max_x (int): If translation occurs, this is the higher bound the
            uniform distribution from which an integer will be picked to translate horizontally.
        translate_min_x (int): If translation occus, this is the lower bound for the uniform
            distribution from which an integer will be picked to translate horizontally. If
            unspecified, it defaults to -translate_max_x.
        translate_max_y (int): If translation occurs, this is the higher bound the
            uniform distribution from which an integer will be picked to translate vertically.
        translate_min_y (int): If translation occurs, this is the lower bound the
            uniform distribution from which an integer will be picked to translate vertically.
            If unspecified, it defaults to -translate_max_y.
        zoom_ratio_min (float): The lower bound of the zooming ratio's uniform distribution.
            A zooming ratio of 1.0 will not affect the image, while values higher than 1 will
            result in 'zooming out' (image gets rendered smaller than the canvas), and vice versa
            for values below 1.0.
        zoom_ratio_max (float): The upper bound of the zooming ratio's uniform distribution.
            A zooming ratio of 1.0 will not affect the image, while values higher than 1 will
            result in 'zooming out' (image gets rendered smaller than the canvas), and vice versa
            for values below 1.0.
        rotate_rad_max (float): The maximal allowed rotation in radians
        rotate_rad_min (float): The minimum allowed rotation in radians. If unspecified,
            defaults to -rotate_rad_max.
        shear_max_ratio_x (float): The maximal allowed shearing of horizontal directions
            per y row.
        shear_min_ratio_x (float): The minimal allowed shearing of horizontal directions
            per y row. If unspecified, defaults to -shear_max_ratio_x.
        shear_max_ratio_y (float): The maximal allowed shearing of vertical directions
            per x column.
        shear_min_ratio_y (float): The minimal allowed shearing of vertical directions
            per y row. If unspecified, defaults to -shear_max_ratio_y.
        batch_size (int): If None, return a single matrix, else return a batch of matrices.
    Returns:
        (tf.Tensor) If batch_size is None, a spatial transformation matrix of shape (3,3)
        and type tf.float32. If batch_size is not None, a tensor of shape (batch_size,3,3).
    """
    # Initialize the spatial transform matrix as a 3x3 identity matrix
    if stm is None:
        batch_shape = [] if batch_size is None else [batch_size]
        stm = tf.eye(3, batch_shape=batch_shape, dtype=tf.float32)

    # Apply horizontal flipping.
    flip = random_flip_matrix(flip_lr_prob, flip_tb_prob, width, height, batch_size)
    stm = tf.matmul(stm, flip)

    # Apply rotation transform.
    rotate_transformation = random_rotation_matrix(
        rotate_rad_max, width, height, batch_size, rotate_rad_min
    )
    stm = tf.matmul(stm, rotate_transformation)

    # Apply zoom transform.
    zoom_transformation = random_zoom_matrix(
        zoom_ratio_min, zoom_ratio_max, width, height, batch_size
    )
    stm = tf.matmul(stm, zoom_transformation)

    # Apply translation.
    translate_transformation = random_translation_matrix(
        translate_max_x, translate_max_y, batch_size, translate_min_x, translate_min_y
    )
    stm = tf.matmul(stm, translate_transformation)

    # Apply shear transform.
    shear_transformation = random_shear_matrix(
        shear_max_ratio_x,
        shear_max_ratio_y,
        width,
        height,
        batch_size,
        shear_min_ratio_x,
        shear_min_ratio_y,
    )
    stm = tf.matmul(stm, shear_transformation)
    return stm
