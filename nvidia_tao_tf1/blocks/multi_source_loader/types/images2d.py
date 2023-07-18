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
"""Images2D are used to represent images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
from nvidia_tao_tf1.core.processors import ColorTransform, SpatialTransform


class Images2D(collections.namedtuple("Images2D", ["images", "canvas_shape"])):
    """
    Geometric primitive for representing images.

    The way this is used:
    1) Data sources create Images2D tuples with 3D tensors.
    2) Dataloader adds batch dimensions, converts the 3D tensors to 5D.
    3) Dataloader applies transformations by calling apply().

    images (tf.Tensor): A 3D tensor of shape [C, H, W], type tf.float32 and scaled to the
        range [0, 1]. The dimensions of the tensor are:
            C: Channel - color channel within a frame (e.g, 0: red, 1: green, 2: blue)
            H: Height - row index spanning from 0 to height - 1 of a frame.
            W: Width - column index spanning from 0 to width - 1 of a frame.
    canvas_shape (Canvas2D): Shape of the canvas on which images reside.
    """

    def apply(self, transformation, **kwargs):
        """Applies transformation.

        Note that at this point we are expecting batching to be applied, so the dataset should have
        two batching dimensions (batch, temporal batch).

        Args:
            transformation (Transformation): The transformation to apply.
            output_image_dtype (tf.dtypes.DType): Output image dtype. Defaults to tf.float32.
        """
        data_format = "channels_first"
        spatial_transform = SpatialTransform(
            method="bilinear",
            background_value=0.5,
            data_format=data_format,
            verbose=False,
        )
        # Fold cast into the color transform op.
        output_dtype = kwargs.get("output_image_dtype") or tf.float32

        color_transform = ColorTransform(
            min_clip=0.0,
            max_clip=1.0,
            data_format=data_format,
            output_dtype=output_dtype,
        )

        images = self.images

        # Shape inference: combine shape known at graph build time with the shape known
        # only at runtime.
        images_shape = images.shape.as_list()
        runtime_images_shape = tf.shape(input=images)
        for i, dim in enumerate(images_shape):
            if dim is None:
                images_shape[i] = runtime_images_shape[i]
        batch_size = images_shape[0]
        sequence_length = images_shape[1]
        num_channels = images_shape[2]
        height = images_shape[3]
        width = images_shape[4]

        stms = transformation.spatial_transform_matrix
        # Introduce sequence dimension.
        stms = tf.expand_dims(stms, axis=1)
        # Tile along the sequence dimension.
        stms = tf.tile(stms, (1, sequence_length, 1, 1))
        # Flatten batch and sequence dimensions.
        stms = tf.reshape(stms, [batch_size * sequence_length, 3, 3])

        ctms = transformation.color_transform_matrix
        # Introduce sequence dimension.
        ctms = tf.expand_dims(ctms, axis=1)
        # Tile along the sequence dimension.
        ctms = tf.tile(ctms, (1, sequence_length, 1, 1))
        # Flatten batch and sequence dimensions.
        ctms = tf.reshape(ctms, [batch_size * sequence_length, 4, 4])

        canvas_height = transformation.canvas_shape.height[0].shape.as_list()[-1]
        canvas_width = transformation.canvas_shape.width[0].shape.as_list()[-1]

        # Flatten batch and sequence dimensions.
        imgs = tf.reshape(
            images, [batch_size * sequence_length, num_channels, height, width]
        )
        imgs = spatial_transform(imgs, stms=stms, shape=(canvas_height, canvas_width))
        # Enable color augmentations only if the input is 3 channel.
        if num_channels == 3:
            imgs = color_transform(imgs, ctms=ctms)
        # Reshape back to separate batch and sequence dimensions.
        transformed_images = tf.reshape(
            imgs,
            [batch_size, sequence_length, num_channels, canvas_height, canvas_width],
        )

        return Images2D(
            images=transformed_images, canvas_shape=transformation.canvas_shape
        )


class LabelledImages2D(collections.namedtuple("LabelledImages2D", ["images", "labels", "shapes"])):
    """
    Geometric primitive for representing images.

    The way this is used:
    1) Data sources create Images2D tuples with 3D tensors.
    2) Dataloader adds batch dimensions, converts the 3D tensors to 5D.
    3) Dataloader applies transformations by calling apply().

    images (tf.Tensor): A 3D tensor of shape [C, H, W], type tf.float32 and scaled to the
        range [0, 1]. The dimensions of the tensor are:
            C: Channel - color channel within a frame (e.g, 0: red, 1: green, 2: blue)
            H: Height - row index spanning from 0 to height - 1 of a frame.
            W: Width - column index spanning from 0 to width - 1 of a frame.
    canvas_shape (Canvas2D): Shape of the canvas on which images reside.
    """

    pass
