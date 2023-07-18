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
"""Processor for rasterizing labels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nvidia_tao_tf1.core.coreobject import TAOObject, save_args
from nvidia_tao_tf1.core.processors import SparsePolygonRasterizer as TAOPolygonRasterizer


class PolygonRasterizer(TAOObject):
    """Processor that rasterizes labels."""

    @save_args
    def __init__(
        self,
        height,
        width,
        nclasses,
        one_hot=False,
        binarize=False,
        converters=None,
        include_background=True,
    ):
        """
        Construct a PolygonRasterizer processor.

        Args:
            height (int): Absolute height to rasterize at.
            width (int): Absolute width to rasterize at.
            nclasses (int): Number of distinct classes that labels can have.
            one_hot (Boolean): When True, rasterization produces rasters with
                class_count + 1 output channels. Channel at index N contains rasterized labels for
                class whose class id is N-1. 0th channel is reserved for background (no labels).
                When false, a single channel raster is generated with each pixel having a value one
                greater than the class id of the label it represents. Background where no label is
                present is represented with pixel value 0.
            binarize (Boolean): When one_hot is true, setting binarize=false will allow
                values between 0 and 1 to appear in rasterized labels.
            include_background (bool): If set to true, the rasterized output would also include
                the background channel at channel index=0. This parameter only takes effect when
                `one_hot` parameter is set to `true`.  Default `true`.

        Raises:
            ValueError: When invalid or incompatible arguments are provided.
        """
        super(PolygonRasterizer, self).__init__()
        if height < 1:
            raise ValueError("height: {} is not a positive number.".format(height))
        if width < 1:
            raise ValueError("width: {} is not a positive number.".format(width))

        self.converters = converters or []

        self._rasterize = TAOPolygonRasterizer(
            width=width,
            height=height,
            nclasses=nclasses,
            one_hot=one_hot,
            binarize=binarize,
            data_format="channels_first",
            include_background=include_background,
        )

    @property
    def one_hot(self):
        """Whether or not this rasterizer is using one hot encoding."""
        return self._rasterize.one_hot

    def process(self, labels2d):
        """
        Rasterize sparse tensors.

        Args:
            labels2d (Polygon2DLabel): A label containing 2D polygons/polylines and their
                associated classes and attributes. The first two dimensions of each tensor
                that this structure contains should be batch/example followed by a frame/time
                dimension. The rest of the dimensions encode type specific information. See
                Polygon2DLabel documentation for details

        Returns:
            (tf.Tensor): A dense tensor of shape [B, F, C, H, W] and type tf.float32. Note
                that the number of frames F must currently be the same for each example.
        """
        for converter in self.converters:
            labels2d = converter.process(labels2d)

        dense_shape = labels2d.vertices.coordinates.dense_shape
        if self._rasterize.one_hot:
            channel_count = (
                (self._rasterize.nclasses + 1)
                if self._rasterize.include_background
                else self._rasterize.nclasses
            )
        else:
            channel_count = 1

        def _no_empty_branch():
            compressed_labels = labels2d.compress_frame_dimension()
            compressed_coordinates = compressed_labels.vertices.coordinates

            canvas_shape = labels2d.vertices.canvas_shape
            canvas_height = canvas_shape.height[0].shape.as_list()[-1]
            canvas_width = canvas_shape.width[0].shape.as_list()[-1]
            raster_width_factor = tf.cast(
                self._rasterize.width / canvas_width, tf.float32
            )
            raster_height_factor = tf.cast(
                self._rasterize.height / canvas_height, tf.float32
            )

            scale = tf.linalg.tensor_diag([raster_width_factor, raster_height_factor])
            vertices_2d = tf.reshape(compressed_coordinates.values, [-1, 2])
            scaled_vertices_2d = tf.matmul(vertices_2d, scale)

            compressed_coordinates = tf.SparseTensor(
                indices=compressed_coordinates.indices,
                values=tf.reshape(scaled_vertices_2d, [-1]),
                dense_shape=compressed_coordinates.dense_shape,
            )

            compressed_rasterized = self._rasterize(
                polygons=compressed_coordinates,
                class_ids_per_polygon=compressed_labels.classes,
            )

            uncompressed_rasterized = tf.reshape(
                compressed_rasterized,
                [
                    dense_shape[0],
                    dense_shape[1],
                    channel_count,
                    self._rasterize.height,
                    self._rasterize.width,
                ],
            )
            return uncompressed_rasterized

        def _empty_branch():
            return tf.zeros(
                [
                    dense_shape[0],
                    dense_shape[1],
                    channel_count,
                    self._rasterize.height,
                    self._rasterize.width,
                ],
                dtype=tf.float32,
            )

        # This is to deal with the situation that the tf.sparse.reshape will throw an error when
        # labels2d.vertices.coordinates is an empty tensor.
        # This happens when all images in this batch have no polygons.
        rasterized = tf.cond(
            pred=tf.equal(
                tf.size(input=labels2d.vertices.coordinates.indices), tf.constant(0)
            ),
            true_fn=_empty_branch,
            false_fn=_no_empty_branch,
        )

        batch_size = labels2d.vertices.canvas_shape.height.shape[0]
        max_timesteps_in_example = labels2d.vertices.coordinates.shape[1]

        rasterized.set_shape(
            [
                batch_size,
                max_timesteps_in_example,
                channel_count,
                self._rasterize.height,
                self._rasterize.width,
            ]
        )
        return rasterized
