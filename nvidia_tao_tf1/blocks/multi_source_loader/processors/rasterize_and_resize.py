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

from nvidia_tao_tf1.blocks.multi_source_loader.data_format import CHANNELS_LAST
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor import (
    Processor,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Example,
    FEATURE_CAMERA,
    LABEL_MAP,
)
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors import PolygonRasterizer


class RasterizeAndResize(Processor):
    """Processor that rasterizes labels and optionally resizes frames to match raster size."""

    @save_args
    def __init__(
        self,
        height,
        width,
        class_count,
        one_hot=False,
        binarize=False,
        resize_frames=False,
        resize_method=tf.image.ResizeMethod.BILINEAR,
    ):
        """
        Construct a RasterizeAndResize processor.

        Args:
            height (int): Absolute height to rasterize at and optionally resize frames to.
            width (int): Absolute height to rasterize at and optionally resize frames to.
            class_count (int): Number of distinct classes that labels can have.
            one_hot (Boolean): When True, rasterization produces rasters with
                class_count + 1 output channels. Channel at index N contains rasterized labels for
                class whose class id is N-1. 0th channel is reserved for background (no labels).
                When false, a single channel raster is generated with each pixel having a value one
                greater than the class id of the label it represents. Background where no label is
                present is represented with pixel value 0.
            binarize (Boolean): When one_hot is true, setting binarize=false will allow
                values between 0 and 1 to appear in rasterized labels.
            resize (Boolean): Whether to resize images/frames to match the raster size.
            resize_method (tf.image.ResizeMethod): Method used to resize frames.

        Raises:
            ValueError: When invalid or incompatible arguments are provided.
        """
        super(RasterizeAndResize, self).__init__()
        if height < 1:
            raise ValueError("height: {} is not a positive number.".format(height))
        if width < 1:
            raise ValueError("width: {} is not a positive number.".format(width))
        if resize_method not in [
            tf.image.ResizeMethod.BILINEAR,
            tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            tf.image.ResizeMethod.BICUBIC,
            tf.image.ResizeMethod.AREA,
        ]:
            raise ValueError("Unrecognized resize_method: '{}'.".format(resize_method))
        self._height = height
        self._width = width
        self._resize_frames = resize_frames
        self._resize_method = resize_method

        self._rasterize = PolygonRasterizer(
            width=width,
            height=height,
            nclasses=class_count,
            one_hot=one_hot,
            binarize=binarize,
            data_format="channels_first",
        )

    @property
    def supported_formats(self):
        """Data formats supported by this processor."""
        return [CHANNELS_LAST]

    def can_compose(self, other):
        """
        Determine whether two processors can be composed into a single one.

        Args:
            other (Processor): Other processor instance.

        Returns:
            (Boolean): Always False - composition not supporoted.
        """
        return False

    def compose(self, other):
        """Compose two processors into a single one.

        Args:
            other (Processor): Other processor instance.

        Raises:
            NotImplementedError: Composition not supported.
        """
        raise NotImplementedError("Composition not supported.")

    def process(self, example):
        """
        RasterizeAndResize labels and optionally resize images/frames to match raster size.

        The rasterized labels are stored in in NHWC format.

        Args:
            example (Example): Example to rasterize.

        Returns:
            (Example): Example with resized frames and rasterized labels.
        """
        frames = example.instances[FEATURE_CAMERA]
        labels = example.labels[LABEL_MAP]
        height = frames.get_shape().as_list()[1]
        width = frames.get_shape().as_list()[2]

        if self._resize_frames:
            frames = tf.image.resize(
                frames, size=(self._height, self._width), method=self._resize_method
            )

        raster_width_factor = self._width / width
        raster_height_factor = self._height / height
        polygons = tf.matmul(
            labels.polygons,
            tf.constant([[raster_width_factor, 0.0], [0.0, raster_height_factor]]),
        )
        rasterized = self._rasterize(
            polygon_vertices=polygons,
            vertex_counts_per_polygon=labels.vertices_per_polygon,
            class_ids_per_polygon=labels.class_ids_per_polygon,
            polygons_per_image=None,
        )
        # Rasterizes to CHW, but this Processor is NHWC
        rasterized = tf.expand_dims(tf.transpose(a=rasterized, perm=[1, 2, 0]), axis=0)

        return Example(
            instances={FEATURE_CAMERA: frames}, labels={LABEL_MAP: rasterized}
        )
