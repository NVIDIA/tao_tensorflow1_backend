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
"""Processor for applying 2D filtering on images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod, abstractproperty
import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.data_format import (
    CHANNELS_FIRST,
    CHANNELS_LAST,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor import (
    Processor,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Canvas2D,
    Example,
    FEATURE_CAMERA,
    FEATURE_SESSION,
    Images2D,
    SequenceExample,
    TransformedExample,
)
from nvidia_tao_tf1.core.coreobject import save_args


class Filter2DProcessor(Processor):
    """Processor that applies 2D filtering operations."""

    @save_args
    def __init__(self):
        """Construct processor that uses a filter operation."""
        super(Filter2DProcessor, self).__init__()

    @property
    def supported_formats(self):
        """Data formats supported by this processor.

        Returns:
            data_formats (list of 'DataFormat'): Input data formats that this processor supports.
        """
        return [CHANNELS_FIRST, CHANNELS_LAST]

    @abstractproperty
    def probability(self):
        """Probability to apply filters."""

    @abstractmethod
    def get_filters(self):
        """Return a list of filters."""
        raise NotImplementedError("Processors.get_fitlers not implemented.")

    def can_compose(self, other):
        """Can't compose in Filter2DProcessor."""
        return False

    def compose(self, other):
        """Does not support composing."""
        raise NotImplementedError("ComposableProcessor.compose not implemented")

    def process(self, example):
        """
        Process examples by applying filters in sequence.

        The filters are applied in a 2D convolution fashion.

        Args:
            example (Example): Examples to process in format specified by data_format.

        Returns:
            example (Example): Example with a 2D filter applied.
        """
        if isinstance(example, TransformedExample):
            example = example.example
        if not isinstance(example, Example) and not isinstance(
            example, SequenceExample
        ):
            raise TypeError("Tried process input of type: {}".format(type(example)))

        if self.probability is None:
            example = self._apply_filters_to_example(self.get_filters(), example)
        else:
            should_filter = np.random.uniform(0, 1, 1)
            if should_filter < self.probability:
                example = self._apply_filters_to_example(self.get_filters(), example)

        return example

    def _apply_filters_to_example(self, processor_filters, example):
        """Return new Example by applying the processor_filter against the input example.

        Args:
            processor_filters (list): A list of filters.
            example (Example): Examples to process in format specified by data_format.
        Return:
            example (Example): Example with a 2D filter applied.
        """
        if isinstance(example, SequenceExample):
            instances = example.instances
            images = instances[FEATURE_CAMERA].images
            images_rank = len(images.get_shape().as_list())
            if images_rank == 5:
                images = tf.squeeze(images, 1)
        else:
            # Legacy LaneNet dataloader expect transformations to be applied
            # here. TODO(mlehr): remove this functionality once we delete
            # the old dataloader.
            instances = example.instances
            images = instances[FEATURE_CAMERA]
            images_rank = len(images.get_shape().as_list())

        # H, W, and C are read as values because we need them to set the shape of the image before
        # leaving the method.
        # We read N as a tensor because its value is not known.
        if self.data_format == CHANNELS_LAST:
            _, H, W, C = images.get_shape().as_list()
            N = tf.shape(input=images)[0]
        else:
            _, C, H, W = images.get_shape().as_list()
            N = tf.shape(input=images)[0]

        if self.data_format == CHANNELS_LAST:
            # images dim = [N, 3, 160, 480] -> images dim = [N, 3, 160, 480]
            images = tf.transpose(a=images, perm=[0, 3, 1, 2])

        # images dim = [N, 3, 160, 480] -> images dim = [N*3, 160, 480, 1]
        images = tf.reshape(images, shape=[N * C, H, W, 1])

        images = self._apply_conv_kernels(processor_filters, images)

        # images dim = [N*3, 160, 480, 1] -> images dim = [N, 3, 160, 480]
        images = tf.reshape(images, shape=[N, C, H, W])

        if self.data_format == CHANNELS_LAST:
            # images dim = [N, 3, 160, 480] -> images dim = [N, 160, 480, 3]
            images = tf.transpose(a=images, perm=[0, 2, 3, 1])

        # TODO(mlehr): Make it more general. The instances dictionary could have other keys too.
        if isinstance(example, SequenceExample):
            canvas_height = example.instances[FEATURE_CAMERA].canvas_shape.height
            canvas_width = example.instances[FEATURE_CAMERA].canvas_shape.width
            canvas_shape = Canvas2D(height=canvas_height, width=canvas_width)
            instances = {
                FEATURE_CAMERA: Images2D(
                    images=tf.expand_dims(images, axis=1)
                    if images_rank == 5
                    else images,
                    canvas_shape=canvas_shape,
                ),
                FEATURE_SESSION: instances[FEATURE_SESSION],
            }
            return SequenceExample(instances=instances, labels=example.labels)

        instances[FEATURE_CAMERA] = images
        return Example(instances=instances, labels=example.labels)

    def _apply_conv_kernels(self, processor_filters, images):
        """Return filtered images.

        Args:
            processor_filters (list): A list of filters.
            images (4DTensor): Image tensor with the shape of NHWC.
        Return:
            images (4DTensor): Image tensor with the shape of NHWC.
        """
        combined_padding_height = sum(
            tf.shape(input=f)[0] - 1 for f in processor_filters
        )
        combined_padding_width = sum(
            tf.shape(input=f)[1] - 1 for f in processor_filters
        )

        # Pad images before filtering.

        pad_top = combined_padding_height // 2
        pad_bottom = combined_padding_height - pad_top
        pad_left = combined_padding_width // 2
        pad_right = combined_padding_width - pad_left

        # images dim = [N*3, 160, 480, 1] -> images dim = [N*3, 164, 484, 1].
        images = tf.pad(
            tensor=images,
            paddings=[[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            mode="SYMMETRIC",
        )

        for one_processor_filter in processor_filters:
            # [kernel_height, kernel_width, 1]
            one_processor_filter = tf.expand_dims(one_processor_filter, axis=-1)
            # [kernel_height, kernel_width, 1, 1]
            one_processor_filter = tf.expand_dims(one_processor_filter, axis=-1)

            # images dim = [N*3, 164, 484, 1] -> images dim = [N*3, 160, 484, 1]
            # images dim = [N*3, 160, 484, 1] -> images dim = [N*3, 160, 480, 1]
            # Always NHWC here b/c conv2d CPU supports this format.
            images = tf.nn.conv2d(
                input=images,
                filters=one_processor_filter,
                strides=[1, 1, 1, 1],
                padding="VALID",
                data_format="NHWC",
            )

        return images
