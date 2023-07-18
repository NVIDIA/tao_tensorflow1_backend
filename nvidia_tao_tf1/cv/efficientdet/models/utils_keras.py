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
"""Common keras utils."""

import functools
from typing import Text
import numpy as np

import tensorflow as tf
from nvidia_tao_tf1.cv.efficientdet.utils import utils


def build_batch_norm(is_training_bn: bool,
                     init_zero: bool = False,
                     data_format: Text = 'channels_last',
                     momentum: float = 0.99,
                     epsilon: float = 1e-3,
                     name: Text = 'tpu_batch_normalization'):
    """Build a batch normalization layer.

    Args:
        is_training_bn: `bool` for whether the model is training.
        init_zero: `bool` if True, initializes scale parameter of batch
            normalization with 0 instead of 1 (default).
        data_format: `str` either "channels_first" for `[batch, channels, height,
            width]` or "channels_last for `[batch, height, width, channels]`.
        momentum: `float`, momentume of batch norm.
        epsilon: `float`, small value for numerical stability.
        name: the name of the batch normalization layer

    Returns:
        A normalized `Tensor` with the same `data_format`.
    """
    if init_zero:
        gamma_initializer = tf.zeros_initializer()
    else:
        gamma_initializer = tf.ones_initializer()

    axis = 1 if data_format == 'channels_first' else -1
    batch_norm_class = utils.batch_norm_class(is_training_bn)
    bn_layer = batch_norm_class(axis=axis,
                                momentum=momentum,
                                epsilon=epsilon,
                                center=True,
                                scale=True,
                                gamma_initializer=gamma_initializer,
                                name=name)
    return bn_layer


class BoxNet:
    """Box regression network."""

    def __init__(self,
                 num_anchors=9,
                 num_filters=32,
                 min_level=3,
                 max_level=7,
                 is_training=False,
                 act_type='swish',
                 repeats=4,
                 separable_conv=True,
                 survival_prob=None,
                 data_format='channels_last',
                 name='box_net',
                 **kwargs):
        """Initialize BoxNet.

        Args:
            num_anchors: number of  anchors used.
            num_filters: number of filters for "intermediate" layers.
            min_level: minimum level for features.
            max_level: maximum level for features.
            is_training: True if we train the BatchNorm.
            act_type: String of the activation used.
            repeats: number of "intermediate" layers.
            separable_conv: True to use separable_conv instead of conv2D.
            survival_prob: if a value is set then drop connect will be used.
            data_format: string of 'channel_first' or 'channels_last'.
            name: Name of the layer.
            **kwargs: other parameters.
        """
        self.num_anchors = num_anchors
        self.num_filters = num_filters
        self.min_level = min_level
        self.max_level = max_level
        self.repeats = repeats
        self.separable_conv = separable_conv
        self.is_training = is_training
        self.survival_prob = survival_prob
        self.act_type = act_type
        self.data_format = data_format

        self.conv_ops = []
        self.bns = []

        for i in range(self.repeats):
            # If using SeparableConv2D
            if self.separable_conv:
                self.conv_ops.append(
                    tf.keras.layers.SeparableConv2D(
                        filters=self.num_filters,
                        depth_multiplier=1,
                        pointwise_initializer=tf.keras.initializers.VarianceScaling(),
                        depthwise_initializer=tf.keras.initializers.VarianceScaling(),
                        data_format=self.data_format,
                        kernel_size=3,
                        activation=None,
                        bias_initializer=tf.zeros_initializer(),
                        padding='same',
                        name='box-%d' % i))
            # If using Conv2d
            else:
                self.conv_ops.append(
                    tf.keras.layers.Conv2D(
                        filters=self.num_filters,
                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                        data_format=self.data_format,
                        kernel_size=3,
                        activation=None,
                        bias_initializer=tf.zeros_initializer(),
                        padding='same',
                        name='box-%d' % i))

            bn_per_level = {}
            for level in range(self.min_level, self.max_level + 1):
                bn_per_level[level] = build_batch_norm(
                    is_training_bn=self.is_training,
                    init_zero=False,
                    data_format=self.data_format,
                    name='box-%d-bn-%d' % (i, level))
            self.bns.append(bn_per_level)

        if self.separable_conv:
            self.boxes = tf.keras.layers.SeparableConv2D(
                filters=4 * self.num_anchors,
                depth_multiplier=1,
                pointwise_initializer=tf.keras.initializers.VarianceScaling(),
                depthwise_initializer=tf.keras.initializers.VarianceScaling(),
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                bias_initializer=tf.zeros_initializer(),
                padding='same',
                name='box-predict')

        else:
            self.boxes = tf.keras.layers.Conv2D(
                filters=4 * self.num_anchors,
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                bias_initializer=tf.zeros_initializer(),
                padding='same',
                name='box-predict')

    def __call__(self, inputs):
        """Call boxnet."""
        box_outputs = {}
        for level in range(self.min_level, self.max_level + 1):
            image = inputs[level]
            for i in range(self.repeats):
                original_image = image
                image = self.conv_ops[i](image)
                image = self.bns[i][level](image, training=self.is_training)
                if self.act_type:
                    image = utils.activation_fn(image, self.act_type)
                if i > 0 and self.survival_prob:
                    image = utils.drop_connect(image, self.is_training,
                                               self.survival_prob)
                    image = image + original_image

            box_outputs[level] = self.boxes(image)

        return box_outputs


class ClassNet:
    """Object class prediction network."""

    def __init__(self,
                 num_classes=91,
                 num_anchors=9,
                 num_filters=32,
                 min_level=3,
                 max_level=7,
                 is_training=False,
                 act_type='swish',
                 repeats=4,
                 separable_conv=True,
                 survival_prob=None,
                 data_format='channels_last',
                 name='class_net',
                 **kwargs):
        """Initialize the ClassNet.

        Args:
            num_classes: number of classes.
            num_anchors: number of anchors.
            num_filters: number of filters for "intermediate" layers.
            min_level: minimum level for features.
            max_level: maximum level for features.
            is_training: True if we train the BatchNorm.
            act_type: String of the activation used.
            repeats: number of intermediate layers.
            separable_conv: True to use separable_conv instead of conv2D.
            survival_prob: if a value is set then drop connect will be used.
            data_format: string of 'channel_first' or 'channels_last'.
            name: the name of this layerl.
            **kwargs: other parameters.
        """
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_filters = num_filters
        self.min_level = min_level
        self.max_level = max_level
        self.repeats = repeats
        self.separable_conv = separable_conv
        self.is_training = is_training
        self.survival_prob = survival_prob
        self.act_type = act_type
        self.data_format = data_format
        self.conv_ops = []
        self.bns = []
        if separable_conv:
            conv2d_layer = functools.partial(
                tf.keras.layers.SeparableConv2D,
                depth_multiplier=1,
                data_format=data_format,
                pointwise_initializer=tf.keras.initializers.VarianceScaling(),
                depthwise_initializer=tf.keras.initializers.VarianceScaling())
        else:
            conv2d_layer = functools.partial(
                tf.keras.layers.Conv2D,
                data_format=data_format,
                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        for i in range(self.repeats):
            # If using SeparableConv2D
            self.conv_ops.append(
                conv2d_layer(self.num_filters,
                             kernel_size=3,
                             bias_initializer=tf.zeros_initializer(),
                             activation=None,
                             padding='same',
                             name='class-%d' % i))

            bn_per_level = {}
            for level in range(self.min_level, self.max_level + 1):
                bn_per_level[level] = build_batch_norm(
                    is_training_bn=self.is_training,
                    init_zero=False,
                    data_format=self.data_format,
                    name='class-%d-bn-%d' % (i, level),
                )
            self.bns.append(bn_per_level)

        self.classes = conv2d_layer(
            num_classes * num_anchors,
            kernel_size=3,
            bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
            padding='same',
            name='class-predict')

    def __call__(self, inputs):
        """Call ClassNet."""

        class_outputs = {}
        for level in range(self.min_level, self.max_level + 1):
            image = inputs[level]
            for i in range(self.repeats):
                original_image = image
                image = self.conv_ops[i](image)
                image = self.bns[i][level](image, training=self.is_training)
                if self.act_type:
                    image = utils.activation_fn(image, self.act_type)
                if i > 0 and self.survival_prob:
                    image = utils.drop_connect(image, self.is_training,
                                               self.survival_prob)
                    image = image + original_image

            class_outputs[level] = self.classes(image)

        return class_outputs
