# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Feature Pyramid Network.

Feature Pyramid Networks were proposed in:
[1] Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan,
    , and Serge Belongie
    Feature Pyramid Networks for Object Detection. CVPR 2017.
"""
import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D


class FPNNetwork(object):
    """Feature Pyramid Network."""

    def __init__(self, min_level=3, max_level=7, filters=256, trainable=True,
                 data_format="channels_first"):
        """Generates multiple scale feature pyramid (FPN).

        Args:
        feats_bottom_up: a dictionary of tensor with level as keys and bottom up
          feature tensors as values. They are the features to generate FPN features.
        min_level: the minimum level number to generate FPN features.
        max_level: the maximum level number to generate FPN features.
        filters: the FPN filter size.

        Returns:
        feats: a dictionary of tensor with level as keys and the generated FPN
          features as values.
        """
        self._local_layers = dict()

        self._min_level = min_level
        self._max_level = max_level

        self._filters = filters

        self._backbone_max_level = 5  # max(feats_bottom_up.keys())
        self._upsample_max_level = (
            self._backbone_max_level
            if self._max_level > self._backbone_max_level else self._max_level
            )

        self._local_layers["stage1"] = dict()
        for level in range(self._min_level, self._upsample_max_level + 1):
            self._local_layers["stage1"][level] = tf.keras.layers.Conv2D(
                filters=self._filters,
                kernel_size=(1, 1),
                padding='same',
                name='l%d' % level,
                trainable=trainable,
                data_format=data_format
            )

        self._local_layers["stage2"] = dict()
        # add post-hoc 3x3 convolution kernel
        for level in range(self._min_level, self._upsample_max_level + 1):
            self._local_layers["stage2"][level] = tf.keras.layers.Conv2D(
                filters=self._filters,
                strides=(1, 1),
                kernel_size=(3, 3),
                padding='same',
                name='post_hoc_d%d' % level,
                trainable=trainable,
                data_format=data_format)

        self._local_layers["stage3_1"] = dict()
        self._local_layers["stage3_2"] = dict()

        if self._max_level == self._upsample_max_level + 1:
            self._local_layers["stage3_1"] = tf.keras.layers.MaxPool2D(
                pool_size=1,
                strides=2,
                padding='valid',
                name='p%d' % self._max_level,
                trainable=trainable,
                data_format=data_format)

        else:
            for level in range(self._upsample_max_level + 1, self._max_level + 1):
                self._local_layers["stage3_2"][level] = tf.keras.layers.Conv2D(
                    filters=self._filters,
                    strides=(2, 2),
                    kernel_size=(3, 3),
                    padding='same',
                    name='p%d' % level,
                    trainable=trainable,
                    data_format=data_format)

    def __call__(self, inputs):
        """Call."""
        feats_bottom_up = inputs

        # lateral connections
        feats_lateral = {}

        for level in range(self._min_level, self._upsample_max_level + 1):
            feats_lateral[level] = self._local_layers["stage1"][level](feats_bottom_up[level])

        # add top-down path
        feats = {self._upsample_max_level: feats_lateral[self._upsample_max_level]}

        for level in range(self._upsample_max_level - 1, self._min_level - 1, -1):
            feats[level] = UpSampling2D(
                size=(2, 2),
                data_format='channels_first',
                name='FPN_up_{}'.format(level))(feats[level + 1])
            feats[level] = tf.keras.layers.Add(
                name='FPN_add_{}'.format(level))([feats_lateral[level], feats[level]])

        # add post-hoc 3x3 convolution kernel
        for level in range(self._min_level, self._upsample_max_level + 1):
            feats[level] = self._local_layers["stage2"][level](feats[level])

        if self._max_level == self._upsample_max_level + 1:
            feats[self._max_level] = self._local_layers["stage3_1"](feats[self._max_level - 1])

        else:
            for level in range(self._upsample_max_level + 1, self._max_level + 1):
                feats[level] = self._local_layers["stage3_2"][level](feats[level - 1])

        return feats
