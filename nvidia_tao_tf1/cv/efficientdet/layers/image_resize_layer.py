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

"""EfficientDet custom layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras


class ImageResizeLayer(keras.layers.Layer):
    '''A Keras layer to wrap tf.image.resize_nearst_neighbor function.'''

    def __init__(self,
                 target_height=128,
                 target_width=128,
                 **kwargs):
        '''Init function.'''
        self.height = target_height
        self.width = target_width
        super(ImageResizeLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """Resize."""
        return tf.cast(tf.compat.v1.image.resize_nearest_neighbor(
            tf.cast(inputs, tf.float32), [self.height, self.width]), dtype=inputs.dtype)

    def get_config(self):
        '''Keras layer get config.'''
        config = {
            'target_height': self.height,
            'target_width': self.width,
        }
        base_config = super(ImageResizeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
