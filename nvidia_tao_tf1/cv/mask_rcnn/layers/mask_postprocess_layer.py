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

"""MaskRCNN custom layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras


class MaskPostprocess(keras.layers.Layer):
    '''A custom Keras layer to generate processed mask output.'''

    def __init__(self,
                 batch_size,
                 num_rois,
                 mrcnn_resolution,
                 num_classes,
                 is_gpu_inference,
                 **kwargs):
        '''Init function.'''
        self.batch_size = batch_size
        self.num_rois = num_rois
        self.mrcnn_resolution = mrcnn_resolution
        self.num_classes = num_classes
        self.is_gpu_inference = is_gpu_inference
        super(MaskPostprocess, self).__init__(**kwargs)

    def call(self, inputs):
        """Proposes RoIs given a group of candidates from different FPN levels."""
        mask_outputs, class_indices = inputs
        if not self.is_gpu_inference:
            class_indices = tf.cast(class_indices, dtype=tf.int32)
        mask_outputs = tf.reshape(
            mask_outputs,
            [-1, self.num_rois, self.num_classes, self.mrcnn_resolution, self.mrcnn_resolution]
        )
        with tf.name_scope('masks_post_processing'):

            indices_dtype = tf.float32 if self.is_gpu_inference else tf.int32

            if self.batch_size == 1:
                indices = tf.reshape(
                    tf.reshape(
                        tf.range(self.num_rois, dtype=indices_dtype),
                        [self.batch_size, self.num_rois, 1]
                    ) * self.num_classes + tf.expand_dims(class_indices, axis=-1),
                    [self.batch_size, -1]
                )
                indices = tf.cast(indices, dtype=tf.int32)

                mask_outputs = tf.gather(
                    tf.reshape(mask_outputs,
                               [self.batch_size, -1, self.mrcnn_resolution, self.mrcnn_resolution]),
                    indices,
                    axis=1
                )

                mask_outputs = tf.squeeze(mask_outputs, axis=1)
                mask_outputs = tf.reshape(
                    mask_outputs,
                    [self.batch_size, self.num_rois, self.mrcnn_resolution, self.mrcnn_resolution])

            else:
                batch_indices = (
                        tf.expand_dims(tf.range(self.batch_size, dtype=indices_dtype), axis=1) *
                        tf.ones([1, self.num_rois], dtype=indices_dtype)
                )

                mask_indices = (
                        tf.expand_dims(tf.range(self.num_rois, dtype=indices_dtype), axis=0) *
                        tf.ones([self.batch_size, 1], dtype=indices_dtype)
                )

                gather_indices = tf.stack([batch_indices, mask_indices, class_indices],
                                          axis=2)

                if self.is_gpu_inference:
                    gather_indices = tf.cast(gather_indices, dtype=tf.int32)

                mask_outputs = tf.gather_nd(mask_outputs, gather_indices)

        return mask_outputs

    def get_config(self):
        '''Keras layer get config.'''
        config = {
            'batch_size': self.batch_size,
            'num_rois': self.num_rois,
            'mrcnn_resolution': self.mrcnn_resolution,
            'num_classes': self.num_classes,
            'is_gpu_inference': self.is_gpu_inference
        }
        base_config = super(MaskPostprocess, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
