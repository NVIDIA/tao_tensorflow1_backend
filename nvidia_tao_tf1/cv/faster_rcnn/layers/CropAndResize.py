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
"""CropAndResize layer in FasterRCNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
from keras.layers import Layer
import tensorflow as tf

from nvidia_tao_tf1.cv.faster_rcnn.layers.utils import normalize_bbox_tf


class CropAndResize(Layer):
    '''Tensorflow style of ROI pooling layer for 2D inputs.

    CropAndResize is a Tensorflow implementation of the original RoI Pooling layer in FasterRCNN
    paper. In this implementation, TensorFlow crop the RoIs and do resize by linear interpolation.
    This is different from the original implementation as there is quantization steps in original
    RoI Pooling layer.

    Reference: Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun.
    '''

    def __init__(self, pool_size, **kwargs):
        """Initialize the CropAndResize layer.

        Args:
            pool_size(int): output feature width/height of this layer.
        """
        self.pool_size = pool_size
        super(CropAndResize, self).__init__(**kwargs)

    def build(self, input_shape):
        """Setup some internal parameters.

        Args:
            input_shape(tuple): the shape of the input tensor. The first input is feature map
                from backbone, so the number of channels is input_shape[0][1]. The second input is
                RoIs, so the number of RoIs is input_shape[1][1]. These parameters will be used to
                compute the output shape.
        """
        self.nb_channels = input_shape[0][1]
        self.num_rois = input_shape[1][1]

    def compute_output_shape(self, input_shape):
        """compute_output_shape.

        Args:
            input_shape(tuple): the shape of the input tensor.
        Returns:
            The output 5D tensor shape: (None, num_rois, C, P, P).
        """
        batch_size = input_shape[0][0]
        return (batch_size, self.num_rois, self.nb_channels, self.pool_size, self.pool_size)

    def call(self, x, mask=None):
        """Call this layer with inputs.

        Args:
            x(list): The list of input tensors.
                x[0]: the input image in shape (N, C, H, W)
                x[1]: the input ROIs in shape (N, B, 4)
                x[2]: input image for normalizing the coordinates
        Returns:
            the output tensor of the layer.
        """
        assert(len(x) == 3)
        img = x[0]
        rois = x[1]
        input_image = x[2]
        image_h = tf.cast(tf.shape(input_image)[2], tf.float32)
        image_w = tf.cast(tf.shape(input_image)[3], tf.float32)
        img_channels_last = tf.transpose(img, (0, 2, 3, 1))
        rois_reshaped = K.reshape(rois, (-1, 4))
        rois_reshaped = normalize_bbox_tf(rois_reshaped, image_h, image_w)
        # (NB, 4)
        box_idxs = tf.floor_div(tf.where(tf.ones_like(rois_reshaped[:, 0]))[:, 0],
                                self.num_rois)
        box_idxs = tf.cast(box_idxs, tf.int32)
        # for testing
        self.rois_reshaped = rois_reshaped
        final_output = tf.image.crop_and_resize(img_channels_last,
                                                tf.stop_gradient(rois_reshaped),
                                                tf.stop_gradient(box_idxs),
                                                [self.pool_size, self.pool_size],
                                                method='bilinear')
        final_output = tf.transpose(final_output, (0, 3, 1, 2))
        # back to 5D (N, B, C, H, W)
        final_output = tf.reshape(final_output, [-1, self.num_rois,
                                                 tf.shape(final_output)[1],
                                                 tf.shape(final_output)[2],
                                                 tf.shape(final_output)[3]])
        return final_output

    def get_config(self):
        """Get config for this layer."""
        config = {'pool_size': self.pool_size}
        base_config = super(CropAndResize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
