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
"""NmsInputs layer in FasterRCNN for post-processing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Layer
import tensorflow as tf


class NmsInputs(Layer):
    '''Prepare input tensors for NMS plugin for post-processing in FasterRCNN.'''

    def __init__(self, regr_std_scaling, **kwargs):
        """Initialize the NmsInputs layer.

        Args:
            regr_std_scaling(tuple): The variances for the RCNN deltas.
        """
        self.regr_std_scaling = regr_std_scaling
        super(NmsInputs, self).__init__(**kwargs)

    def build(self, input_shape):
        """Setup some internal parameters."""
        self.batch_size = input_shape[0][0]
        self.roi_num = input_shape[0][1]
        self.class_num = input_shape[2][2] // 4

    def compute_output_shape(self, input_shape):
        """compute_output_shape.

        Args:
            input_shape(tuple): the shape of the input tensor.
        Returns:
            The output shapes for loc_data, conf_data and prior_data
        """
        batch_size = input_shape[0][0]
        roi_num = input_shape[0][1]
        class_num = input_shape[2][2] // 4
        return [
            (batch_size, roi_num * (class_num + 1) * 4, 1, 1),
            (batch_size, roi_num * (class_num + 1), 1, 1),
            (batch_size, 2, roi_num * 4, 1),
        ]

    def call(self, x, mask=None):
        """Call this layer with inputs.

        Args:
            x(list): The list of input tensors.
                x[0]: the input ROIs in shape (N, B, 4), absolute coordinates.
                x[1]: RCNN confidence in the shape (N, B, C+1),including the background.
                x[2]: RCNN deltas in the shape (N, B, C*4), for valid classes.
        Returns:
            the output tensor of the layer.
        """
        # ROIs: (N, B, 4) to (N, 1, B*4, 1)
        rois = x[0]
        if self.batch_size is None:
            self.batch_size = tf.shape(rois)[0]
        rois = tf.reshape(rois, (self.batch_size, self.roi_num, 4, 1))
        # ROIs is (y1, x1, y2, x2), reorg to (x1, y1, x2, y2) conforming with NMSPlugin
        rois = tf.concat(
            (rois[:, :, 1:2, :],
             rois[:, :, 0:1, :],
             rois[:, :, 3:4, :],
             rois[:, :, 2:3, :]),
            axis=2
        )
        rois = tf.reshape(rois, (self.batch_size, 1, self.roi_num * 4, 1))
        # variances: (N, 1, B*4, 1), encoded in targets
        # so just concat with a dummy tensor
        # to get a tensor of shape (N, 2, B*4, 1)
        prior_data = tf.concat((rois, rois), axis=1, name="prior_data")

        # conf_data: -> (N, B*(C+1), 1, 1)
        # strip the groundtruth class in confidence
        # (N, B, C+1) to (N, B*(C+1), 1, 1)
        conf_data = tf.reshape(
            x[1],
            (self.batch_size, self.roi_num * (self.class_num + 1), 1, 1),
            name="conf_data"
        )

        # loc_data: -> (N, B*(C+1)*4, 1, 1)
        # (N, B, C*4) to (N, B, C, 4)
        loc_data = tf.reshape(x[2], (self.batch_size, self.roi_num, self.class_num, 4))
        loc_data_0 = loc_data[:, :, :, 0:1] * (1.0 / self.regr_std_scaling[0])
        loc_data_1 = loc_data[:, :, :, 1:2] * (1.0 / self.regr_std_scaling[1])
        loc_data_2 = loc_data[:, :, :, 2:3] * (1.0 / self.regr_std_scaling[2])
        loc_data_3 = loc_data[:, :, :, 3:4] * (1.0 / self.regr_std_scaling[3])
        loc_data = tf.concat((loc_data_0, loc_data_1, loc_data_2, loc_data_3), axis=3)
        # padding dummy deltas for background class to get (N, B, C+1, 4)
        # as required by the NMSPlugin
        loc_data = tf.concat((loc_data, loc_data[:, :, 0:1, :]), axis=2)
        loc_data = tf.reshape(
            loc_data,
            (self.batch_size, self.roi_num * (self.class_num + 1)*4, 1, 1),
            name="loc_data"
        )
        return [loc_data, conf_data, prior_data]

    def get_config(self):
        """Get config for this layer."""
        config = {
            'regr_std_scaling': self.regr_std_scaling,
        }
        base_config = super(NmsInputs, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
