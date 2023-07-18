# Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.

"""IVA YOLOv4 BBoxPostProcessingLayer Layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine.topology import Layer
import tensorflow as tf


class BBoxPostProcessingLayer(Layer):
    '''
    BBoxPostProcessing layer to map prediction to GT target format.

    xy = softmax(xy) * grid_scale_xy - (grid_scale_xy - 1.0) / 2.0
    wh = exp(wh)

    Args:
        grid_scale_xy: how many boxes you want for final outputs (padded by zeros)
    '''

    def __init__(self,
                 grid_scale_xy=1.0,
                 **kwargs):
        '''Init function.'''

        self.grid_scale_xy = grid_scale_xy
        super(BBoxPostProcessingLayer, self).__init__(**kwargs)

    def call(self, x):
        """
        Post-process detection bbox prediction.

        Input:
            grid_scale_xy: a float indicating how much the grid scale should be
        Output:
            a function takes in detection prediction and returns processed detection prediction.
        """
        # Workaround for UFF export. Need to change if using ONNX
        x_shape = tf.shape(x)
        x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[2], 1])

        # input last dim: [pred_y, pred_x, pred_h, pred_w, object, cls...]
        yx = x[:, :, :2, :]
        hw = x[:, :, 2:4, :]
        yx = tf.sigmoid(yx) * self.grid_scale_xy - (self.grid_scale_xy - 1.0) / 2.0

        # Limit HW max to np.exp(8) to avoid numerical instability.
        # Do not change following seemingly stupid way to construct tf.constant(8)
        # otherwise TensorRT will complain. EXP(8.0) = 2981, more than enough for hw multiplier
        hw = tf.exp(tf.minimum(hw, hw + 8.0 - hw))

        result = tf.concat([yx, hw, x[:, :, 4:, :]], 2)

        result = tf.reshape(result, [x_shape[0], x_shape[1], x_shape[2]])

        return result

    def compute_output_shape(self, input_shape):
        '''Layer output shape function.'''
        return input_shape

    def get_config(self):
        '''Layer get_config function.'''
        config = {
            'grid_scale_xy': self.grid_scale_xy,
        }
        base_config = super(BBoxPostProcessingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
