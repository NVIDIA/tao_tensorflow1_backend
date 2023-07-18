# Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.

"""IVA YOLO anchor box layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
from keras.engine.topology import InputSpec, Layer
import numpy as np
import tensorflow as tf


class YOLOAnchorBox(Layer):
    '''
    YOLOAnchorBox layer.

    This is a keras custom layer for YOLO v3/v4 AnchorBox. Dynamic / static
    input shape anchors are built in different ways so that:
        Dynamic: A lot of TF Ops. Slower
        Static: Only support fixed size. Faster and can be exported to TensorRT.

    Input shape:
        4D tensor of shape `(batch, channels, height, width)`

    Output shape:
        3D tensor of shape `(batch, n_boxes, 6)`. The last axis is
        (cy, cx, ph, pw, step_y, step_x).
    '''

    def __init__(self,
                 anchor_size=None,
                 **kwargs):
        '''
        init function.

        All arguments need to be set to the same values as in the box encoding process,
        otherwise the behavior is undefined. Some of these arguments are explained in
        more detail in the documentation of the `SSDBoxEncoder` class.

        Arguments:
            anchor_size: array of tuple of 2 ints [(width, height), (width, height), ...]
                size must be normalized (i.e. div by image size)
        '''

        self.anchor_size = anchor_size
        super(YOLOAnchorBox, self).__init__(**kwargs)

    def build(self, input_shape):
        """Layer build function."""
        self.input_spec = [InputSpec(shape=input_shape)]
        if (input_shape[2] is not None) and (input_shape[3] is not None):
            anchors = np_get_anchor_hw((input_shape[2], input_shape[3]),
                                       [(i[1], i[0]) for i in self.anchor_size])

            # Build a 4D tensor so that TensorRT UFF parser can work correctly.
            anchors = anchors.reshape(1, 1, -1, 6)
            self.num_anchors = anchors.shape[2]

            self.anchors = K.constant(anchors, dtype='float32')
        else:
            self.num_anchors = None
            anchors = np.array([[i[1], i[0]] for i in self.anchor_size]).reshape(1, 1, -1, 2)
            self.anchors = K.constant(anchors, dtype='float32')

        # (feature_map, n_boxes, 6)
        super(YOLOAnchorBox, self).build(input_shape)

    def call(self, x):
        '''
        Return an anchor box tensor based on the shape of the input tensor.

        Note that this tensor does not participate in any graph computations at runtime.
        It is being created as a constant once during graph creation and is just being
        output along with the rest of the model output during runtime. Because of this,
        all logic is implemented as Numpy array operations and it is sufficient to convert
        the resulting Numpy array into a Keras tensor at the very end before outputting it.

        Arguments:
            x (tensor): 4D tensor of shape `(batch, channels, height, width)`.
                The input for this layer must be the output
                of the localization predictor layer.
        '''

        # Compute box width and height for each aspect ratio
        # The shorter side of the image will be used to compute `w` and `h`.
        if self.num_anchors is not None:
            anchor_dup = tf.identity(self.anchors)
            with tf.name_scope(None, 'FirstDimTile'):
                x_dup = tf.identity(x)
                anchors = K.tile(anchor_dup, (K.shape(x_dup)[0], 1, 1, 1))
            # this step is done for TRT export. The BatchDimTile supports 4-D input
            anchors = K.reshape(anchors, [-1, self.num_anchors, 6])
        else:
            feature_w = tf.shape(x)[3]
            feature_h = tf.shape(x)[2]
            anchors = tf.tile(self.anchors, [feature_h, feature_w, 1, 1])
            xx, yy = tf.meshgrid(tf.range(0.0, 1.0, 1.0 / tf.cast(feature_w, tf.float32)),
                                 tf.range(0.0, 1.0, 1.0 / tf.cast(feature_h, tf.float32)))
            xx = tf.reshape(xx, [feature_h, feature_w, 1, 1])
            yy = tf.reshape(yy, [feature_h, feature_w, 1, 1])
            xx = tf.tile(xx, [1, 1, len(self.anchor_size), 1])
            yy = tf.tile(yy, [1, 1, len(self.anchor_size), 1])
            shape_template = tf.zeros_like(yy)
            y_step = shape_template + 1.0 / tf.cast(feature_h, tf.float32)
            x_step = shape_template + 1.0 / tf.cast(feature_w, tf.float32)
            anchors = tf.concat([yy, xx, anchors, y_step, x_step], -1)
            anchors = tf.reshape(anchors, [1, -1, 6])
            anchors = K.tile(anchors, (K.shape(x)[0], 1, 1))

        return anchors

    def compute_output_shape(self, input_shape):
        '''Layer output shape function.'''
        batch_size = input_shape[0]
        return (batch_size, self.num_anchors, 6)

    def get_config(self):
        '''Layer get_config function.'''
        config = {
            'anchor_size': self.anchor_size,
        }
        base_config = super(YOLOAnchorBox, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def np_get_anchor_hw(feature_map_size, anchor_size_hw):
    '''
    Get YOLO Anchors.

    Args:
        im_size: tuple of 2 ints (height, width)
        feature_map_size: tuple of 2 ints (height, width)
        anchor_size_hw: array of tuple of 2 ints [(height, width), (height, width), ...]
    Returns:
        anchor_results: (cy, cx, ph, pw, step_y, step_x)
    '''

    anchor_results = np.zeros((feature_map_size[0], feature_map_size[1], len(anchor_size_hw), 6))
    x, y = np.meshgrid(np.arange(0, 1.0, 1.0 / feature_map_size[1]),
                       np.arange(0, 1.0, 1.0 / feature_map_size[0]))
    y = np.expand_dims(y, -1)
    x = np.expand_dims(x, -1)
    anchor_results[..., 0] += y
    anchor_results[..., 1] += x
    anchor_results[..., 4] += 1.0 / feature_map_size[0]
    anchor_results[..., 5] += 1.0 / feature_map_size[1]

    for idx, anchor in enumerate(anchor_size_hw):
        anchor_results[:, :, idx, 2] += float(anchor[0])
        anchor_results[:, :, idx, 3] += float(anchor[1])

    return anchor_results
