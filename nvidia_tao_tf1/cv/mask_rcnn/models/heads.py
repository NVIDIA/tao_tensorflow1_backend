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
"""Functions to build various prediction heads in Mask-RCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.cv.mask_rcnn.layers.mask_postprocess_layer import MaskPostprocess
from nvidia_tao_tf1.cv.mask_rcnn.layers.reshape_layer import ReshapeLayer

__all__ = ["RPN_Head_Model", "Box_Head_Model", "Mask_Head_Model"]


class RPN_Head_Model(object):
    """RPN Head."""

    def __init__(self, name,
                 num_anchors,
                 trainable,
                 data_format):
        """Create components."""
        """Shared RPN heads."""
        self._local_layers = dict()

        # TODO(chiachenc): check the channel depth of the first convolution.
        self._local_layers["conv1"] = tf.keras.layers.Conv2D(
            256,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=tf.nn.relu,
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            padding='same',
            trainable=trainable,
            name='rpn',
            data_format=data_format
        )

        # Proposal classification scores
        # scores = tf.keras.layers.Conv2D(
        self._local_layers["conv2"] = tf.keras.layers.Conv2D(
            num_anchors,
            kernel_size=(1, 1),
            strides=(1, 1),
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            padding='valid',
            trainable=trainable,
            name='rpn-class',
            data_format=data_format
        )

        # Proposal bbox regression deltas
        # bboxes = tf.keras.layers.Conv2D(
        self._local_layers["conv3"] = tf.keras.layers.Conv2D(
            4 * num_anchors,
            kernel_size=(1, 1),
            strides=(1, 1),
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            padding='valid',
            trainable=trainable,
            name='rpn-box',
            data_format=data_format
        )

    def __call__(self, inputs):
        """Build graph."""
        net = self._local_layers["conv1"](inputs)
        scores = self._local_layers["conv2"](net)
        bboxes = self._local_layers["conv3"](net)
        scores = tf.keras.layers.Permute([2, 3, 1])(scores)
        bboxes = tf.keras.layers.Permute([2, 3, 1])(bboxes)
        return scores, bboxes


class Box_Head_Model(object):
    """Box Head."""

    def __init__(self, num_classes, mlp_head_dim,
                 name, trainable, *args, **kwargs):
        """Box and class branches for the Mask-RCNN model.

        Args:
        roi_features: A ROI feature tensor of shape
          [batch_size, num_rois, height_l, width_l, num_filters].
        num_classes: a integer for the number of classes.
        mlp_head_dim: a integer that is the hidden dimension in the fully-connected
          layers.
        """

        self._num_classes = num_classes
        self._mlp_head_dim = mlp_head_dim

        self._dense_fc6 = tf.keras.layers.Dense(
            units=mlp_head_dim,
            activation=tf.nn.relu,
            trainable=trainable,
            name='fc6'
        )

        self._dense_fc7 = tf.keras.layers.Dense(
            units=mlp_head_dim,
            activation=tf.nn.relu,
            trainable=trainable,
            name='fc7'
        )

        self._dense_class = tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            bias_initializer=tf.keras.initializers.Zeros(),
            trainable=trainable,
            name='class-predict'
        )

        self._dense_box = tf.keras.layers.Dense(
            num_classes * 4,
            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
            bias_initializer=tf.keras.initializers.Zeros(),
            trainable=trainable,
            name='box-predict'
        )

    def __call__(self, inputs):
        """Build graph.

        Returns:
        class_outputs: a tensor with a shape of
          [batch_size, num_rois, num_classes], representing the class predictions.
        box_outputs: a tensor with a shape of
          [batch_size, num_rois, num_classes * 4], representing the box predictions.
        box_features: a tensor with a shape of
          [batch_size, num_rois, mlp_head_dim], representing the box features.
        """

        # reshape inputs before FC.
        batch_size, num_rois, filters, height, width = inputs.get_shape().as_list()

        net = ReshapeLayer(shape=[batch_size * num_rois, height * width * filters],
                           name='box_head_reshape1')(inputs)
        net = self._dense_fc6(net)

        box_features = self._dense_fc7(net)

        class_outputs = self._dense_class(box_features)

        box_outputs = self._dense_box(box_features)

        class_outputs = ReshapeLayer(shape=[batch_size, num_rois, self._num_classes],
                                     name='box_head_reshape2')(class_outputs)
        box_outputs = ReshapeLayer(shape=[batch_size, num_rois, self._num_classes * 4],
                                   name='box_head_reshape3')(box_outputs)
        return class_outputs, box_outputs, box_features


class Mask_Head_Model(object):
    """Mask Head."""

    @staticmethod
    def _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out):
        """Returns the stddev of random normal initialization as MSRAFill."""
        # Reference:
        # https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.h#L445-L463
        # For example, kernel size is (3, 3) and fan out is 256, stddev is 0.029.
        # stddev = (2/(3*3*256))^0.5 = 0.029
        return (2 / (kernel_size[0] * kernel_size[1] * fan_out)) ** 0.5

    def __init__(self,
                 num_classes,
                 mrcnn_resolution,
                 is_gpu_inference,
                 name,
                 trainable,
                 data_format,
                 *args, **kwargs):
        """Mask branch for the Mask-RCNN model.

        Args:
        roi_features: A ROI feature tensor of shape
          [batch_size, num_rois, height_l, width_l, num_filters].
        class_indices: a Tensor of shape [batch_size, num_rois], indicating
          which class the ROI is.
        num_classes: an integer for the number of classes.
        mrcnn_resolution: an integer that is the resolution of masks.
        is_gpu_inference: whether to build the model for GPU inference.
        """
        self._num_classes = num_classes
        self._mrcnn_resolution = mrcnn_resolution
        self._is_gpu_inference = is_gpu_inference

        self._conv_stage1 = []
        kernel_size = (3, 3)
        fan_out = 256

        init_stddev = Mask_Head_Model._get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)

        for conv_id in range(4):
            self._conv_stage1.append(tf.keras.layers.Conv2D(
                fan_out,
                kernel_size=kernel_size,
                strides=(1, 1),
                padding='same',
                dilation_rate=(1, 1),
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
                bias_initializer=tf.keras.initializers.Zeros(),
                trainable=trainable,
                name='mask-conv-l%d' % conv_id,
                data_format=data_format
            ))

        kernel_size = (2, 2)
        fan_out = 256

        init_stddev = Mask_Head_Model._get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)

        self._conv_stage2 = tf.keras.layers.Conv2DTranspose(
            fan_out,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding='valid',
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
            bias_initializer=tf.keras.initializers.Zeros(),
            trainable=trainable,
            name='conv5-mask',
            data_format=data_format
        )

        kernel_size = (1, 1)
        fan_out = self._num_classes

        init_stddev = Mask_Head_Model._get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)

        self._conv_stage3 = tf.keras.layers.Conv2D(
            fan_out,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='valid',
            kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
            bias_initializer=tf.keras.initializers.Zeros(),
            trainable=trainable,
            name='mask_fcn_logits',
            data_format=data_format
        )

    def __call__(self, inputs, class_indices):
        """Build graph.

        Returns:
        mask_outputs: a tensor with a shape of
          [batch_size, num_masks, mask_height, mask_width],
          representing the mask predictions.
        fg_gather_indices: a tensor with a shape of [batch_size, num_masks, 2],
          representing the fg mask targets.
        Raises:
        ValueError: If boxes is not a rank-3 tensor or the last dimension of
          boxes is not 4.
        """

        batch_size, num_rois, filters, height, width = inputs.get_shape().as_list()

        net = ReshapeLayer(shape=[batch_size * num_rois, filters, height, width],
                           name='mask_head_reshape_1')(inputs)

        for conv_id in range(4):
            net = self._conv_stage1[conv_id](net)

        net = self._conv_stage2(net)

        mask_outputs = self._conv_stage3(net)
        mpp_layer = MaskPostprocess(
            batch_size,
            num_rois,
            self._mrcnn_resolution,
            self._num_classes,
            self._is_gpu_inference)

        return mpp_layer((mask_outputs, class_indices))
