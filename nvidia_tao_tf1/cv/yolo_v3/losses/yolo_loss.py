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

"""YOLOv3 Loss for training."""

import tensorflow as tf
from nvidia_tao_tf1.cv.common.losses.base_loss import BaseLoss
from nvidia_tao_tf1.cv.ssd.utils.box_utils import iou


class YOLOv3Loss(BaseLoss):
    '''
    YOLOv3 Loss class.

    See details here: https://arxiv.org/pdf/1506.02640.pdf
    '''

    def __init__(self,
                 lambda_coord=5.0,
                 lambda_no_obj=50.0,
                 lambda_cls=1.0,
                 ignore_threshold=0.7):
        '''
        Loss init function.

        NOTE: obj classification loss weight for positive sample is fixed at 1

        Args:
            lambda_coord: coord (bbox regression) loss weight
            lambda_no_obj: obj classification loss weight for negative sample
            lambda_cls: classification loss weight for positive sample
            ignore_threshold: iou threshold to ignore boxes when calculating negative obj loss
        '''

        self.lambda_coord = lambda_coord
        self.lambda_no_obj = lambda_no_obj
        self.lambda_cls = lambda_cls
        self.ignore_threshold = ignore_threshold

    def decode_bbox(self, cy, cx, h, w):
        """Decode bbox from (cy, cx, h, w) format to (xmin, ymin, xmax, ymax) format."""

        x_min = cx - 0.5 * w
        x_max = cx + 0.5 * w
        y_min = cy - 0.5 * h
        y_max = cy + 0.5 * h
        return tf.stack([x_min, y_min, x_max, y_max], -1)

    def compute_loss(self, y_true, y_pred):
        '''
        Compute the loss of the YOLO v3 model prediction against the ground truth.

        Arguments:
            y_true (tensor): array of `(batch_size, #boxes,
                                        [cy, cx, h, w, objectness, cls])`
            y_pred (tensor): array of `(batch_size, #boxes,
            [cy, cx, ph, pw, step_y, step_x, pred_y, pred_x, pred_h, pred_w, object, cls...])`

        Returns:
            A scalar, the total multitask loss for classification and localization.
        '''

        gt_t_yx = tf.truediv(y_true[:, :, 0:2] - y_pred[:, :, 0:2], y_pred[:, :, 4:6])

        # limit w and h within 0.01 to 100 times of anchor size to avoid gradient explosion.
        # if boxes' true shift is not within this range, the anchor is very very badly chosen...
        gt_t_hw = tf.log(tf.clip_by_value(tf.truediv(y_true[:, :, 2:4], y_pred[:, :, 2:4]),
                                          1e-2, 1e2))

        # assign more loc loss weights to smaller boxes
        loss_scale = 2.0 - y_true[:, :, 2] * y_true[:, :, 3]

        pred_t_yx = tf.sigmoid(y_pred[:, :, 6:8])

        loc_loss = self.L2_loss(tf.concat([gt_t_yx, gt_t_hw], axis=-1),
                                tf.concat([pred_t_yx, y_pred[:, :, 8:10]], axis=-1))
        loc_loss = loc_loss * y_true[:, :, 4] * loss_scale

        obj_loss = self.bce_loss(y_true[:, :, 4:5], y_pred[:, :, 10:11])

        def max_iou_fn(x):
            # x[0] is y_true (#bbox, ...), x[1] is pred_yx (#bbox, ...), x[2] is pred_hw

            # we need to calculate neutral bboxes.
            valid_bbox = tf.boolean_mask(x[0], x[0][:, 4])
            # shape (batch_size, #boxes, 4)
            valid_bbox = self.decode_bbox(valid_bbox[:, 0], valid_bbox[:, 1],
                                          valid_bbox[:, 2], valid_bbox[:, 3])

            pred_bbox = self.decode_bbox(x[1][:, 0], x[1][:, 1],
                                         x[2][:, 0], x[2][:, 1])

            return tf.reduce_max(iou(pred_bbox, valid_bbox), -1)

        pred_yx = pred_t_yx * y_pred[:, :, 4:6] + y_pred[:, :, 0:2]
        pred_hw = tf.exp(y_pred[:, :, 8:10]) * y_pred[:, :, 2:4]

        max_iou = tf.map_fn(max_iou_fn, (y_true, pred_yx, pred_hw), dtype=tf.float32)
        is_neg = tf.cast(tf.less(max_iou, self.ignore_threshold), tf.float32)

        # Do not count positive box as negative even it's iou is small!
        is_neg = (1. - y_true[:, :, 4]) * is_neg

        # we need to avoid divide by zero errors
        pos_box_count = tf.maximum(tf.reduce_sum(y_true[:, :, 4]), 1e-5)
        neg_box_count = tf.maximum(tf.reduce_sum(is_neg), 1e-5)

        loc_loss = tf.truediv(tf.reduce_sum(loc_loss), pos_box_count)
        obj_pos_loss = tf.truediv(tf.reduce_sum(obj_loss * y_true[:, :, 4]), pos_box_count)
        obj_neg_loss = tf.truediv(tf.reduce_sum(obj_loss * is_neg), neg_box_count)
        cls_loss = tf.truediv(tf.reduce_sum(self.bce_loss(y_true[:, :, 5:],
                                                          y_pred[:, :, 11:]) * y_true[:, :, 4]),
                              pos_box_count)
        total_loss = self.lambda_coord * loc_loss + obj_pos_loss + self.lambda_cls * cls_loss + \
            self.lambda_no_obj * obj_neg_loss
        return total_loss
