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

"""YOLOv4 Loss for training."""

import numpy as np
import tensorflow as tf
from nvidia_tao_tf1.cv.common.losses.base_loss import BaseLoss
from nvidia_tao_tf1.cv.ssd.utils.box_utils import iou


class YOLOv4Loss(BaseLoss):
    '''
    YOLOv4 Loss class.

    See details here: https://arxiv.org/pdf/1506.02640.pdf
    '''

    def __init__(self,
                 lambda_coord=5.0,
                 lambda_no_obj=50.0,
                 lambda_cls=1.0,
                 smoothing=0.0,
                 ignore_threshold=0.7,
                 focal_loss_alpha=0.25,
                 focal_loss_gamma=2.0):
        '''
        Loss init function.

        NOTE: obj classification loss weight for positive sample is fixed at 1

        Args:
            lambda_coord: coord (bbox regression) loss weight
            lambda_no_obj: obj classification loss weight for negative sample
            lambda_cls: classification loss weight for positive sample
            smoothing: classification label smoothing
            ignore_threshold: iou threshold to ignore boxes when calculating negative obj loss
        '''

        self.lambda_coord = lambda_coord
        self.lambda_no_obj = lambda_no_obj
        self.lambda_cls = lambda_cls
        self.smoothing = smoothing
        self.ignore_threshold = ignore_threshold
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

    def l_ciou(self, yx_gt, hw_gt, yx_pred, hw_pred):
        """CIoU loss of two boxes.

        See https://arxiv.org/pdf/1911.08287.pdf

        l_ciou = l_diou + alpha * v
        l_diou = 1.0 - iou + p^2 / c^2
        v = 4/pi^2 * (arctan(w_gt / h_gt) - arctan(w / h))^2
        alpha = v / ((1-iou) + v)

        Args:
            yx_gt (tensor): Tensor of shape [B,N,2], holding GT y, x
            hw_gt (tensor): Tensor of shape [B,N,2], holding GT h, w
            yx_pred (tensor): Tensor of shape [B,N,2], holding pred y, x
            hw_pred (tensor): Tensor of shape [B,N,2], holding pred h, w

        Returns:
            a tensor with shape [B,N] representing element-wise l_ciou
        """

        # Basic calculations
        y_gt = yx_gt[..., 0]
        x_gt = yx_gt[..., 1]
        h_gt = hw_gt[..., 0]
        w_gt = hw_gt[..., 1]
        ymin_gt = y_gt - 0.5 * h_gt
        ymax_gt = y_gt + 0.5 * h_gt
        xmin_gt = x_gt - 0.5 * w_gt
        xmax_gt = x_gt + 0.5 * w_gt

        y_pred = yx_pred[..., 0]
        x_pred = yx_pred[..., 1]
        h_pred = hw_pred[..., 0]
        w_pred = hw_pred[..., 1]
        ymin_pred = y_pred - 0.5 * h_pred
        ymax_pred = y_pred + 0.5 * h_pred
        xmin_pred = x_pred - 0.5 * w_pred
        xmax_pred = x_pred + 0.5 * w_pred

        min_ymax = tf.minimum(ymax_gt, ymax_pred)
        max_ymin = tf.maximum(ymin_gt, ymin_pred)
        min_xmax = tf.minimum(xmax_gt, xmax_pred)
        max_xmin = tf.maximum(xmin_gt, xmin_pred)
        intersect_h = tf.maximum(0.0, min_ymax - max_ymin)
        intersect_w = tf.maximum(0.0, min_xmax - max_xmin)

        intersect_area = intersect_h * intersect_w

        union_area = h_gt * w_gt + h_pred * w_pred - intersect_area

        # avoid div by zero, add 1e-18
        iou = tf.truediv(intersect_area, union_area + 1e-18)

        max_ymax = tf.maximum(ymax_gt, ymax_pred)
        min_ymin = tf.minimum(ymin_gt, ymin_pred)
        max_xmax = tf.maximum(xmax_gt, xmax_pred)
        min_xmin = tf.minimum(xmin_gt, xmin_pred)

        enclose_h = max_ymax - min_ymin
        enclose_w = max_xmax - min_xmin

        # c2 is square of diagonal length of enclosing box
        c2 = enclose_h * enclose_h + enclose_w * enclose_w

        diff_cy = y_gt - y_pred
        diff_cx = x_gt - x_pred

        p2 = diff_cy * diff_cy + diff_cx * diff_cx

        l_diou = 1.0 - iou + tf.truediv(p2, c2 + 1e-18)

        v_atan_diff = tf.math.atan2(w_gt, h_gt + 1e-18) - tf.math.atan2(w_pred, h_pred + 1e-18)

        v = (4.0 / (np.pi**2)) * v_atan_diff * v_atan_diff

        alpha = tf.truediv(v, 1.0 - iou + v + 1e-18)

        l_ciou = l_diou + alpha * v

        return l_ciou

    def decode_bbox(self, cy, cx, h, w):
        """Decode bbox from (cy, cx, h, w) format to (xmin, ymin, xmax, ymax) format."""
        x_min = cx - 0.5 * w
        x_max = cx + 0.5 * w
        y_min = cy - 0.5 * h
        y_max = cy + 0.5 * h
        return tf.stack([x_min, y_min, x_max, y_max], -1)

    def compute_loss_no_reduce(self, y_true, y_pred):
        '''
        Compute the loss of the YOLO v3 model prediction against the ground truth.

        Arguments:
            y_true (tensor): array of `(batch_size, #boxes,
                                        [cy, cx, h, w, objectness, objectness_neg, cls])`
            !!!objectness_neg is deprecated. DO NOT USE, it's not reliable!!!

            y_pred (tensor): array of `(batch_size, #boxes,
            [cy, cx, ph, pw, step_y, step_x, pred_y, pred_x, pred_h, pred_w, object, cls...])`

        Returns:
            A scalar, the total multitask loss for classification and localization.
        '''
        gt_yx = y_true[:, :, 0:2]

        pred_yx = y_pred[:, :, 0:2] + y_pred[:, :, 6:8] * y_pred[:, :, 4:6]

        gt_hw = y_true[:, :, 2:4]

        pred_hw = tf.minimum(y_pred[:, :, 2:4] * y_pred[:, :, 8:10], 1e18)

        cls_wts = y_true[:, :, -1]

        # assign more loc loss weights to smaller boxes
        loss_scale = 2.0 - y_true[:, :, 2] * y_true[:, :, 3]

        loc_loss = self.l_ciou(gt_yx, gt_hw, pred_yx, pred_hw)

        loc_loss = loc_loss * y_true[:, :, 4] * loss_scale * cls_wts

        # if self.focal_loss_alpha > 0 and self.focal_loss_gamma > 0:
        #     # use focal loss
        #     obj_loss = self.bce_focal_loss(
        #         y_true[:, :, 4:5],
        #         y_pred[:, :, 10:11],
        #         self.focal_loss_alpha,
        #         self.focal_loss_gamma
        #     )
        # else:
        #     # default case(no focal loss)
        #     obj_loss = self.bce_loss(y_true[:, :, 4:5], y_pred[:, :, 10:11])

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

        max_iou = tf.map_fn(max_iou_fn, (y_true, pred_yx, pred_hw), dtype=tf.float32)
        is_neg = tf.cast(tf.less(max_iou, self.ignore_threshold), tf.float32)

        # Do not count positive box as negative even it's iou is small!
        is_neg = (1. - y_true[:, :, 4]) * is_neg

        # loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis=1), axis=0)
        # obj_pos_loss = tf.reduce_mean(
        #     tf.reduce_sum(obj_loss * y_true[:, :, 4], axis=1),
        #     axis=0
        # )
        # obj_neg_loss = tf.reduce_mean(
        #     tf.reduce_sum(obj_loss * tf.stop_gradient(is_neg), axis=1),
        #     axis=0
        # )
        if self.focal_loss_alpha > 0 and self.focal_loss_gamma > 0:
            cls_loss = tf.reduce_sum(
                self.bce_focal_loss(
                    y_true[:, :, 6:-1],
                    y_pred[:, :, 11:],
                    self.focal_loss_alpha,
                    self.focal_loss_gamma,
                    self.smoothing
                ) * y_true[:, :, 4] * cls_wts,
                axis=1
            )
            cls_loss = tf.reduce_mean(cls_loss, axis=0)
        else:
            # cls_loss = tf.reduce_sum(
            #     self.bce_loss(
            #         y_true[:, :, 6:-1],
            #         y_pred[:, :, 11:],
            #         self.smoothing
            #     ) * y_true[:, :, 4] * cls_wts,
            #     axis=1
            # )
            # cls_loss = tf.reduce_mean(cls_loss, axis=0)
            cls_loss = self.bce_loss(
                    y_true[:, :, 6:-1],
                    y_pred[:, :, 11:],
                    self.smoothing
                ) * y_true[:, :, 4] * cls_wts
        # total_loss = self.lambda_coord * loc_loss + obj_pos_loss + self.lambda_cls * cls_loss + \
        #     self.lambda_no_obj * obj_neg_loss
        # return total_loss
        # loc_loss: [#batch, #anchor]; cls_loss: [#batch, #anchor]
        return loc_loss, cls_loss

    def compute_loss(self, y_true, y_pred):
        '''
        Compute the loss of the YOLO v3 model prediction against the ground truth.

        Arguments:
            y_true (tensor): array of `(batch_size, #boxes,
                                        [cy, cx, h, w, objectness, objectness_neg, cls])`
            !!!objectness_neg is deprecated. DO NOT USE, it's not reliable!!!

            y_pred (tensor): array of `(batch_size, #boxes,
            [cy, cx, ph, pw, step_y, step_x, pred_y, pred_x, pred_h, pred_w, object, cls...])`

        Returns:
            A scalar, the total multitask loss for classification and localization.
        '''

        gt_yx = y_true[:, :, 0:2]

        pred_yx = y_pred[:, :, 0:2] + y_pred[:, :, 6:8] * y_pred[:, :, 4:6]

        gt_hw = y_true[:, :, 2:4]

        pred_hw = tf.minimum(y_pred[:, :, 2:4] * y_pred[:, :, 8:10], 1e18)

        cls_wts = y_true[:, :, -1]

        # assign more loc loss weights to smaller boxes
        loss_scale = 2.0 - y_true[:, :, 2] * y_true[:, :, 3]

        loc_loss = self.l_ciou(gt_yx, gt_hw, pred_yx, pred_hw)

        loc_loss = loc_loss * y_true[:, :, 4] * loss_scale * cls_wts

        if self.focal_loss_alpha > 0 and self.focal_loss_gamma > 0:
            # use focal loss
            obj_loss = self.bce_focal_loss(
                y_true[:, :, 4:5],
                y_pred[:, :, 10:11],
                self.focal_loss_alpha,
                self.focal_loss_gamma,
                smoothing=self.smoothing
            )
        else:
            # default case(no focal loss)
            obj_loss = self.bce_loss(
                y_true[:, :, 4:5],
                y_pred[:, :, 10:11],
                smoothing=self.smoothing
            )

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

        max_iou = tf.map_fn(max_iou_fn, (y_true, pred_yx, pred_hw), dtype=tf.float32)
        is_neg = tf.cast(tf.less(max_iou, self.ignore_threshold), tf.float32)

        # Do not count positive box as negative even it's iou is small!
        is_neg = (1. - y_true[:, :, 4]) * is_neg

        loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis=1), axis=0)
        obj_pos_loss = tf.reduce_mean(
            tf.reduce_sum(obj_loss * y_true[:, :, 4], axis=1),
            axis=0
        )
        obj_neg_loss = tf.reduce_mean(
            tf.reduce_sum(obj_loss * tf.stop_gradient(is_neg), axis=1),
            axis=0
        )
        if self.focal_loss_alpha > 0 and self.focal_loss_gamma > 0:
            cls_loss = tf.reduce_sum(
                self.bce_focal_loss(
                    y_true[:, :, 6:-1],
                    y_pred[:, :, 11:],
                    self.focal_loss_alpha,
                    self.focal_loss_gamma,
                    self.smoothing
                ) * y_true[:, :, 4] * cls_wts,
                axis=1
            )
            cls_loss = tf.reduce_mean(cls_loss, axis=0)
        else:
            cls_loss = tf.reduce_sum(
                self.bce_loss(
                    y_true[:, :, 6:-1],
                    y_pred[:, :, 11:],
                    self.smoothing
                ) * y_true[:, :, 4] * cls_wts,
                axis=1
            )
            cls_loss = tf.reduce_mean(cls_loss, axis=0)
        total_loss = self.lambda_coord * loc_loss + obj_pos_loss + self.lambda_cls * cls_loss + \
            self.lambda_no_obj * obj_neg_loss
        return total_loss
