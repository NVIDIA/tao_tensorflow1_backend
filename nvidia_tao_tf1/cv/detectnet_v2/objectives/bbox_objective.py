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

"""Bounding box coordinates objective."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf

import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.cv.detectnet_v2.cost_function.cost_functions import \
    weighted_GIOU_cost, weighted_L1_cost
from nvidia_tao_tf1.cv.detectnet_v2.objectives.base_objective import BaseObjective
from nvidia_tao_tf1.cv.detectnet_v2.visualization.visualizer import \
    DetectNetTBVisualizer as Visualizer

BBOX_LOSS_BASE_TYPES = {"L1", "GIOU"}

logger = logging.getLogger(__name__)


class BboxObjective(BaseObjective):
    """Bounding box objective.

    BBoxObjective implements the bounding box objective-specific parts of the
    following functionalities:
    - Rasterization (labels -> tensors)
    - Bounding box objective transforms (label domain <-> DNN output domain)
    - Cost function
    - Bounding box objective-specific visualization
    - Spatial transformation of objectives (applying spatial transformation
      matrices to predicted tensors)
    """

    def __init__(self, input_layer_name, output_height, output_width,
                 input_height, input_width, scale, offset, loss_ratios=None):
        """Constructor for the bounding box objective.

        Args:
            input_layer_name (string): Name of the input layer of the Objective head.
                If None the last layer of the model will be used.
            output_height, output_width: Shape of the DNN output tensor.
            input_height, input_width: Shape of the DNN input tensor.
            scale (float): Bounding box scaling factor
            offset (float): Bounding box offset
            loss_ratios (dict(str, float)): Ratios of loss_function.
                Keys should be "L1" or "GIOU", values are the ratios of the certain loss,
                eg. {"L1": 0.5, "GIOU": 0.5} means 0.5 * L1_loss + 0.5 * GIOU_loss.
        """
        super(BboxObjective, self).__init__(
            input_layer_name, output_height, output_width)
        self.name = 'bbox'
        self.num_channels = 4
        self.gradient_flag = tao_core.processors.BboxRasterizer.GRADIENT_MODE_PASSTHROUGH
        # Bbox objective specific properties
        self.input_height = input_height
        self.input_width = input_width
        self.scale = scale
        self.offset = offset

        self.loss_ratios = {}
        if not loss_ratios:
            logger.info("Default L1 loss function will be used.")
            self.loss_ratios = {"L1": 1.0}
        else:
            for loss_function_name, ratio in loss_ratios.items():
                loss_function_time = loss_function_name.upper()
                if loss_function_time not in BBOX_LOSS_BASE_TYPES:
                    raise ValueError("Bbox loss function '{}' is not supported"
                                     .format(loss_function_name))
                elif ratio <= 0.0:
                    raise ValueError("Ratio of loss {} is {} and should be a positive number."
                                     .format(loss_function_name, ratio))
                else:
                    self.loss_ratios[loss_function_name] = ratio

    def cost(self, y_true, y_pred, target_class, loss_mask=None):
        """Bounding box cost function.

        Args:
            y_true: GT tensor dictionary. Contains keys 'bbox' and 'cov_norm'
            y_pred: Prediction tensor dictionary. Contains key 'bbox'
            target_class: (TargetClass) for which to create the cost
            loss_mask: (tf.Tensor) Loss mask to multiply the cost by.

        Returns:
            cost: TF scalar.
        """
        assert 'cov_norm' in y_true
        assert 'bbox' in y_true
        assert 'bbox' in y_pred
        # Compute 'bbox' cost.
        bbox_target = y_true['bbox']
        bbox_pred = y_pred['bbox']
        bbox_weight = y_true['cov_norm']
        bbox_loss_mask = 1.0 if loss_mask is None else loss_mask
        bbox_cost = weighted_L1_cost(bbox_target, bbox_pred,
                                     bbox_weight, bbox_loss_mask)
        bbox_cost = 0.0
        for loss_function_name, ratio in self.loss_ratios.items():
            if loss_function_name == "L1":
                # Use L1-loss for bbox regression.
                cost_item = weighted_L1_cost(bbox_target, bbox_pred,
                                             bbox_weight, bbox_loss_mask)
            else:
                # Use GIOU-loss for bbox regression.
                abs_bbox_target = self._predictions_to_absolute_per_class(
                    bbox_target)
                abs_bbox_pred = self._predictions_to_absolute_per_class(
                    bbox_pred)
                cost_item = weighted_GIOU_cost(abs_bbox_target, abs_bbox_pred,
                                               bbox_weight, bbox_loss_mask)
            bbox_cost = bbox_cost + ratio * cost_item

        mean_cost = tf.reduce_mean(bbox_cost)

        # Visualize cost, target, and prediction.
        if Visualizer.enabled:
            # Visualize mean losses (scalar) always.
            tf.summary.scalar('mean_cost_%s_bbox' %
                              target_class.name, mean_cost)

            # Visualize tensors, if it is enabled in the spec. Use absolute
            # scale to avoid Tensorflow automatic scaling. This facilitates
            # comparing images as the network trains.
            Visualizer.image('%s_bbox_cost' % target_class.name, bbox_cost,
                             value_range=[-0.125, 0.125],
                             collections=[tao_core.hooks.utils.INFREQUENT_SUMMARY_KEY])
            Visualizer.image('%s_bbox_gt' % target_class.name, bbox_target,
                             value_range=[-4.0, 4.0],
                             collections=[tao_core.hooks.utils.INFREQUENT_SUMMARY_KEY])
            Visualizer.image('%s_bbox_pred' % target_class.name, bbox_pred,
                             value_range=[-4.0, 4.0],
                             collections=[tao_core.hooks.utils.INFREQUENT_SUMMARY_KEY])
            if isinstance(loss_mask, tf.Tensor):
                Visualizer.image('%s_bbox_loss_mask' % target_class.name, bbox_loss_mask,
                                 collections=[tao_core.hooks.utils.INFREQUENT_SUMMARY_KEY])

        return mean_cost

    def target_gradient(self, ground_truth_label):
        """Bounding box gradient.

        The bounding box gradients (4, one for each box edge) tell the rasterizer
        to output the distances to each of the bounding box edges from the output pixel.

        Args:
            ground_truth_label: dictionary of label attributes. Uses the attribute
                on key 'target/output_space_coordinates'

        Returns:
            The gradients' coefficients.
        """
        coordinates = ground_truth_label['target/output_space_coordinates']

        # Input coordinates are already in network output "space", i.e. input image pixel
        #  space divided by stride.
        xmin = coordinates[0]
        ymin = coordinates[1]
        xmax = coordinates[2]
        ymax = coordinates[3]

        # Scaled width and height of the bounding box
        bbox_scale_x = float(self.input_width) / \
            (float(self.output_width) * self.scale)
        bbox_scale_y = float(self.input_height) / \
            (float(self.output_height) * self.scale)
        dx = (xmax - xmin) * bbox_scale_x
        dy = (ymax - ymin) * bbox_scale_y
        # Bounding box gradient offset values for x and y directions
        ox = oy = self.offset / self.scale
        # Values of the gradient for distance to left edge of the bounding box at
        # columns xmin and xmax. The gradient increases linearly from ox at column
        # xmin to dx+ox at column xmax.
        L = [ox, dx + ox]
        # Values of the gradient for distance to top edge of the bounding box at
        # rows ymin and ymax. The gradient increases linearly from oy at row
        # ymin to dy+oy at row ymax.
        T = [oy, dy + oy]
        # Values of the gradient for distance to right edge of the bounding box at
        # columns xmin and xmax. The gradient decreases linearly from dx-ox at column
        # xmin to -ox at column xmax.
        R = [dx - ox, -ox]
        # Values of the gradient for distance to bottom edge of the bounding box at
        # rows ymin and ymax. The gradient decreases linearly from dy-oy at row
        # ymin to -oy at row ymax.
        B = [dy - oy, -oy]

        # Bbox coordinates gradient definitions. Gradient coefficients are of the form
        # [x_slope, y_slope, offset], and are computed from values at two endpoints by
        # the helper function _gradient_from_endpoints.
        #
        # The first element in the bbox_coeffs list is the gradient for distance to
        # the left edge of the bounding box. That gradient has a value of L[0] at
        # the left edge of the bbox (x=xmin) and value of L[1] at the right edge of
        # the bbox (x=xmax), with a linear gradient in between. The gradient is vertical,
        # i.e. constant for every row, because the y-coordinate of the both endpoints
        # is the same (here 0.0, but could be in fact chosen arbitrarily).
        # Similarly, the second element contains the gradient coefficients for the
        # distance to the top edge of the bounding box. Here the value is T[0] at
        # the top edge (y=ymin), increasing linearly to T[1] at the bottom edge (y=ymax).
        # The other two gradients are set up similarly.
        bbox_coeffs = [_gradient_from_endpoints(xmin, 0., L[0], xmax, 0., L[1]),
                       _gradient_from_endpoints(0., ymin, T[0], 0., ymax, T[1]),
                       _gradient_from_endpoints(xmin, 0., R[0], xmax, 0., R[1]),
                       _gradient_from_endpoints(0., ymin, B[0], 0., ymax, B[1])]
        gradient = tf.transpose(bbox_coeffs, (2, 0, 1))

        return gradient

    def predictions_to_absolute(self, prediction):
        """Convert grid cell center-relative coordinates to absolute coordinates.

        Convert the bounding box coordinate prediction to absolute coordinates in
        input image plane. Undo scaling and offset done for training.

        Absolute bbox coordinate is computed as (example for left edge):
        L = x_center + offset - pred[:, :, 0, :, :] * scale

        The output coordinates are further clipped here such that the predictions
        are wifthin the input image boundaries.

        Args:
            prediction (tensor): shape (batch, class, self.num_channels, height, width)

        Returns:
            transformed prediction (tensor)
        """
        in_h = self.input_height
        in_w = self.input_width
        out_h = self.output_height
        out_w = self.output_width
        # Construct a 2D grid of cell x and y coordinates and add offset.
        grid_max_x = tf.cast(out_w - 1, tf.float32) * \
            float(in_w) / float(out_w)
        grid_max_y = tf.cast(out_h - 1, tf.float32) * \
            float(in_h) / float(out_h)
        grid_x, grid_y = tf.meshgrid(tf.linspace(self.offset, grid_max_x + self.offset, out_w),
                                     tf.linspace(self.offset, grid_max_y + self.offset, out_h))

        # Multiply LTRB values by bbox_scale to obtain the same scale as the image.
        # Convert from relative to absolute coordinates by adding the grid cell center
        # coordinates. Clip by image boundary and constrain widht and height
        # to be non-negative.
        coords = tf.unstack(prediction * self.scale, axis=2)
        coordsL = tf.clip_by_value(grid_x - coords[0], 0., in_w)
        coordsT = tf.clip_by_value(grid_y - coords[1], 0., in_h)
        coordsR = tf.clip_by_value(grid_x + coords[2], coordsL, in_w)
        coordsB = tf.clip_by_value(grid_y + coords[3], coordsT, in_h)
        coords = tf.stack([coordsL, coordsT, coordsR, coordsB], axis=2)

        return coords

    def transform_predictions(self, prediction, matrices=None):
        """Transform bounding box predictions by spatial transformation matrices.

        Args:
            prediction (tensor): shape (batch, class, self.num_channels, height, width)
            matrices: A tensor of 3x3 transformation matrices, shape (batch, 3, 3).

        Returns:
            transformed prediction (tensor)
        """
        if matrices is None:
            return prediction

        num_classes = int(prediction.shape[1])
        height = int(prediction.shape[3])
        width = int(prediction.shape[4])
        x1 = prediction[:, :, 0]
        y1 = prediction[:, :, 1]
        x2 = prediction[:, :, 2]
        y2 = prediction[:, :, 3]
        one = tf.ones_like(x1)
        # Construct a batch of top-left and bottom-right bbox coordinate vectors.
        # x1-y2 = [n,c,h,w], matrices = [n,3,3].
        # Stack top-left and bottom right coordinate vectors into a tensor [n,c,h,w,6].
        c = tf.stack([x1, y1, one, x2, y2, one], axis=4)
        # Reshape into a batch of vec3s, shape [n,c*h*w*2,3].
        c = tf.reshape(c, (-1, num_classes*height*width*2, 3))
        # Transform the coordinate vectors by the matrices. This loops over the outmost
        # dimension performing n matmuls, each consisting of c*h*w*2 vec3 by mat3x3 multiplies.
        c = tf.matmul(c, matrices)
        # Reshape back into a tensor of shape [n,c,h,w,6].
        c = tf.reshape(c, (-1, num_classes, height, width, 6))
        # Unstack the last dimension to arrive at a list of 6 coords of shape [n,c,h,w].
        c = tf.unstack(c, axis=4)
        # Compute min-max of bbox corners.
        x1 = tf.minimum(c[0], c[3])
        y1 = tf.minimum(c[1], c[4])
        x2 = tf.maximum(c[0], c[3])
        y2 = tf.maximum(c[1], c[4])
        # Reconstruct bbox coordinate tensor [n,c,4,h,w].
        bbox_tensor = tf.stack([x1, y1, x2, y2], axis=2)

        return bbox_tensor

    def _predictions_to_absolute_per_class(self, relative_coords):
        """Wrap predictions_to_absolute to be adapted to coordinate shape [B, C, H, W].

        Args:
            relative_coords (tf.Tensor): Tensors of relative coordinates in output feature space.

        Returns:
            abs_coords (tf.Tensor): Tensors of absolute coordinates in input image space.
        """
        original_shape = [-1, 4, self.output_height, self.output_width]
        expand_shape = [-1, 1, 4, self.output_height, self.output_width]

        relative_coords_expand = tf.reshape(relative_coords, expand_shape)

        abs_coords_expand = self.predictions_to_absolute(
            relative_coords_expand)

        abs_coords = tf.reshape(abs_coords_expand, original_shape)
        return abs_coords


def _gradient_from_endpoints(sx, sy, svalue, ex, ey, evalue):
    """Compute gradient coefficients based on values at two points.

    Args:
        sx: starting point x coordinate
        sy: starting point y coordinate
        svalue: value at the starting point
        ex: ending point x coordinate
        ey: ending point y coordinate
        evalue: value at the ending point

    Returns:
        Gradient coefficients (slope_x, slope_y, offset).
    """
    # edge = [ex - sx, ey - sy]
    # p = [px - sx, py - sy]
    # ratio = dot(p, edge) / |edge|^2
    # value = (1-ratio) * svalue + ratio * evalue
    # ->
    # l = 1 / |edge|^2
    # ratio = ((ex - sx) * (px - sx) + (ey - sy) * (py - sy)) * l
    # ->
    # dvalue = (evalue - svalue), dx = (ex - sx), dy = (ey - sy)
    # value = dvalue * dx * l * px +
    #         dvalue * dy * l * py +
    #         svalue - dvalue * dx * l * sx - dvalue * dy * l * sy
    # ->
    # A = dvalue * dx * l
    # B = dvalue * dy * l
    # C = svalue - dvalue * dx * l * sx - dvalue * dy * l * sy

    dx = ex - sx
    dy = ey - sy
    l = dx * dx + dy * dy  # noqa: E741
    # Avoid division by zero with degenerate bboxes. This effectively clamps the smallest
    # allowed bbox to a tenth of a pixel (note that l is edge length squared). Note that
    # this is just a safety measure. Dataset converters should have removed degenerate
    # bboxes, but augmentation might scale them.
    l = tf.maximum(l, 0.01)  # noqa: E741
    dvalue = (evalue - svalue) / l
    dvx = dvalue * dx
    dvy = dvalue * dy
    offset = svalue - (dvx * sx + dvy * sy)
    vec = [dvx, dvy, offset]
    return vec
