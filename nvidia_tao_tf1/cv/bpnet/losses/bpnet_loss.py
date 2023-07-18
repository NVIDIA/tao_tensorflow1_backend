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
"""Loss Functions used by BpNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.blocks.losses.loss import Loss
import nvidia_tao_tf1.core as tao_core
import tensorflow as tf


class BpNetLoss(Loss):
    """Loss Class for BpNet."""

    @tao_core.coreobject.save_args
    def __init__(self,
                 focal_loss_params=None,
                 **kwargs):
        """__init__ method.

        Args:
            use_focal_loss (bool): Enable focal L2 loss
        """
        super(BpNetLoss, self).__init__(**kwargs)

        if not focal_loss_params:
            self._use_focal_loss = False
        else:
            self._use_focal_loss = focal_loss_params["use_focal_loss"]
        self._focal_loss_params = focal_loss_params

    def __call__(self,
                 labels,
                 predictions,
                 masks,
                 output_sequence,
                 gt_slice_index):
        """__call__ method.

        Calculate the loss.

        Args:
            labels (tf.Tensor): labels.
            predictions (tf.Tensor): predictions.
            masks (tf.Tensor): regions over which loss is ignored.
            output_sequence (list): specifies the feature map type for each prediction.
                ex. [('cmap', 1), ('cmap', 2), ('paf', 1), ('paf', 2)] for 2 stage model
                where 'cmap' is confidence map (or) heatmap and,
                'paf' is part affinity fields.
            config (BpNetConfig): contains meta data

        Returns:
            losses (list): list of tensors containing loss for each stage/prediction
        """

        paf_end_idx = gt_slice_index['paf'][1]
        paf_start_idx = gt_slice_index['paf'][0]
        cmap_end_idx = gt_slice_index['heatmap_with_background'][1]
        cmap_start_idx = gt_slice_index['heatmap_with_background'][0]

        # split the mask into paf_mask and heat_mask
        paf_mask = masks[:, :, :, paf_start_idx:paf_end_idx]
        cmap_mask = masks[:, :, :, cmap_start_idx:cmap_end_idx]

        # split the label into paf_label and heat_label
        paf_labels = labels[:, :, :, paf_start_idx:paf_end_idx]
        cmap_labels = labels[:, :, :, cmap_start_idx:cmap_end_idx]
        assert len(output_sequence) == len(predictions)

        if self._use_focal_loss:
            losses = self.focal_loss(
                predictions,
                cmap_labels,
                paf_labels,
                cmap_mask,
                paf_mask,
                output_sequence
            )
        else:
            losses = self.l2_loss(
                predictions,
                cmap_labels,
                paf_labels,
                cmap_mask,
                paf_mask,
                output_sequence
            )

        return losses

    def l2_loss(self,
                predictions,
                cmap_labels,
                paf_labels,
                cmap_mask,
                paf_mask,
                output_sequence):
        """Function to compute l2 loss.

        Args:
            predictions (tf.Tensor): model predictions.
            cmap_labels (tf.Tensor): heatmap ground truth labels.
            paf_labels (tf.Tensor): part affinity field ground truth labels.
            cmap_mask (tf.Tensor): heatmap regions over which loss is ignored.
            paf_mask (tf.Tensor): paf regions over which loss is ignored.
            output_sequence (list): specifies the feature map type for each prediction.
                ex. [('cmap', 1), ('cmap', 2), ('paf', 1), ('paf', 2)] for 2 stage model
                where 'cmap' is confidence map (or) heatmap and,
                'paf' is part affinity fields.

        Returns:
            losses (list): list of tensors containing loss for each stage/prediction
        """

        losses = []
        for idx, pred in enumerate(predictions):
            if output_sequence[idx][0] == 'cmap':
                loss = tf.nn.l2_loss((pred - cmap_labels) * cmap_mask)
                tf.summary.scalar(name='cmap_loss_stage{}'.format(
                    output_sequence[idx][1]),
                                  tensor=loss)
            elif output_sequence[idx][0] == 'paf':
                loss = tf.nn.l2_loss((pred - paf_labels) * paf_mask)
                tf.summary.scalar(name='paf_loss_stage{}'.format(
                    output_sequence[idx][1]),
                                  tensor=loss)
            else:
                raise ValueError("output_sequence must be in ['cmap', 'paf']")

            losses.append(loss)

        return losses

    def focal_loss(self,
                   predictions,
                   cmap_labels,
                   paf_labels,
                   cmap_mask,
                   paf_mask,
                   output_sequence):
        """Function to focal l2 loss.

        Args:
            predictions (tf.Tensor): model predictions.
            cmap_labels (tf.Tensor): heatmap ground truth labels.
            paf_labels (tf.Tensor): part affinity field ground truth labels.
            cmap_mask (tf.Tensor): heatmap regions over which loss is ignored.
            paf_mask (tf.Tensor): paf regions over which loss is ignored.
            output_sequence (list): specifies the feature map type for each prediction.
                ex. [('cmap', 1), ('cmap', 2), ('paf', 1), ('paf', 2)] for 2 stage model
                where 'cmap' is confidence map (or) heatmap and,
                'paf' is part affinity fields.

        Returns:
            losses (list): list of tensors containing loss for each stage/prediction
        """

        cmap_bkg_mask = tf.greater(cmap_labels, self._focal_loss_params['bkg_thresh'])
        paf_bkg_mask = tf.greater(paf_labels, self._focal_loss_params['bkg_thresh'])

        losses = []
        for idx, pred in enumerate(predictions):
            if output_sequence[idx][0] == 'cmap':
                focal_scaling = self.compute_focal_loss_factor(
                    cmap_bkg_mask, cmap_labels, pred)
                loss = tf.nn.l2_loss((pred - cmap_labels) * cmap_mask * focal_scaling)
                tf.summary.scalar(name='cmap_loss_stage{}'.format(
                    output_sequence[idx][1]),
                                  tensor=loss)
            elif output_sequence[idx][0] == 'paf':
                focal_scaling = self.compute_focal_loss_factor(
                    paf_bkg_mask, paf_labels, pred)
                loss = tf.nn.l2_loss((pred - paf_labels) * paf_mask * focal_scaling)
                tf.summary.scalar(name='paf_loss_stage{}'.format(
                    output_sequence[idx][1]),
                                  tensor=loss)
            else:
                raise ValueError("output_sequence must be in ['cmap', 'paf']")

            losses.append(loss)

        return losses

    def compute_focal_loss_factor(self, bkg_mask, label, prediction):
        """Function to compute focal loss factor.

        Args:
            bkg_mask (tf.Tensor): binary mask with background pixels as `False`.
            prediction (tf.Tensor): model prediction.
            label (tf.Tensor): ground truth label.

        Returns:
            scale (tf.Tensor): scale factor to use for loss
        """

        adjusted_scores = tf.where(
            bkg_mask,
            prediction - self._focal_loss_params['alpha'],
            1 - prediction - self._focal_loss_params['beta']
        )
        scale = tf.pow(tf.abs(1 - adjusted_scores), self._focal_loss_params['gamma'])

        return scale
