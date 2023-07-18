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
"""Tests for BpNet loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.cv.bpnet.losses.bpnet_loss import BpNetLoss


class Config():
    """Mock config class."""

    def __init__(self, num_paf_layers, num_cmap_layers):
        """__init__ method.

        Args:
            num_paf_layers (int): number of channels in part-affinity fields head.
            num_cmap_layers (int): number of channels in heatmap head.
        """
        self.gt_slice_index = {
            'paf': [0, num_paf_layers],
            'heatmap_with_background': [num_paf_layers, num_cmap_layers + num_paf_layers]
        }


# Number of output channels for the network heads
NUM_PAF_CHANNELS = 38
NUM_CMAP_CHANNELS = 19
# Ground truth feature map shape [batchsize, height, width, channels]
GT_SHAPE = [2, 64, 64, NUM_PAF_CHANNELS + NUM_CMAP_CHANNELS]
# Part affinity field head output shape [batchsize, height, width, channels]
# where num_connections
PAF_SHAPE = [2, 64, 64, NUM_PAF_CHANNELS]
# Confidence map head output shape [batchsize, height, width, channels]
CMAP_SHAPE = [2, 64, 64, NUM_CMAP_CHANNELS]

test_inputs = [
    # Different sequences
    ([('cmap', 1), ('paf', 1)], [1, 1], [38, 19]),
    ([('cmap', 1), ('cmap', 2), ('paf', 1)], [1, 1], [38, 19]),
    ([('cmap', 1), ('cmap', 2), ('cmap', 3)], [1, 1], [38, 19]),
    # Different masks [mask_paf, mask_cmap]
    ([('cmap', 1), ('cmap', 2), ('paf', 1), ('paf', 2)], [0, 0], [38, 19]),
    ([('cmap', 1), ('cmap', 2), ('paf', 1), ('paf', 2)], [0, 1], [38, 19]),
    ([('cmap', 1), ('cmap', 2), ('paf', 1), ('paf', 2)], [1, 0], [38, 19]),
    ([('cmap', 1), ('cmap', 2), ('paf', 1), ('paf', 2)], [1, 1], [38, 19]),
    # Different feature map sizes [num_paf_layers, num_cmap_layers]
    ([('cmap', 1), ('cmap', 2), ('cmap', 3)], [1, 1], [20, 20]),
    ([('cmap', 1), ('cmap', 2), ('cmap', 3)], [1, 1], [0, 20]),
]


def make_np_predictions(output_sequence):
    """Calculate the predictions in numpy.

    Args:
        output_sequence (list): specifies the feature map type for each prediction.
            ex. [('cmap', 1), ('cmap', 2), ('paf', 1), ('paf', 2)] for 2 stage model
            where 'cmap' is confidence map (or) heatmap and,
            'paf' is part affinity fields.
    """

    np_pred = []
    for output in output_sequence:
        if output[0] == 'cmap':
            np_pred.append(
                np.random.uniform(size=tuple(CMAP_SHAPE)).astype('float32'))
        elif output[0] == 'paf':
            np_pred.append(
                np.random.uniform(size=tuple(PAF_SHAPE)).astype('float32'))
        else:
            raise ValueError("output_sequence must be in ['cmap', 'paf']")

    return np_pred


def make_np_masks(mask_configuration, feat_size):
    """Calculate the masks in numpy.

    Args:
        mask_configuration (list): list of int specifying which output head to mask
            [paf_mask_val, cmap_mask_val]
        feat_size (list): list of int specifying number of channels in each output
            heads [num_paf_features, num_cmap_features]
    """

    np_mask = np.ones(shape=tuple(GT_SHAPE)).astype('float32')
    np_mask[:, :, :, :feat_size[0]] = float(mask_configuration[0])
    np_mask[:, :, :, feat_size[1]:] = float(mask_configuration[1])

    return np_mask


def calculate_np_loss(np_cmap_gt, np_paf_gt, np_preds, np_paf_mask,
                      np_cmap_mask, output_sequence):
    """Calculate the expected losses in numpy.

    Args:
        np_cmap_gt (np.ndarray): ground truth cmap (NHWC).
        np_paf_gt (np.ndarray): ground truth pafmap (NHWC).
        np_preds (list): list of predictions of each head and stage (np.ndarray)
        np_paf_mask (np.ndarray): mask for pafmap (NHWC).
        np_cmap_mask (np.ndarray): mask for cmap (NHWC).
        output_sequence (list): specifies the feature map type for each prediction.
            ex. [('cmap', 1), ('cmap', 2), ('paf', 1), ('paf', 2)] for 2 stage model
            where 'cmap' is confidence map (or) heatmap and,
            'paf' is part affinity fields.
    """

    losses = []
    for idx, pred in enumerate(np_preds):
        if output_sequence[idx][0] == 'cmap':
            loss = np.sum(np.square(((pred - np_cmap_gt) * np_cmap_mask))) / 2
        if output_sequence[idx][0] == 'paf':
            loss = np.sum(np.square(((pred - np_paf_gt) * np_paf_mask))) / 2
        losses.append(loss)
    return losses


def assert_equal_losses(losses, expected_losses):
    """Assert if the losses calculated match the expected losses.

    Args:
        losses (list): list of tensors containing loss for each stage/prediction
        expected_losses (list): list of np.ndarray containing expected loss for
            each stage/prediction.
    """

    assert len(losses) == len(expected_losses)
    for idx in range(len(losses)):
        np.testing.assert_almost_equal(np.mean(losses[idx]),
                                       np.mean(expected_losses[idx]),
                                       decimal=2)


@pytest.mark.parametrize("output_sequence, mask_configuration, feat_size",
                         test_inputs)
def test_bpnet_loss(output_sequence, mask_configuration, feat_size):
    """Function to test BpNet loss.

    Args:
        output_sequence (list): specifies the feature map type for each prediction.
            ex. [('cmap', 1), ('cmap', 2), ('paf', 1), ('paf', 2)] for 2 stage model
            where 'cmap' is confidence map (or) heatmap and,
            'paf' is part affinity fields.
        mask_configuration (list): list of int specifying which output head to mask
            [paf_mask_val, cmap_mask_val]
        feat_size (list): list of int specifying number of channels in each output
            heads [num_paf_features, num_cmap_features]
    """

    # Update params based on test case
    GT_SHAPE[3] = np.sum(feat_size)
    PAF_SHAPE[3] = feat_size[0]
    CMAP_SHAPE[3] = feat_size[1]
    config = Config(feat_size[0], feat_size[1])

    # Generate the groundtruth and prediction.
    np_gt = np.random.uniform(size=tuple(GT_SHAPE)).astype('float32')
    np_preds = make_np_predictions(output_sequence)
    np_mask = make_np_masks(mask_configuration, feat_size)

    np_paf_gt = np_gt[:, :, :, :feat_size[0]]
    np_cmap_gt = np_gt[:, :, :, feat_size[0]:]
    np_paf_mask = np_mask[:, :, :, :feat_size[0]]
    np_cmap_mask = np_mask[:, :, :, feat_size[0]:]

    expected_losses = calculate_np_loss(np_cmap_gt, np_paf_gt, np_preds,
                                        np_paf_mask, np_cmap_mask,
                                        output_sequence)

    # Build loss
    losses_tensor = BpNetLoss()(K.constant(np_gt),
                                [K.constant(np_pred) for np_pred in np_preds],
                                K.constant(np_mask), output_sequence, config.gt_slice_index)

    with tf.Session() as sess:
        losses = sess.run(losses_tensor)

    assert_equal_losses(losses, expected_losses)
