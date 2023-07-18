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

"""Data augmentation helper functions for FasterRCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def random_hflip(image, prob, seed):
    """Random horizontal flip.

    Args:
        image(Tensor): The input image in (C, H, W).
        prob(float): The probability for horizontal flip.
        seed(int): The random seed.

    Returns:
        out_image(Tensor): The output image.
        flipped(boolean Tensor): A boolean scalar tensor to indicate whether flip is
        applied or not. This can be used to manipulate the labels accordingly.
    """

    val = tf.random.uniform([], maxval=1.0, seed=seed)
    is_flipped = tf.cast(
        tf.cond(
            tf.less_equal(val, prob),
            true_fn=lambda: tf.constant(1.0, dtype=tf.float32),
            false_fn=lambda: tf.constant(0.0, dtype=tf.float32)
        ),
        tf.bool
    )
    # CHW to HWC
    image_hwc = tf.transpose(image, (1, 2, 0))
    # flip and to CHW
    flipped_image = tf.transpose(tf.image.flip_left_right(image_hwc), (2, 0, 1))
    out_image = tf.cond(
        is_flipped,
        true_fn=lambda: flipped_image,
        false_fn=lambda: image
    )
    return out_image, is_flipped


def hflip_bboxes(boxes, image_width):
    """Flip the bboxes horizontally.

    Args:
        boxes(Tensor): (N, 4) shaped bboxes in [y1, x1, y2, x2] absolute coordinates.
        image_width(Tensor): image width for calculating the flipped coordinates.

    Returns:
        out_boxes(Tensor): horizontally flipped boxes.
    """

    # x1 becomes new x2, while x2 becomes new x1
    # (N,)
    x1_new = tf.cast(image_width, tf.float32) - 1.0 - boxes[:, 3]
    x2_new = tf.cast(image_width, tf.float32) - 1.0 - boxes[:, 1]
    # (N, 4)
    flipped_boxes = tf.stack([boxes[:, 0], x1_new, boxes[:, 2], x2_new], axis=1)
    # keep all-zero boxes untouched as they are padded boxes
    out_boxes = tf.where(
        tf.cast(tf.reduce_sum(tf.math.abs(boxes), axis=-1), tf.bool),
        x=flipped_boxes,
        y=boxes
    )
    return out_boxes
