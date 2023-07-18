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
"""Modulus Random Blur Processor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.augment.blur import Blur
from nvidia_tao_tf1.core.processors.augment.pixel_removal import PixelRemoval


class RandomBlur(Blur):
    """Random Blur Transformation class."""

    @save_args
    def __init__(self, random=True, **kwargs):
        """__init__ method."""
        super(RandomBlur, self).__init__(random=random, **kwargs)
        self.pixel_remover = PixelRemoval(random)

    def call(self, images, size, std, blur_pct, blur_max_block, prob=1.0):
        """
        Randomly blur patches of the image.

        Args:
            images (tensor): The images to augment in NCHW format.
            size (int): The largest size for the blur filter if random.
                If not random, then this is the size of the filter to be used to
                blur the image.
            std (float): The maximum standard deviation of the gaussian kernel.
                If not random, then this is the standard deviation to be used to
                blur the image.
            blur_pct (float): The percentage of pixels to blur.
            blur_max_block (float): The maximum block size with which to group blurred pixels.
            prob (float): The probability of applying the augmentation. Only used if
                random is set.

        Outputs:
            The randomly blurred image.
        """
        fully_blurred = self._gaussian_blur(images, size=size, std=std)
        blur_condition = self.pixel_remover.make_selection_condition(
            pct=blur_pct, max_block=blur_max_block, shape=images.shape
        )
        blurred = tf.compat.v1.where(blur_condition, fully_blurred, images)

        if self.random:
            application_prob = tf.random.uniform(shape=[], maxval=1.0)
            no_aug_cond = tf.greater(application_prob, prob)
            return tf.cond(
                pred=no_aug_cond, true_fn=lambda: images, false_fn=lambda: blurred
            )
        return blurred
