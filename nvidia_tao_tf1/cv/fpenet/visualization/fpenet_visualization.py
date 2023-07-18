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
"""FpeNet Image Visualization utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import tensorflow as tf


class FpeNetVisualizer(object):
    """FpeNetVisualizer object."""

    def __init__(self, checkpoint_dir, num_images=3):
        """Instantiate FpeNetVisualizer object.

        Args:
            checkpoint_dir (str): Path to directory containing checkpoints.
            num_images (int): Number of data images to show on Tensorboard.
        """
        if num_images < 0:
            raise ValueError("FpeNetVisualizer.num_images ({}) is "
                             "not positive.".format(num_images))
        self._checkpoint_dir = os.path.join(checkpoint_dir, 'visualization')
        self._model_id = os.path.basename(os.path.normpath(checkpoint_dir))
        self._num_images = int(num_images)

    def visualize_images(self,
                         image,
                         kpts,
                         name='input_image',
                         data_format='channels_first',
                         viz_phase='training'):
        """Add a 4D tensor to Tensorboard.

        Args:
            image (Tensor): 4D tensor in NCHW or NHWC format.
            kpts (Tensor): 3D tensor of keypoints in (batch x num_keypoints x 2).
            name (str): Image name. Default is 'input_image'.
            data_format (string): Format of the input values.
                Must be either 'channels_first' (NCHW) or 'channels_last' (NHWC).
        """
        def _visualize_image(image, kpts):
                # Do the actual drawing in python
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                for kpt in kpts:
                    x = int(kpt[0])
                    y = int(kpt[1])
                    cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
                image = np.expand_dims(image, axis=0)
                return image

        if self._num_images > 0:
            image = tf.cast(tf.clip_by_value(image, 0., 255.), tf.uint8)
            # Images must be in NHWC format. Convert as needed.
            if data_format == 'channels_first':
                image = tf.transpose(image, (0, 2, 3, 1))
            image_kpts = tf.compat.v1.py_func(_visualize_image, [image[0], kpts[0]], tf.uint8)
            tf.summary.image("kpts_image_" + viz_phase, image_kpts)
            tf.summary.image(name+'_face_' + viz_phase, image, self._num_images)
