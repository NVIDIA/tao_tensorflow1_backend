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

"""Helper function to image processing."""

import random
import cv2
import numpy as np


def reverse_color(img):
    '''reverse color.'''

    img = 255 - img
    return img


class GaussianBlur:
    '''Gaussian blur.'''

    def __init__(self, kernel_list):
        """init blur."""
        self.kernel_list = kernel_list

    def __call__(self, img):
        """Gaussian blur the image."""
        kernel_size = self.kernel_list[random.randint(0, len(self.kernel_list)-1)]
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


class CenterRotate:
    '''rotate within degree at center of license plate.'''

    def __init__(self, degree):
        """init rotation degree."""

        self.degree = degree

    def __call__(self, img):
        """rotate the image."""

        degree = random.randint(-self.degree,
                                self.degree)
        rows, cols = img.shape[0:2]
        m = cv2.getRotationMatrix2D((cols/2, rows/2), degree, 1)
        img = cv2.warpAffine(img, m, (cols, rows))

        return img


def preprocess(batch_img,
               output_width,
               output_height,
               is_training=True,
               output_channel=3,
               keep_original_prob=0.3,
               max_rotate_degree=5,
               rotate_prob=0.5,
               gaussian_kernel_size=None,
               blur_prob=0.5,
               reverse_color_prob=0.5
               ):
    """preprocess pipeline of the images."""

    # if it is training, then augmented
    # resize and scale: cv2 resize((width, height))
    # Augmentation: blur, reverse color, rotate
    augmented_batch_img = []
    if is_training:
        transform_op_list = []
        transform_prob_list = []

        if max_rotate_degree > 0:
            rotate = CenterRotate(max_rotate_degree)
            transform_op_list.append(rotate)
            transform_prob_list.append(rotate_prob)

        if (len(gaussian_kernel_size) > 0) and (blur_prob > 0):
            blur = GaussianBlur(gaussian_kernel_size)
            transform_op_list.append(blur)
            transform_prob_list.append(blur_prob)

        if reverse_color_prob > 0:
            transform_op_list.append(reverse_color)
            transform_prob_list.append(reverse_color_prob)

        for img in batch_img:
            if random.random() > keep_original_prob:
                for transform, transform_prob in zip(transform_op_list, transform_prob_list):
                    if random.random() > transform_prob:
                        continue
                    img = transform(img)
            augmented_batch_img.append(img)
    else:
        augmented_batch_img = batch_img

    # batch_resized_img = np.array([(cv2.resize(img, (96, 48)))/255.0
    batch_resized_img = np.array([(cv2.resize(img, (output_width, output_height)))/255.0
                                  for img in augmented_batch_img],
                                 dtype=np.float32)
    if output_channel == 1:
        batch_resized_img = batch_resized_img[..., np.newaxis]
    # transpose image batch from NHWC (0, 1, 2, 3) to NCHW (0, 3, 1, 2)
    batch_resized_img = batch_resized_img.transpose(0, 3, 1, 2)

    return batch_resized_img
