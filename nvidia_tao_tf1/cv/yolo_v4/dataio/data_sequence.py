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

"""TLT YOLOv4 data sequence."""

import cv2
import numpy as np

from nvidia_tao_tf1.cv.common.dataio.augmentation_lib import (
    aug_flip,
    aug_hsv,
    aug_jitter,
    aug_letterbox_resize,
    aug_random_crop
)
from nvidia_tao_tf1.cv.yolo_v3.dataio.data_sequence import YOLOv3DataSequence


class YOLOv4DataSequence(YOLOv3DataSequence):
    """YOLOv4 data sequence."""

    def _build_single_training_img(self, image, label, resize_ar, use_jitter, jitter_or_min_ratio):
        '''Helper function for mosaic technique.

        Args:
            image: np array
            label: np array
            resize_ar: image resize aspect ratio,
            use_jitter: false - use random crop. true - use jitter
            jitter_or_min_ratio: if use_jitter, this value should be jitter
                otherwise, this value should be min_ratio for crop
        '''

        bboxes = label[:, -4:]

        image = aug_hsv(image,
                        self.augmentation_config.hue,
                        self.augmentation_config.saturation,
                        self.augmentation_config.exposure,
                        depth=self.image_depth)

        if np.random.rand() < self.augmentation_config.vertical_flip:
            image, bboxes = aug_flip(image, bboxes, ftype=0)

        if np.random.rand() < self.augmentation_config.horizontal_flip:
            image, bboxes = aug_flip(image, bboxes, ftype=1)

        if use_jitter:
            image, bboxes = aug_jitter(image, bboxes, jitter_or_min_ratio, resize_ar)
        else:
            image, bboxes = aug_random_crop(image, bboxes, resize_ar, jitter_or_min_ratio)

        label[:, -4:] = bboxes
        label = self._filter_invalid_labels(label)

        return image, label

    def _resize_label(self, label, t_shift, l_shift, h_ratio, w_ratio):
        """Helper function to resize labels in mosaic."""

        # xmin
        label[:, -4] = l_shift + w_ratio * label[:, -4]
        # xmax
        label[:, -2] = l_shift + w_ratio * label[:, -2]
        # ymin
        label[:, -3] = t_shift + h_ratio * label[:, -3]
        # ymax
        label[:, -1] = t_shift + h_ratio * label[:, -1]

    def _preprocessing(self, image, label, output_img_size):
        if self.is_training:
            if np.random.rand() < self.augmentation_config.mosaic_prob:
                # do mosaic augmentation
                mosaic_min_ratio = self.augmentation_config.mosaic_min_ratio
                assert 0 < mosaic_min_ratio < 0.5, "mosaic_min_ratio must in range (0, 0.5)"
                # step 1. determine border
                y_border = np.random.randint(int(output_img_size[1] * mosaic_min_ratio),
                                             1 + int(output_img_size[1] * (1.0 - mosaic_min_ratio)))
                x_border = np.random.randint(int(output_img_size[0] * mosaic_min_ratio),
                                             1 + int(output_img_size[0] * (1.0 - mosaic_min_ratio)))

                mosaic_image = np.zeros((output_img_size[1], output_img_size[0], 3))
                mosaic_label = []

                # Load 3 additional images from training data
                additional_ids = np.random.randint(self.n_samples, size=3)

                # step 2. process images
                # left-top

                image, label = self._build_single_training_img(
                    image, label,
                    x_border / float(y_border),
                    False, 1.0 - self.augmentation_config.jitter)

                mosaic_image[:y_border, :x_border] = cv2.resize(image,
                                                                (x_border, y_border),
                                                                cv2.INTER_LINEAR)

                self._resize_label(label, 0, 0, y_border / float(output_img_size[1]),
                                   x_border / float(output_img_size[0]))

                mosaic_label.append(label)

                # right-top
                image, label = self._get_single_item_raw(additional_ids[0])

                image, label = self._build_single_training_img(
                    image, label,
                    (output_img_size[0] - x_border) / float(y_border),
                    False, 1.0 - self.augmentation_config.jitter)

                mosaic_image[:y_border, x_border:] = cv2.resize(image,
                                                                (output_img_size[0] - x_border,
                                                                 y_border),
                                                                cv2.INTER_LINEAR)

                self._resize_label(label, 0, x_border / float(output_img_size[0]),
                                   y_border / float(output_img_size[1]),
                                   1.0 - (x_border / float(output_img_size[0])))

                mosaic_label.append(label)

                # left-bottom
                image, label = self._get_single_item_raw(additional_ids[1])

                image, label = self._build_single_training_img(
                    image, label,
                    x_border / float(output_img_size[1] - y_border),
                    False, 1.0 - self.augmentation_config.jitter)

                mosaic_image[y_border:, :x_border] = cv2.resize(image,
                                                                (x_border,
                                                                 output_img_size[1] - y_border),
                                                                cv2.INTER_LINEAR)

                self._resize_label(label, y_border / float(output_img_size[1]), 0,
                                   1.0 - (y_border / float(output_img_size[1])),
                                   x_border / float(output_img_size[0]))

                mosaic_label.append(label)

                # right-bottom
                image, label = self._get_single_item_raw(additional_ids[2])

                image, label = self._build_single_training_img(
                    image, label,
                    (output_img_size[0] - x_border) / float(output_img_size[1] - y_border),
                    False, 1.0 - self.augmentation_config.jitter)

                mosaic_image[y_border:, x_border:] = cv2.resize(image,
                                                                (output_img_size[0] - x_border,
                                                                 output_img_size[1] - y_border),
                                                                cv2.INTER_LINEAR)

                self._resize_label(label, y_border / float(output_img_size[1]),
                                   x_border / float(output_img_size[0]),
                                   1.0 - (y_border / float(output_img_size[1])),
                                   1.0 - (x_border / float(output_img_size[0])))

                mosaic_label.append(label)

                label = np.concatenate(mosaic_label, axis=0)
                image = mosaic_image

            else:
                # YOLOv3 style loading
                image, label = self._build_single_training_img(
                    image, label,
                    output_img_size[0] / float(output_img_size[1]),
                    True, self.augmentation_config.jitter)

                image = cv2.resize(image, output_img_size, cv2.INTER_LINEAR)
        else:
            bboxes = label[:, -4:]
            image, bboxes = aug_letterbox_resize(image, bboxes, resize_shape=output_img_size)
            label[:, -4:] = bboxes

        label = self._filter_invalid_labels(label)
        raw_label = label[:, 2:6]
        if self.encode_fn is not None:
            label = self.encode_fn(output_img_size, label)
        if self.output_raw_label:
            return image, label, raw_label
        return image, label
