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

"""TLT YOLOv3 data sequence."""

import cv2
import numpy as np

from nvidia_tao_tf1.cv.common.dataio.augmentation_lib import (
    aug_flip,
    aug_hsv,
    aug_jitter,
    aug_letterbox_resize
)
from nvidia_tao_tf1.cv.common.dataio.detection_data_sequence import DetectionDataSequence


class YOLOv3DataSequence(DetectionDataSequence):
    """YOLOv3 data sequence."""

    def __init__(self, *args, **kwargs):
        """Init function."""

        super().__init__(*args, **kwargs)

        # randomize input shape
        if self.augmentation_config.randomize_input_shape_period > 0 and self.is_training:
            self._gen_random_shape()
        else:
            self.w_rand = None
            self.h_rand = None
        # Defaults to 8-bit image
        self.image_depth = int(self.augmentation_config.output_depth) or 8
        if self.image_depth not in [8, 16]:
            raise ValueError(
                f"Only 8-bit and 16-bit images are supported, got {self.image_depth}-bit image"
            )

    def _gen_random_shape(self):
        """generate random input shape for all data points."""
        w = self.augmentation_config.output_width / 32.0
        h = self.augmentation_config.output_height / 32.0
        assert self.augmentation_config.randomize_input_shape_period > 0, "Incorrect rand period!"

        # +1 to make sure we have enough random shape
        rand_len = len(self) // self.augmentation_config.randomize_input_shape_period + 1

        self.w_rand = np.random.randint(low=int(round(0.6*w)),
                                        high=int(round(1.5*w + 1.0)),
                                        size=rand_len) * 32
        self.h_rand = np.random.randint(low=int(round(0.6*h)),
                                        high=int(round(1.5*h + 1.0)),
                                        size=rand_len) * 32

        self.w_rand = np.repeat(self.w_rand, self.augmentation_config.randomize_input_shape_period)
        self.h_rand = np.repeat(self.h_rand, self.augmentation_config.randomize_input_shape_period)

    def _preprocessing(self, image, label, output_img_size):
        bboxes = label[:, -4:]

        if self.is_training:
            # Build augmentation pipe.
            image = aug_hsv(image,
                            self.augmentation_config.hue,
                            self.augmentation_config.saturation,
                            self.augmentation_config.exposure,
                            depth=self.image_depth)

            if np.random.rand() < self.augmentation_config.vertical_flip:
                image, bboxes = aug_flip(image, bboxes, ftype=0)

            if np.random.rand() < self.augmentation_config.horizontal_flip:
                image, bboxes = aug_flip(image, bboxes, ftype=1)

            image, bboxes = aug_jitter(image, bboxes,
                                       jitter=self.augmentation_config.jitter,
                                       resize_ar=float(output_img_size[0]) / output_img_size[1])

            image = cv2.resize(image, output_img_size, cv2.INTER_LINEAR)
        else:
            image, bboxes = aug_letterbox_resize(image, bboxes, resize_shape=output_img_size)

        # Finalize
        label[:, -4:] = bboxes
        label = self._filter_invalid_labels(label)

        if self.encode_fn is not None:
            label = self.encode_fn(output_img_size, label)

        return image, label

    def __getitem__(self, batch_idx):
        """Load a full batch."""
        images = []
        labels = []
        raw_labels = []
        if self.w_rand is not None:
            output_img_size = (self.w_rand[batch_idx], self.h_rand[batch_idx])
        else:
            output_img_size = self.output_img_size
        for idx in range(batch_idx * self.batch_size,
                         min(self.n_samples, (batch_idx + 1) * self.batch_size)):
            if self.output_raw_label:
                image, label, raw_label = self._get_single_item(idx, output_img_size)
            else:
                image, label = self._get_single_item(idx, output_img_size)
            images.append(image)
            labels.append(label)
            if self.output_raw_label:
                raw_labels.append(raw_label)
        image_batch, label_batch = self._batch_post_processing(images, labels)
        if self.output_raw_label:
            return image_batch, (label_batch, raw_labels)
        return image_batch, label_batch

    def on_epoch_end(self):
        """shuffle data at end."""
        if self.is_training:
            np.random.shuffle(self.data_inds)
            if self.augmentation_config.randomize_input_shape_period > 0:
                self._gen_random_shape()
