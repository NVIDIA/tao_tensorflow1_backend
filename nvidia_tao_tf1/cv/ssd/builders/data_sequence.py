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

"""TLT SSD data sequence."""

import numpy as np

from nvidia_tao_tf1.cv.common.dataio.detection_data_sequence import DetectionDataSequence
from nvidia_tao_tf1.cv.ssd.builders.data_generator.data_augmentation_chain_original_ssd import (
    SSDDataAugmentation
)
from nvidia_tao_tf1.cv.ssd.builders.data_generator.object_detection_2d_geometric_ops import Resize


class SSDDataSequence(DetectionDataSequence):
    """SSD data sequence."""

    def __init__(self, dataset_config, *args, **kwargs):
        """Init function."""
        super().__init__(dataset_config=dataset_config, *args, **kwargs)

        self.output_height = self.augmentation_config.output_height
        self.output_width = self.augmentation_config.output_width

        # mapping class to 1-based integer
        mapping_dict = dataset_config.target_class_mapping
        self.classes = sorted({str(x).lower() for x in mapping_dict.values()})
        val_class_mapping = dict(
            zip(self.classes, range(1, len(self.classes)+1)))
        self.class_mapping = {key.lower(): val_class_mapping[str(val.lower())]
                              for key, val in mapping_dict.items()}

    def set_encoder(self, encode_fn):
        '''Set label encoder.'''
        self.encode_fn = encode_fn

    def _load_gt_label(self, label_path):
        """Load Kitti labels.

        Returns:
            [class_idx, is_difficult, x_min, y_min, x_max, y_max]
        """
        entries = open(label_path, 'r').read().strip().split('\n')
        results = []
        for entry in entries:
            items = entry.strip().split()
            if len(items) < 9:
                continue
            items[0] = items[0].lower()
            if items[0] not in self.class_mapping:
                continue
            label = [self.class_mapping[items[0]], 1 if int(
                items[2]) != 0 else 0, *items[4:8]]
            results.append([float(x) for x in label])

        return np.array(results).reshape(-1, 6)

    def _preprocessing(self, image, label, output_img_size):
        '''
        SSD-style data augmentation will be performed in training.

        And in evaluation/inference phase, only resize will be performed;
        '''
        # initial SSD augmentation parameters:
        if self.is_training:
            augmentation_func = \
                SSDDataAugmentation(img_height=self.augmentation_config.output_height,
                                    img_width=self.augmentation_config.output_width,
                                    rc_min=self.augmentation_config.random_crop_min_scale or 0.3,
                                    rc_max=self.augmentation_config.random_crop_max_scale or 1.0,
                                    rc_min_ar=self.augmentation_config.random_crop_min_ar or 0.5,
                                    rc_max_ar=self.augmentation_config.random_crop_max_ar or 2.0,
                                    zo_min=self.augmentation_config.zoom_out_min_scale or 1.0,
                                    zo_max=self.augmentation_config.zoom_out_max_scale or 4.0,
                                    b_delta=self.augmentation_config.brightness,
                                    c_delta=self.augmentation_config.contrast,
                                    s_delta=self.augmentation_config.saturation,
                                    h_delta=self.augmentation_config.hue,
                                    flip_prob=self.augmentation_config.random_flip,
                                    background=(123.68, 116.779, 103.939))
        else:
            augmentation_func = Resize(height=self.augmentation_config.output_height,
                                       width=self.augmentation_config.output_width)

        if self.is_training:
            bboxes = label[:, -4:]
            cls_id = label[:, 0:1]
            label = np.concatenate((cls_id, bboxes), axis=-1)
            image, label = augmentation_func(image, label)
        else:
            bboxes = label[:, -4:]
            cls_id = label[:, 0:1]
            temp_label = np.concatenate((cls_id, bboxes), axis=-1)
            image, temp_label = augmentation_func(image, temp_label)

            # Finalize
            label[:, -4:] = temp_label[:, -4:]

        label = self._filter_invalid_labels(label)

        if self.encode_fn is not None:
            label = self.encode_fn(label)

        return image, label

    def _filter_invalid_labels(self, labels):
        """filter out invalid labels.

        Arg:
            labels: size (N, 5) or (N, 6), where bboxes is normalized to 0~1.
        Returns:
            labels: size (M, 5) or (N, 6), filtered bboxes with clipped boxes.
        """

        # clip
        # -4     -3    -2     -1
        x_coords = labels[:, [-4, -2]]
        x_coords = np.clip(x_coords, 0, self.output_width - 1)
        labels[:, [-4, -2]] = x_coords
        y_coords = labels[:, [-3, -1]]
        y_coords = np.clip(y_coords, 0, self.output_height - 1)
        labels[:, [-3, -1]] = y_coords

        # exclude invalid boxes
        x_cond = labels[:, -2] - labels[:, -4] > 1e-3
        y_cond = labels[:, -1] - labels[:, -3] > 1e-3

        return labels[x_cond & y_cond]

    def _get_single_item(self, idx, output_img_size):
        """Load and process single image and its label."""

        image = self._load_gt_image(self.image_paths[self.data_inds[idx]])
        label = self._load_gt_label(self.label_paths[self.data_inds[idx]])

        return self._preprocessing(image, label, output_img_size)
