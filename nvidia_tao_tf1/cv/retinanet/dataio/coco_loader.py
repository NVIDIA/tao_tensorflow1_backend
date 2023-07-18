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

"""TLT detection data sequence."""

import numpy as np

from nvidia_tao_tf1.cv.common.dataio.coco_data_sequence import CocoDataSequence
from nvidia_tao_tf1.cv.ssd.builders.data_generator.data_augmentation_chain_original_ssd import \
    SSDDataAugmentation
from nvidia_tao_tf1.cv.ssd.builders.data_generator.object_detection_2d_geometric_ops import Resize


class RetinaCocoDataSequence(CocoDataSequence):
    """RetinaNet data loader for data with COCO format."""

    def __init__(self, dataset_config, *args, **kwargs):
        """Class initialization."""
        super().__init__(dataset_config=dataset_config, *args, **kwargs)

    def _preprocessing(self, image, label, enable_mask=False):
        '''
        SSD-style data augmentation will be performed in training.

        And in evaluation/inference phase, only resize will be performed;
        '''
        # initialize augmentation
        if self.is_training:
            augmentation_func = \
                SSDDataAugmentation(img_height=self.augmentation_config.output_height,
                                    img_width=self.augmentation_config.output_width,
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

    def _batch_post_processing(self, images, labels):
        """Post processing for a batch."""

        images = np.array(images)
        # RGB -> BGR, channels_last -> channels_first
        images = images[..., [2, 1, 0]].transpose(0, 3, 1, 2)
        # subtract imagenet mean
        images -= np.array([[[[103.939]], [[116.779]], [[123.68]]]])

        # try to make labels a numpy array
        is_make_array = True
        x_shape = None
        for x in labels:
            if not isinstance(x, np.ndarray):
                is_make_array = False
                break
            if x_shape is None:
                x_shape = x.shape
            elif x_shape != x.shape:
                is_make_array = False
                break

        if is_make_array:
            labels = np.array(labels)

        return images, labels
