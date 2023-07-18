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

import logging
import os
import numpy as np

from nvidia_tao_tf1.cv.common.dataio.base_data_sequence import BaseDataSequence


logging.basicConfig(
    format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
    level="DEBUG"
)
logger = logging.getLogger(__name__)


class DetectionDataSequence(BaseDataSequence):
    """Abstract class for TLT detection network.

    To use dataloader:
        1. call __init__(configs)
        2. call add_source(image_folder, label_folder) to add sources
        3. Use data generator in keras model.fit_generator()

    Functions below must be implemented in derived classes:
        1. _preprocessing
    """

    def __init__(self,
                 dataset_config,
                 augmentation_config,
                 batch_size=10,
                 is_training=True,
                 encode_fn=None,
                 root_path=None,
                 output_raw_label=False):
        """Class initialization."""
        self.image_paths = []
        self.label_paths = []

        # mapping class to 1-based integer
        mapping_dict = dataset_config.target_class_mapping
        self.classes = sorted({str(x).lower() for x in mapping_dict.values()})
        val_class_mapping = dict(zip(self.classes, range(len(self.classes))))
        self.class_mapping = {key.lower(): val_class_mapping[str(val.lower())]
                              for key, val in mapping_dict.items()}

        # load data sources
        if is_training:
            data_sources = dataset_config.data_sources
        else:
            data_sources = dataset_config.validation_data_sources

        for data_source in data_sources:
            self._add_source(
                    os.path.join(root_path, data_source.image_directory_path)
                    if root_path else data_source.image_directory_path,
                    os.path.join(root_path, data_source.label_directory_path)
                    if root_path else data_source.label_directory_path)
        # use numpy array to accelerate
        self.image_paths = np.array(self.image_paths)
        self.label_paths = np.array(self.label_paths)
        self.data_inds = np.arange(len(self.image_paths))

        if is_training:
            np.random.shuffle(self.data_inds)

        self.is_training = is_training
        self.batch_size = batch_size
        self.output_img_size = (augmentation_config.output_width, augmentation_config.output_height)
        self.augmentation_config = augmentation_config
        self.exclude_difficult = is_training and (not dataset_config.include_difficult_in_training)

        self.encode_fn = encode_fn
        self.n_samples = len(self.data_inds)
        self.output_raw_label = output_raw_label
        self.image_depth = 8

    def _add_source(self, image_folder, label_folder):
        """Add Kitti sources."""

        img_paths = os.listdir(image_folder)
        label_paths = set(os.listdir(label_folder))
        supported_img_format = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

        for img_path in img_paths:
            # Only add valid items to paths
            filename, img_ext = os.path.splitext(img_path)
            if img_ext in supported_img_format and filename + '.txt' in label_paths:
                self.image_paths.append(os.path.join(image_folder, img_path))
                self.label_paths.append(os.path.join(label_folder, filename + '.txt'))

    def __len__(self):
        """Get length of Sequence."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))

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

    def _filter_invalid_labels(self, labels):
        """filter out invalid labels.

        Arg:
            labels: size (N, 6), where bboxes is normalized to 0~1.
        Returns:
            labels: size (M, 6), filtered bboxes with clipped boxes.
        """

        labels[:, -4:] = np.clip(labels[:, -4:], 0, 1)

        # exclude invalid boxes
        difficult_cond = (labels[:, 1] < 0.5) | (not self.exclude_difficult)
        if np.any(difficult_cond == 0):
            logger.warning(
                "Got label marked as difficult(occlusion > 0), "
                "please set occlusion field in KITTI label to 0 "
                "or set `dataset_config.include_difficult_in_training` to True "
                "in spec file, if you want to include it in training."
            )
        x_cond = labels[:, 4] - labels[:, 2] > 1e-3
        y_cond = labels[:, 5] - labels[:, 3] > 1e-3

        return labels[difficult_cond & x_cond & y_cond]

    def _get_single_item_raw(self, idx):
        """Load single image and its label."""

        image = self._load_gt_image(self.image_paths[self.data_inds[idx]])
        label = self._load_gt_label(self.label_paths[self.data_inds[idx]])

        # change bbox to 0~1
        h, w, _ = image.shape
        label[:, 2] /= w
        label[:, 3] /= h
        label[:, 4] /= w
        label[:, 5] /= h

        return image, label

    def _get_single_item(self, idx, output_img_size):
        """Load and process single image and its label."""

        image, label = self._get_single_item_raw(idx)

        return self._preprocessing(image, label, output_img_size)

    def _batch_post_processing(self, images, labels):
        """Post processing for a batch."""

        images = np.array(images)
        # RGB -> BGR, channels_last -> channels_first
        images = images[..., [2, 1, 0]].transpose(0, 3, 1, 2)
        img_mean = self.augmentation_config.image_mean
        if self.augmentation_config.output_channel == 3:
            assert self.image_depth == 8, (
                f"RGB images only support 8-bit depth, got {self.image_depth}, "
                "please check `augmentation_config.output_depth` in spec file"
            )
            if img_mean:
                bb, gg, rr = img_mean['b'], img_mean['g'], img_mean['r']
            else:
                bb, gg, rr = 103.939, 116.779, 123.68
        else:
            if img_mean:
                bb, gg, rr = img_mean['l'], img_mean['l'], img_mean['l']
            elif self.image_depth == 8:
                bb, gg, rr = 117.3786, 117.3786, 117.3786
            elif self.image_depth == 16:
                # 117.3786 * 256
                bb, gg, rr = 30048.9216, 30048.9216, 30048.9216
            else:
                raise ValueError(
                    f"Unsupported image depth: {self.image_depth}, should be 8 or 16, "
                    "please check `augmentation_config.output_depth` in spec file"
                )
        # subtract imagenet mean
        images -= np.array([[[[bb]], [[gg]], [[rr]]]])
        if self.augmentation_config.output_channel == 1:
            # See conversion: https://pillow.readthedocs.io/en/3.2.x/reference/Image.html
            bgr_ = np.array([0.1140, 0.5870, 0.2990]).reshape(1, 3, 1, 1)
            images = np.sum(images * bgr_, axis=1, keepdims=True)

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

    def __getitem__(self, batch_idx):
        """Load a full batch."""
        images = []
        labels = []

        for idx in range(batch_idx * self.batch_size,
                         min(self.n_samples, (batch_idx + 1) * self.batch_size)):
            image, label = self._get_single_item(idx, self.output_img_size)
            images.append(image)
            labels.append(label)

        return self._batch_post_processing(images, labels)

    def on_epoch_end(self):
        """shuffle data at end."""
        if self.is_training:
            np.random.shuffle(self.data_inds)
