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

import os
import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from nvidia_tao_tf1.cv.common.dataio.base_data_sequence import BaseDataSequence


class CocoDataSequence(BaseDataSequence):
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
                 augmentation_config=None,
                 batch_size=10,
                 is_training=True,
                 encode_fn=None,
                 enable_mask=False,
                 root_path=None):
        """Class initialization."""
        self.coco = []
        # load data sources
        if is_training:
            data_sources = dataset_config.data_sources
            set_name = 'train2017'
        else:
            data_sources = dataset_config.validation_data_sources
            set_name = 'val2017'

        for data_source in data_sources:
            self._add_source(
                os.path.join(root_path, data_source.image_directory_path)
                if root_path else data_source.image_directory_path,
                os.path.join(root_path, data_source.label_directory_path)
                if root_path else data_source.label_directory_path,
                set_name)

        self.load_classes()
        # use numpy array to accelerate
        self.image_ids = self.coco.getImgIds()

        if is_training:
            np.random.shuffle(self.image_ids)

        self.is_training = is_training
        self.enable_mask = enable_mask
        self.batch_size = batch_size
        self.augmentation_config = augmentation_config
        self.output_height = self.augmentation_config.output_height
        self.output_width = self.augmentation_config.output_width
        self.output_img_size = (self.output_width, self.output_height)
        self.encode_fn = encode_fn
        self.n_samples = len(self.image_ids)
        print("Number of images: {}".format(self.n_samples))

    def set_encoder(self, encode_fn):
        '''Set label encoder.'''
        self.encode_fn = encode_fn

    def _add_source(self, image_folder, label_folder, set_name='train2017'):
        """Add COCO sources."""

        self.raw_image_dir = image_folder
        self.coco = COCO(os.path.join(label_folder, 'instances_' + set_name + '.json'))

    def __len__(self):
        """Get length of Sequence."""
        return int(np.ceil(self.n_samples / self.batch_size))

    def read_image_rgb(self, path):
        """Read an image in BGR format.

        Args
            path: Path to the image.
        """
        image = np.ascontiguousarray(Image.open(path).convert('RGB'))
        return image.astype(np.float32)

    def load_classes(self):
        """create class mapping."""
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes) + 1
            self.classes[c['name']] = len(self.classes) + 1

    def coco_label_to_label(self, coco_label):
        """coco label to label mapping."""
        return self.coco_labels_inverse[coco_label]

    def _load_gt_image(self, image_index):
        """Load image."""
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.raw_image_dir, image_info['file_name'])
        return self.read_image_rgb(path)

    def _load_gt_label(self, image_index):
        """Load COCO labels.

        Returns:
            [class_idx, is_difficult, x_min, y_min, x_max, y_max]
            where is_diffcult is hardcoded to 0 in the current COCO GT labels.
        """
        # get image info
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]

        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = {
            'labels': np.empty((0,)),
            'bboxes': np.empty((0, 4)),
            'masks': [],
        }

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            # return empty annotations
            return np.empty((0, 6))

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for _, a in enumerate(coco_annotations):
            if 'segmentation' not in a:
                raise ValueError('Expected \'segmentation\' key in annotation, got: {}'.format(a))

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotations['labels'] = np.concatenate(
                [annotations['labels'], [self.coco_label_to_label(a['category_id'])]], axis=0)
            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
                a['bbox'][0],
                a['bbox'][1],
                a['bbox'][0] + a['bbox'][2],
                a['bbox'][1] + a['bbox'][3],
            ]]], axis=0)

            if self.enable_mask:
                mask = np.zeros((image_info['height'], image_info['width'], 1), dtype=np.uint8)
                for seg in a['segmentation']:
                    points = np.array(seg).reshape((len(seg) // 2, 2)).astype(int)

                    # draw mask
                    cv2.fillPoly(mask, [points.astype(int)], (1,))

                annotations['masks'].append(mask.astype(float))

        labels = np.expand_dims(annotations['labels'], axis=-1)
        return np.concatenate((labels, np.full_like(labels, 0), annotations['bboxes']), axis=1)

    def _get_single_item(self, idx):
        """Load and process single image and its label."""

        image = self._load_gt_image(idx)
        label = self._load_gt_label(idx)
        image, label = self._preprocessing(image, label, self.enable_mask)
        return image, label

    def __getitem__(self, batch_idx):
        """Load a full batch."""
        images = []
        labels = []

        for idx in range(batch_idx * self.batch_size,
                         min(self.n_samples, (batch_idx + 1) * self.batch_size)):
            image, label = self._get_single_item(idx)
            images.append(image)
            labels.append(label)

        return self._batch_post_processing(images, labels)

    def on_epoch_end(self):
        """shuffle data at end."""
        if self.is_training:
            np.random.shuffle(self.image_ids)

    def _batch_post_processing(self, images, labels):
        """Post processing for a batch."""
        return images, labels
