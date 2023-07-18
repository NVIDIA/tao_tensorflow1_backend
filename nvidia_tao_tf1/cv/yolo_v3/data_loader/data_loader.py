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

"""IVA YOLO v3 data loader builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.cv.ssd.utils.tensor_utils import get_non_empty_rows_2d_sparse
from nvidia_tao_tf1.cv.yolo_v3.data_loader.augmentation import (
    outer_augmentations
)
from nvidia_tao_tf1.cv.yolo_v3.data_loader.yolo_v3_data_loader import build_dataloader


def unstack_images(images, shapes, bs):
    """unstack images into a list of images."""
    images_list = []
    for b_idx in range(bs):
        image = images[b_idx, ...]
        shape = shapes[b_idx, ...]
        image = image[0:shape[0], 0:shape[1], ...]
        if image.dtype != tf.float32:
            image = tf.cast(image, tf.float32)
        images_list.append(image)
    return images_list


class YOLOv3DataPipe:
    """
    Data loader class.

    DataLoader can be used in two ways:
    1. build groundtruth image and label TF tensors. Those two tensors can be
        directly used for training.
    2. build a generator that yields image and label numpy arrays. In this case,
        a TF session needs to be passed into the class initializer.
    """

    def __init__(self,
                 experiment_spec,
                 label_encoder=None,
                 training=True,
                 sess=None,
                 h_tensor=None,
                 w_tensor=None,
                 visualizer=None,
                 rank=0):
        """
        Data loader init function.

        Arguments:
            experiment_spec: The loaded config pb2.
            label_encoder (function, optional): If passed in, groundtruth label will be encoded.
            training (bool): Return training set or validation set.
            sess (TF Session): Required if generator() function needs to be called. Otherwise, just
                pass None.
            visualizer(object): The Visualizer object.
            rank(int): Horovod rank.
        """
        dataset_proto = experiment_spec.dataset_config
        self._exclude_difficult = not dataset_proto.include_difficult_in_training
        dataloader = build_dataloader(
            dataset_proto=dataset_proto,
            augmentation_proto=experiment_spec.augmentation_config,
            h_tensor=h_tensor,
            w_tensor=w_tensor,
            training=training
        )
        self.dataloader = dataloader
        if training:
            batch_size = experiment_spec.training_config.batch_size_per_gpu
        else:
            batch_size = experiment_spec.eval_config.batch_size
        self.batch_size = batch_size
        self.images, self.ground_truth_labels, self.shapes, self.num_samples = \
            dataloader.get_dataset_tensors(
                batch_size
            )
        # original images for debugging
        self._images = self.images
        if self.num_samples == 0:
            return
        self.n_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        cls_mapping_dict = experiment_spec.dataset_config.target_class_mapping
        self.classes = sorted({str(x) for x in cls_mapping_dict.values()})
        cls_map = tao_core.processors.LookupTable(
            keys=self.classes,
            values=list(range(len(self.classes))),
            default_value=-1)
        cls_map.build()
        self.label_encoder = label_encoder

        gt_labels = []
        source_classes = self.ground_truth_labels.object_class
        mapped_classes = tf.SparseTensor(
            values=cls_map(source_classes.values),
            indices=source_classes.indices,
            dense_shape=source_classes.dense_shape
        )
        mapped_labels = self.ground_truth_labels._replace(object_class=mapped_classes)
        valid_indices = tf.not_equal(mapped_classes.values, -1)
        filtered_labels = mapped_labels.filter(valid_indices)
        filtered_obj_ids = tf.sparse.reshape(filtered_labels.object_class, [batch_size, -1, 1])
        filtered_coords = tf.sparse.reshape(filtered_labels.vertices.coordinates,
                                            [batch_size, -1, 4])
        filtered_occlusion = tf.sparse.reshape(
            filtered_labels.occlusion,
            [batch_size, -1, 1]
        )
        filtered_obj_ids = tf.sparse.SparseTensor(
            values=tf.cast(tf.round(filtered_obj_ids.values), tf.float32),
            indices=filtered_obj_ids.indices,
            dense_shape=filtered_obj_ids.dense_shape
        )
        filtered_coords = tf.sparse.SparseTensor(
            values=tf.cast(filtered_coords.values, tf.float32),
            indices=filtered_coords.indices,
            dense_shape=filtered_coords.dense_shape
        )
        filtered_occlusion = tf.sparse.SparseTensor(
            values=tf.cast(filtered_occlusion.values, tf.float32),
            indices=filtered_occlusion.indices,
            dense_shape=filtered_occlusion.dense_shape,
        )
        labels_all = tf.sparse.concat(
            axis=-1,
            sp_inputs=[filtered_obj_ids, filtered_occlusion, filtered_coords]
        )
        labels_split = tf.sparse.split(sp_input=labels_all, num_split=batch_size, axis=0)
        labels_split = [tf.sparse.reshape(x, [-1, 6]) for x in labels_split]
        labels = [tf.sparse.to_dense(get_non_empty_rows_2d_sparse(x)) for x in labels_split]
        for l_idx, l in enumerate(labels):
            obj_id = tf.cast(l[:, 0], tf.float32)
            is_difficult = tf.cast(l[:, 1], tf.float32)
            x1 = l[:, 2] / tf.cast(self.shapes[l_idx, 1], tf.float32)
            x2 = l[:, 4] / tf.cast(self.shapes[l_idx, 1], tf.float32)
            y1 = l[:, 3] / tf.cast(self.shapes[l_idx, 0], tf.float32)
            y2 = l[:, 5] / tf.cast(self.shapes[l_idx, 0], tf.float32)
            # only select valid labels
            select = tf.logical_and(
                tf.not_equal(obj_id, -1),
                tf.logical_and(tf.less(x1, x2), tf.less(y1, y2))
            )
            label = tf.stack([obj_id, is_difficult, x1, y1, x2, y2], axis=1)
            label = tf.boolean_mask(label, select)
            # exclude difficult boxes is forced to do so
            if self._exclude_difficult:
                label = tf.boolean_mask(
                    label,
                    tf.math.logical_not(tf.cast(label[:, 1], tf.bool))
                )
            gt_labels.append(label)
        self.gt_labels = gt_labels
        self.frame_ids = self.ground_truth_labels.frame_id
        # images
        images_list = unstack_images(
            self.images,
            self.shapes,
            self.batch_size
        )
        ratio = tf.cast(w_tensor, tf.float32) / tf.cast(h_tensor, tf.float32)
        augmented_images = []
        augmented_labels = []
        for idx, img in enumerate(images_list):
            if training:
                aug_img, aug_label = outer_augmentations(
                    img,
                    self.gt_labels[idx][:, 2:6],
                    ratio,
                    experiment_spec.augmentation_config
                )
                aug_img = tf.image.resize_images(aug_img, [h_tensor, w_tensor])
            else:
                aug_img, aug_label = img, self.gt_labels[idx][:, 2:6]
            aug_img.set_shape([None, None, 3])
            aug_img = tf.transpose(aug_img, (2, 0, 1))
            aug_label = tf.concat([self.gt_labels[idx][:, 0:2], aug_label], axis=-1)
            # filter out bad boxes after augmentation
            aug_x1 = aug_label[:, 2]
            aug_x2 = aug_label[:, 4]
            aug_y1 = aug_label[:, 3]
            aug_y2 = aug_label[:, 5]
            # only select valid labels
            select = tf.logical_and(
                aug_x2 - aug_x1 > 1e-3,
                aug_y2 - aug_y1 > 1e-3
            )
            aug_label = tf.boolean_mask(aug_label, select)
            augmented_images.append(aug_img)
            augmented_labels.append(aug_label)
        self.images = tf.stack(augmented_images, axis=0)

        num_channels = experiment_spec.augmentation_config.output_channel
        # See conversion: https://pillow.readthedocs.io/en/3.2.x/reference/Image.html
        bgr_ = tf.reshape(
            tf.constant([0.1140, 0.5870, 0.2990], dtype=tf.float32),
            (1, 3, 1, 1)
        )
        # Vis the augmented images in TensorBoard, only rank 0
        if rank == 0 and visualizer is not None:
            if visualizer.enabled:
                if num_channels == 3:
                    aug_images = self.images
                else:
                    # Project RGB to grayscale
                    aug_images = tf.reduce_sum(
                        self.images * bgr_,
                        axis=1,
                        keepdims=True
                    )
                aug_images = tf.transpose(aug_images, (0, 2, 3, 1))
                _max_box_num = tf.shape(augmented_labels[0])[0]
                for _aug_label in augmented_labels[1:]:
                    _max_box_num = tf.reduce_max(
                        tf.stack([_max_box_num, tf.shape(_aug_label)[0]], axis=0)
                    )
                _aug_label_list = []
                for _aug_label in augmented_labels:
                    _num_pad = _max_box_num - tf.shape(_aug_label)[0]
                    _aug_label_list.append(tf.pad(_aug_label, [(0, _num_pad), (0, 0)]))
                _aug_label_concat = tf.stack(_aug_label_list, axis=0)[:, :, 2:]
                # (xmin, ymin, xmax, ymax) to (ymin, xmin, ymax, xmax)
                _aug_label_concat = tf.gather(_aug_label_concat, [1, 0, 3, 2], axis=2)
                aug_images = tf.image.draw_bounding_boxes(
                    aug_images,
                    _aug_label_concat
                )
                aug_images = tf.cast(aug_images, tf.uint8)
                visualizer.image(
                    "augmented_images",
                    aug_images,
                    data_format="channels_last"
                )

        self._images = self.images
        self.gt_labels = augmented_labels
        img_mean = experiment_spec.augmentation_config.image_mean
        bb, gg, rr = 103.939, 116.779, 123.68
        if img_mean:
            if num_channels == 3:
                bb, gg, rr = img_mean['b'], img_mean['g'], img_mean['r']
            else:
                bb, gg, rr = img_mean['l'], img_mean['l'], img_mean['l']
        perm = tf.constant([2, 1, 0])
        self.images = tf.gather(self.images, perm, axis=1)
        self.images -= tf.constant([[[[bb]], [[gg]], [[rr]]]])
        if num_channels == 1:
            self.images = tf.reduce_sum(self.images * bgr_, axis=1, keepdims=True)
        self.encoded_labels = self.gt_labels
        if self.label_encoder is not None:
            self.encoded_labels = self.label_encoder(self.gt_labels)
        self.sess = sess

    def set_encoder(self, label_encoder):
        """Set a new label encoder for output labels."""

        self.encoded_labels = label_encoder(self.gt_labels)

    def generator(self):
        """Yields img and label numpy arrays."""

        if self.sess is None:
            raise ValueError('TF session can not be found. Pass a session to the initializer!')
        while True:
            img, enc_label, label = self.sess.run(
                [self.images, self.encoded_labels, self.gt_labels]
            )
            yield img, enc_label, label

    def get_array(self):
        '''get the array for a batch.'''
        return self.sess.run([self.images, self.gt_labels])

    def get_array_and_frame_ids(self):
        '''get the array and frame IDs for a batch.'''
        return self.sess.run([self.frame_ids, self.images, self.gt_labels])
