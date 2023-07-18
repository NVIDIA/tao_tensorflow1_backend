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

"""IVA SSD data loader builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader.types.bbox_2d_label import Bbox2DLabel
import nvidia_tao_tf1.core
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.build_dataloader import build_dataloader
from nvidia_tao_tf1.cv.ssd.utils.tensor_utils import get_non_empty_rows_2d_sparse


class ssd_data:
    """
    Data loader class.

    ssd_data can be used in two ways:
    1. build groundtruth image and label TF tensors. Those two tensors can be
        directly used for training.
    2. build a generator that yields image and label numpy arrays. In this case,
        a TF session needs to be passed into the class initializer.
    """

    def __init__(self,
                 experiment_spec,
                 label_encoder=None,
                 training=True,
                 sess=None):
        """
        Data loader init function.

        Arguments:
            experiment_spec: The loaded config pb2.
            label_encoder (function, optional): If passed in, groundtruth label will be encoded.
            training (bool): Return training set or validation set.
            sess (TF Session): Required if generator() function needs to be called. Otherwise, just
                pass None.
        """

        dataset_proto = experiment_spec.dataset_config
        dataloader = build_dataloader(
            dataset_proto=dataset_proto,
            augmentation_proto=experiment_spec.augmentation_config)
        if training:
            batch_size = experiment_spec.training_config.batch_size_per_gpu
        else:
            batch_size = experiment_spec.eval_config.batch_size
        self.images, self.ground_truth_labels, self.num_samples = \
            dataloader.get_dataset_tensors(batch_size, training=training,
                                           enable_augmentation=training)
        if self.num_samples == 0:
            return
        cls_mapping_dict = experiment_spec.dataset_config.target_class_mapping
        self.classes = sorted({str(x) for x in cls_mapping_dict.values()})
        cls_map = nvidia_tao_tf1.core.processors.LookupTable(keys=self.classes,
                                                 values=list(range(len(self.classes))),
                                                 default_value=-1)
        cls_map.build()
        self.H, self.W = self.images.get_shape().as_list()[2:]
        self.label_encoder = label_encoder

        # preprocess input.
        self.images *= 255.0
        num_channels = experiment_spec.augmentation_config.preprocessing.output_image_channel
        if num_channels == 3:
            perm = tf.constant([2, 1, 0])
            self.images = tf.gather(self.images, perm, axis=1)
            self.images -= tf.constant([[[[103.939]], [[116.779]], [[123.68]]]])
        elif num_channels == 1:
            self.images -= 117.3786
        else:
            raise NotImplementedError(
                "Invalid number of channels {} requested.".format(num_channels)
            )

        gt_labels = []
        if isinstance(self.ground_truth_labels, list):
            for l in self.ground_truth_labels:
                obj_id = cls_map(l['target/object_class'])
                x1 = tf.clip_by_value(tf.cast(tf.round(l['target/coordinates_x1']), tf.int32), 0,
                                      self.W - 1)
                x2 = tf.clip_by_value(tf.cast(tf.round(l['target/coordinates_x2']), tf.int32), 0,
                                      self.W - 1)
                y1 = tf.clip_by_value(tf.cast(tf.round(l['target/coordinates_y1']), tf.int32), 0,
                                      self.H - 1)
                y2 = tf.clip_by_value(tf.cast(tf.round(l['target/coordinates_y2']), tf.int32), 0,
                                      self.H - 1)
                # only select valid labels
                select = tf.logical_and(tf.not_equal(obj_id, -1),
                                        tf.logical_and(tf.less(x1, x2), tf.less(y1, y2)))
                label = tf.stack([obj_id, x1, y1, x2, y2], axis=1)
                gt_labels.append(tf.boolean_mask(label, select))

        elif isinstance(self.ground_truth_labels, Bbox2DLabel):
            source_classes = self.ground_truth_labels.object_class
            mapped_classes = tf.SparseTensor(
                values=cls_map(source_classes.values),
                indices=source_classes.indices,
                dense_shape=source_classes.dense_shape)
            mapped_labels = self.ground_truth_labels._replace(object_class=mapped_classes)
            valid_indices = tf.not_equal(mapped_classes.values, -1)
            filtered_labels = mapped_labels.filter(valid_indices)
            filtered_obj_ids = tf.sparse.reshape(filtered_labels.object_class, [batch_size, -1, 1])
            filtered_coords = tf.sparse.reshape(filtered_labels.vertices.coordinates,
                                                [batch_size, -1, 4])
            filtered_coords = tf.sparse.SparseTensor(
                values=tf.cast(tf.round(filtered_coords.values), tf.int32),
                indices=filtered_coords.indices,
                dense_shape=filtered_coords.dense_shape)
            labels_all = tf.sparse.concat(axis=-1, sp_inputs=[filtered_obj_ids, filtered_coords])
            labels_split = tf.sparse.split(sp_input=labels_all, num_split=batch_size, axis=0)
            labels_split = [tf.sparse.reshape(x, [-1, 5]) for x in labels_split]
            labels = [tf.sparse.to_dense(get_non_empty_rows_2d_sparse(x)) for x in labels_split]

            for l in labels:
                obj_id = l[:, 0]
                x1 = tf.clip_by_value(l[:, 1], 0, self.W - 1)
                x2 = tf.clip_by_value(l[:, 3], 0, self.W - 1)
                y1 = tf.clip_by_value(l[:, 2], 0, self.H - 1)
                y2 = tf.clip_by_value(l[:, 4], 0, self.H - 1)

                # only select valid labels
                select = tf.logical_and(tf.not_equal(obj_id, -1),
                                        tf.logical_and(tf.less(x1, x2), tf.less(y1, y2)))
                label = tf.stack([obj_id, x1, y1, x2, y2], axis=1)
                gt_labels.append(tf.boolean_mask(label, select))

        else:
            raise TypeError('Input must be either list or Bbox2DLabel instance')
        self.gt_labels = gt_labels
        self.ground_truth_labels = gt_labels

        if self.label_encoder is not None:
            self.ground_truth_labels = self.label_encoder(gt_labels)
        self.sess = sess

    def set_encoder(self, label_encoder):
        """Set a new label encoder for output labels."""

        self.ground_truth_labels = label_encoder(self.gt_labels)

    def generator(self):
        """Yields img and label numpy arrays."""

        if self.sess is None:
            raise ValueError('TF session can not be found. Pass a session to the initializer!')
        while True:
            img, label = self.sess.run([self.images, self.ground_truth_labels])
            yield img, label
