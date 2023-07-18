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

'''DALI pipeline for SSD.'''

import glob
from math import ceil
import os
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.types as types
import tensorflow as tf


ssd_features = {'frame/id': tfrec.FixedLenFeature((), tfrec.string,  ""),
                'frame/width': tfrec.FixedLenFeature([1], tfrec.int64,  -1),
                'frame/height': tfrec.FixedLenFeature([1], tfrec.int64,  -1),
                'frame/encoded': tfrec.FixedLenFeature((), tfrec.string,  ""),
                'target/object_class_id': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'target/observation_angle': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'target/occlusion': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'target/truncation': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'target/coordinates_x1': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'target/coordinates_y1': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'target/coordinates_x2': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'target/coordinates_y2': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'target/world_bbox_h': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'target/world_bbox_w': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'target/world_bbox_l': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'target/world_bbox_x': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'target/world_bbox_y': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'target/world_bbox_z': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'target/world_bbox_rot_y': tfrec.VarLenFeature(tfrec.float32, 0.0)}


def create_ssd_training_pipeline(experiment_spec, local_rank, shard_id, num_shards):
    """Create DALI pipeline for SSD."""

    batch_size = experiment_spec.training_config.batch_size_per_gpu
    n_workers = experiment_spec.training_config.n_workers or 4
    dataset_config = experiment_spec.dataset_config
    augmentation_config = experiment_spec.augmentation_config

    # Configure the augmentation
    output_width = augmentation_config.output_width
    output_height = augmentation_config.output_height
    output_channel = augmentation_config.output_channel
    min_ar = augmentation_config.random_crop_min_ar or 0.5
    max_ar = augmentation_config.random_crop_max_ar or 2.0
    min_scale = augmentation_config.random_crop_min_scale or 0.3
    max_scale = augmentation_config.random_crop_max_scale or 1.0
    zoom_out_prob = 0.5
    min_zoom_out = int(augmentation_config.zoom_out_min_scale) or 1
    max_zoom_out = int(augmentation_config.zoom_out_max_scale) or 4
    s_delta = augmentation_config.saturation
    c_delta = augmentation_config.contrast
    b_delta = int(augmentation_config.brightness)
    b_delta /= 256
    h_delta = int(augmentation_config.hue)
    random_flip = augmentation_config.random_flip
    img_mean = augmentation_config.image_mean

    # get the tfrecord list from pattern
    data_sources = dataset_config.data_sources
    tfrecord_path_list = []
    for data_source in data_sources:
        tfrecord_pattern = str(data_source.tfrecords_path)
        tf_file_list = sorted(glob.glob(tfrecord_pattern))
        tfrecord_path_list.extend(tf_file_list)

    # create index for tfrecords
    idx_path_list = []
    total_sample_cnt = 0
    for tfrecord_path in tfrecord_path_list:
        root_path, tfrecord_file = os.path.split(tfrecord_path)
        idx_path = os.path.join(root_path, "idx-"+tfrecord_file)
        if not os.path.exists(idx_path):
            raise ValueError("The index file {} for {} does not exist.".format(idx_path,
                                                                               tfrecord_path))
        else:
            with open(idx_path, "r") as f:
                total_sample_cnt += len(f.readlines())

        idx_path_list.append(idx_path)

    # create ssd augmentation pipeline
    pipe = Pipeline(batch_size=batch_size,
                    num_threads=n_workers,
                    device_id=local_rank)
    with pipe:
        inputs = fn.readers.tfrecord(
            path=tfrecord_path_list,
            index_path=idx_path_list,
            features=ssd_features,
            num_shards=num_shards,
            shard_id=shard_id
        )
        images = inputs['frame/encoded']

        # @TODO(tylerz): hsv api requires RGB color space.
        # Need permutation channels before feeding to model.
        images = fn.decoders.image(
            images,
            device="mixed",
            output_type=types.RGB
        )

        x1 = inputs['target/coordinates_x1']
        y1 = inputs['target/coordinates_y1']
        x2 = inputs['target/coordinates_x2']
        y2 = inputs['target/coordinates_y2']
        bboxes = fn.stack(x1, y1, x2, y2, axis=1)
        class_ids = fn.cast(inputs['target/object_class_id'],
                            dtype=types.INT32)
        # random expand:
        zoom_flip_coin = fn.random.coin_flip(probability=zoom_out_prob)
        ratio = 1 + zoom_flip_coin * fn.random.uniform(range=[min_zoom_out-1, max_zoom_out-1])
        paste_x = fn.random.uniform(range=[0, 1])
        paste_y = fn.random.uniform(range=[0, 1])
        images = fn.paste(images, ratio=ratio, paste_x=paste_x, paste_y=paste_y,
                          fill_value=(123.68, 116.779, 103.939))
        bboxes = fn.bbox_paste(bboxes, ratio=ratio, paste_x=paste_x, paste_y=paste_y,
                               ltrb=True)
        # random bbox crop:
        crop_begin, crop_size, bboxes, class_ids = \
            fn.random_bbox_crop(bboxes, class_ids,
                                device="cpu",
                                aspect_ratio=[min_ar, max_ar],
                                thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
                                scaling=[min_scale, max_scale],
                                bbox_layout="xyXY",
                                allow_no_crop=True,
                                num_attempts=50)
        # image slice according to cropped bboxes
        images = fn.slice(images, crop_begin, crop_size,
                          out_of_bounds_policy="trim_to_shape")
        # resize
        images = fn.resize(images,
                           resize_x=output_width,
                           resize_y=output_height,
                           min_filter=types.DALIInterpType.INTERP_LINEAR,
                           antialias=False)
        # color augmentation:
        # - saturation
        # - contrast
        # - brightness
        # - hue
        if s_delta > 0:
            random_saturation = fn.random.uniform(range=[1 - s_delta, 1 + s_delta])
        else:
            random_saturation = 1
        if c_delta > 0:
            random_contrast = fn.random.uniform(range=[1 - c_delta, 1 + c_delta])
        else:
            random_contrast = 1
        if b_delta > 0:
            random_brightness = fn.random.uniform(range=[1 - b_delta, 1 + b_delta])
        else:
            random_brightness = 1
        if h_delta > 0:
            random_hue = fn.random.uniform(range=[0, h_delta])
        else:
            random_hue = 0
        images = fn.hsv(images,
                        dtype=types.FLOAT,
                        hue=random_hue,
                        saturation=random_saturation)
        images = fn.brightness_contrast(images,
                                        contrast_center=128,  # input is in float, but in 0..255
                                        dtype=types.UINT8,
                                        brightness=random_brightness,
                                        contrast=random_contrast)
        # RGB -> BGR
        images = fn.color_space_conversion(images, image_type=types.RGB, output_type=types.BGR)
        # flip the the image and bbox
        horizontal_flip_coin = fn.random.coin_flip(probability=random_flip)
        bboxes = fn.bb_flip(bboxes, ltrb=True, horizontal=horizontal_flip_coin)
        # crop_mirror_normalize
        # mean=[0.485 * 255, 0.456 * 255, 0.406 * 255] is for RGB
        # std=[0.229 * 255, 0.224 * 255, 0.225 * 255], is for RGB
        if output_channel == 3:
            mean = [0.406 * 255, 0.456 * 255, 0.485 * 255]
        elif output_channel == 1:
            mean = [117.3786, 117.3786, 117.3786]

        if img_mean:
            if output_channel == 3:
                mean = [img_mean['b'], img_mean['g'], img_mean['r']]
            else:
                mean = [img_mean['l'], img_mean['l'], img_mean['l']]

        if output_channel == 3:
            images = fn.crop_mirror_normalize(images,
                                              mean=mean,
                                              mirror=horizontal_flip_coin,
                                              dtype=types.FLOAT,
                                              output_layout="CHW",
                                              pad_output=False)
        elif output_channel == 1:
            images = fn.crop_mirror_normalize(images,
                                              mean=mean,
                                              mirror=horizontal_flip_coin,
                                              dtype=types.UINT8,
                                              output_layout="HWC",
                                              pad_output=False)
            images = fn.color_space_conversion(images, image_type=types.BGR, output_type=types.GRAY)
            images = fn.transpose(images, perm=[2, 0, 1], output_layout="CHW")
            images = fn.cast(images, dtype=types.FLOAT)

        # @TODO(tylerz): Encoding bboxes labels using the fn.box_encoder()
        # # encode bbox with anchors
        # # - bboxes shape [#boxes, 4]
        # # - class_ids shape [#boxes, ]
        # bboxes, class_ids = fn.box_encoder(bboxes, class_ids,
        #                                    criteria=0.5,
        #                                    anchors=default_boxes)
        # # generate one-hot class vector
        # class_ids = fn.one_hot(class_ids,
        #                        dtype=types.FLOAT,
        #                        num_classes=num_classes)
        # # generate final label #class + bbox
        # labels = fn.cat(class_ids, bboxes, axis=1)
        class_ids = fn.cast(class_ids,
                            dtype=types.FLOAT)
        class_ids = fn.pad(class_ids, fill_value=-1)
        bboxes = fn.pad(bboxes)
        pipe.set_outputs(images, class_ids, bboxes)

    image_shape = (batch_size, output_channel, output_height, output_width)

    return pipe, total_sample_cnt, image_shape


class SSDDALIDataset():
    """SSD dataset using DALI. Only works for train dataset now."""

    def __init__(self, experiment_spec, device_id, shard_id, num_shards):
        """
        Init the SSD DALI dataset.

        Arguments:
            experiment_spec: experiment spec for training.
            device_id: local rank.
            shard_id: rank.
            num_shards: number of gpus.
        """
        pipe, total_sample_cnt, image_shape = \
            create_ssd_training_pipeline(experiment_spec,
                                         device_id,
                                         shard_id,
                                         num_shards)
        self.n_samples = total_sample_cnt
        self.image_shape = image_shape
        self.batch_size = image_shape[0]
        daliop = dali_tf.DALIIterator()
        self.images, self.class_ids, self.bboxes = daliop(
            pipeline=pipe,
            shapes=[self.image_shape, (), ()],
            dtypes=[tf.float32,
                    tf.float32,
                    tf.float32]
        )
        self._post_process_label()

    def _post_process_label(self):
        """Generate final labels for encoder."""
        labels = []

        for batch_idx in range(self.batch_size):
            cls_in_batch = tf.expand_dims(self.class_ids[batch_idx], axis=-1)
            bboxes_in_batch = self.bboxes[batch_idx]
            mask = tf.greater_equal(cls_in_batch, 0)
            mask.set_shape([None])
            label = tf.concat([cls_in_batch, bboxes_in_batch], axis=-1)
            label = tf.boolean_mask(label, mask)
            label = tf.reshape(label, (-1, 5))
            labels.append(label)

        self.labels = labels

    def set_encoder(self, encoder):
        """Set the label encoder for ground truth."""
        self.labels = encoder(self.labels)

    def __len__(self):
        """Return the total sample number."""
        return int(ceil(self.n_samples / self.batch_size))
