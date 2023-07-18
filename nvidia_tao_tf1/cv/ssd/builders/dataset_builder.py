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

'''build train dataset and val dataset.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from nvidia_tao_tf1.cv.ssd.architecture.ssd_arch import ssd
from nvidia_tao_tf1.cv.ssd.box_coder.input_encoder import SSDInputEncoder
from nvidia_tao_tf1.cv.ssd.box_coder.ssd_input_encoder import SSDInputEncoderNP
from nvidia_tao_tf1.cv.ssd.builders.dalipipeline_builder import SSDDALIDataset
from nvidia_tao_tf1.cv.ssd.builders.data_sequence import SSDDataSequence
from nvidia_tao_tf1.cv.ssd.utils.spec_loader import eval_str


def build_dataset(experiment_spec,
                  is_dssd,
                  pos_iou_threshold=0.5,
                  neg_iou_limit=0.5,
                  device_id=0,
                  shard_id=0,
                  num_shards=1):
    """Build the DALI based dataset or Keras sequence based dataset."""

    img_channels = experiment_spec.augmentation_config.output_channel
    img_height = experiment_spec.augmentation_config.output_height
    img_width = experiment_spec.augmentation_config.output_width
    cls_mapping = experiment_spec.dataset_config.target_class_mapping
    classes = sorted({str(x) for x in cls_mapping.values()})
    # n_classes + 1 for background class
    n_classes = len(classes) + 1
    scales = eval_str(experiment_spec.ssd_config.scales)
    aspect_ratios_global = eval_str(
        experiment_spec.ssd_config.aspect_ratios_global)
    aspect_ratios_per_layer = eval_str(
        experiment_spec.ssd_config.aspect_ratios)
    steps = eval_str(experiment_spec.ssd_config.steps)
    offsets = eval_str(experiment_spec.ssd_config.offsets)
    variances = eval_str(experiment_spec.ssd_config.variances)
    freeze_blocks = eval_str(experiment_spec.ssd_config.freeze_blocks)
    freeze_bn = eval_str(experiment_spec.ssd_config.freeze_bn)

    # Use a fake model to get predictor's size
    model_fake = ssd(image_size=(img_channels, img_height, img_width),
                     n_classes=n_classes,
                     is_dssd=is_dssd,
                     nlayers=experiment_spec.ssd_config.nlayers,
                     pred_num_channels=experiment_spec.ssd_config.pred_num_channels,
                     kernel_regularizer=None,
                     freeze_blocks=freeze_blocks,
                     freeze_bn=freeze_bn,
                     scales=scales,
                     min_scale=experiment_spec.ssd_config.min_scale,
                     max_scale=experiment_spec.ssd_config.max_scale,
                     aspect_ratios_global=aspect_ratios_global,
                     aspect_ratios_per_layer=aspect_ratios_per_layer,
                     two_boxes_for_ar1=experiment_spec.ssd_config.two_boxes_for_ar1,
                     steps=steps,
                     offsets=offsets,
                     clip_boxes=experiment_spec.ssd_config.clip_boxes,
                     variances=variances,
                     arch=experiment_spec.ssd_config.arch,
                     input_tensor=None,
                     qat=False)

    predictor_sizes = [model_fake.get_layer('ssd_conf_0').output_shape[2:],
                       model_fake.get_layer('ssd_conf_1').output_shape[2:],
                       model_fake.get_layer('ssd_conf_2').output_shape[2:],
                       model_fake.get_layer('ssd_conf_3').output_shape[2:],
                       model_fake.get_layer('ssd_conf_4').output_shape[2:],
                       model_fake.get_layer('ssd_conf_5').output_shape[2:]]

    use_dali = False
    # @TODO(tylerz): if there is tfrecord, then use dali.
    if experiment_spec.dataset_config.data_sources[0].tfrecords_path != "":
        use_dali = True
        print("Using DALI augmentation pipeline.")

    if use_dali:
        train_dataset = SSDDALIDataset(experiment_spec=experiment_spec,
                                       device_id=device_id,
                                       shard_id=shard_id,
                                       num_shards=num_shards)
    else:
        train_dataset = \
            SSDDataSequence(dataset_config=experiment_spec.dataset_config,
                            augmentation_config=experiment_spec.augmentation_config,
                            batch_size=experiment_spec.training_config.batch_size_per_gpu,
                            is_training=True)

    if experiment_spec.ssd_config.pos_iou_thresh != 0.0:
        pos_iou_threshold = experiment_spec.ssd_config.pos_iou_thresh
    else:
        pos_iou_threshold = 0.5

    if experiment_spec.ssd_config.neg_iou_thresh != 0.0:
        neg_iou_limit = experiment_spec.ssd_config.neg_iou_thresh
    else:
        neg_iou_limit = 0.5

    ssd_input_encoder = \
        SSDInputEncoderNP(img_height=img_height,
                          img_width=img_width,
                          n_classes=n_classes,
                          predictor_sizes=predictor_sizes,
                          scales=scales,
                          min_scale=experiment_spec.ssd_config.min_scale,
                          max_scale=experiment_spec.ssd_config.max_scale,
                          aspect_ratios_global=aspect_ratios_global,
                          aspect_ratios_per_layer=aspect_ratios_per_layer,
                          two_boxes_for_ar1=experiment_spec.ssd_config.two_boxes_for_ar1,
                          steps=steps,
                          offsets=offsets,
                          clip_boxes=experiment_spec.ssd_config.clip_boxes,
                          variances=variances,
                          pos_iou_threshold=pos_iou_threshold,
                          neg_iou_limit=neg_iou_limit)

    if use_dali:
        ssd_input_encoder_tf = \
            SSDInputEncoder(img_height=img_height,
                            img_width=img_width,
                            n_classes=n_classes,
                            predictor_sizes=predictor_sizes,
                            scales=scales,
                            min_scale=experiment_spec.ssd_config.min_scale,
                            max_scale=experiment_spec.ssd_config.max_scale,
                            aspect_ratios_global=aspect_ratios_global,
                            aspect_ratios_per_layer=aspect_ratios_per_layer,
                            two_boxes_for_ar1=experiment_spec.ssd_config.two_boxes_for_ar1,
                            steps=steps,
                            offsets=offsets,
                            clip_boxes=experiment_spec.ssd_config.clip_boxes,
                            variances=variances,
                            pos_iou_threshold=0.5,
                            neg_iou_limit=0.5,
                            gt_normalized=True)
        train_encoder = ssd_input_encoder_tf
    else:
        train_encoder = ssd_input_encoder

    train_dataset.set_encoder(train_encoder)

    def eval_encode_fn(gt_label):
        bboxes = gt_label[:, -4:]
        cls_id = gt_label[:, 0:1]
        gt_label_without_diff = np.concatenate((cls_id, bboxes), axis=-1)
        return (ssd_input_encoder(gt_label_without_diff), gt_label)

    val_dataset = SSDDataSequence(dataset_config=experiment_spec.dataset_config,
                                  augmentation_config=experiment_spec.augmentation_config,
                                  batch_size=experiment_spec.eval_config.batch_size,
                                  is_training=False,
                                  encode_fn=eval_encode_fn)

    return train_dataset, val_dataset
