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

'''build model for training or inference.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import keras.backend as K

from nvidia_tao_tf1.cv.ssd.architecture.ssd_arch import ssd
from nvidia_tao_tf1.cv.ssd.utils.model_io import load_model
from nvidia_tao_tf1.cv.ssd.utils.spec_loader import eval_str


def build(experiment_spec,
          is_dssd,
          input_tensor=None,
          kernel_regularizer=None):
    '''
    Build a model for training with or without training tensors.

    For inference, this function can be used to build a base model, which can be passed into
    eval_builder to attach a decode layer.
    '''

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

    # original_learning_phase = K.learning_phase()

    # set learning to be 1 for generating train graph
    model_train = ssd(image_size=(img_channels, img_height, img_width),
                      n_classes=n_classes,
                      is_dssd=is_dssd,
                      nlayers=experiment_spec.ssd_config.nlayers,
                      pred_num_channels=experiment_spec.ssd_config.pred_num_channels,
                      kernel_regularizer=kernel_regularizer,
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
                      input_tensor=input_tensor,
                      qat=experiment_spec.training_config.enable_qat)

    if experiment_spec.training_config.enable_qat:
        # Save a temp model to avoid multiple calls to create_quantized_keras_model
        _, temp_model_path = tempfile.mkstemp(suffix='.hdf5')
        model_train.save(temp_model_path)
        # Set learning to be 0 for generating eval graph
        K.set_learning_phase(0)
        model_eval = load_model(temp_model_path, experiment_spec,
                                is_dssd)
        os.remove(temp_model_path)
    else:
        K.set_learning_phase(0)
        model_eval = ssd(image_size=(img_channels, img_height, img_width),
                         n_classes=n_classes,
                         is_dssd=is_dssd,
                         nlayers=experiment_spec.ssd_config.nlayers,
                         pred_num_channels=experiment_spec.ssd_config.pred_num_channels,
                         kernel_regularizer=kernel_regularizer,
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
                         qat=experiment_spec.training_config.enable_qat)

    # TODO(tylerz): Hard code the learning phase to 1 since the build only be called in train
    K.set_learning_phase(1)

    return model_train, model_eval
