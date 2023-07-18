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

import keras.backend as K
from nvidia_tao_tf1.cv.retinanet.architecture.retinanet import retinanet
from nvidia_tao_tf1.cv.retinanet.utils.helper import eval_str


def build(experiment_spec,
          n_classes,
          kernel_regularizer=None,
          input_tensor=None):
    '''
    Build a model for training with or without training tensors.

    For inference, this function can be used to build a base model, which can be passed into
    eval_builder to attach a decode layer.
    '''

    img_channels = experiment_spec.augmentation_config.output_channel
    img_height = experiment_spec.augmentation_config.output_height
    img_width = experiment_spec.augmentation_config.output_width
    scales = eval_str(experiment_spec.retinanet_config.scales)
    aspect_ratios_global = eval_str(experiment_spec.retinanet_config.aspect_ratios_global)
    aspect_ratios_per_layer = eval_str(experiment_spec.retinanet_config.aspect_ratios)
    steps = eval_str(experiment_spec.retinanet_config.steps)
    offsets = eval_str(experiment_spec.retinanet_config.offsets)
    variances = eval_str(experiment_spec.retinanet_config.variances)
    freeze_blocks = eval_str(experiment_spec.retinanet_config.freeze_blocks)
    freeze_bn = eval_str(experiment_spec.retinanet_config.freeze_bn)
    nlayers = experiment_spec.retinanet_config.nlayers
    arch = experiment_spec.retinanet_config.arch
    n_anchor_levels = experiment_spec.retinanet_config.n_anchor_levels or 3
    # Config FPN
    n_kernels = experiment_spec.retinanet_config.n_kernels
    feature_size = experiment_spec.retinanet_config.feature_size
    # Enable QAT
    use_qat = experiment_spec.training_config.enable_qat

    # set learning to be 1 for generating train graph
    original_learning_phase = K.learning_phase()
    K.set_learning_phase(1)
    model_train = retinanet(image_size=(img_channels, img_height, img_width),
                            n_classes=n_classes,
                            kernel_regularizer=kernel_regularizer,
                            freeze_blocks=freeze_blocks,
                            freeze_bn=freeze_bn,
                            scales=scales,
                            min_scale=experiment_spec.retinanet_config.min_scale,
                            max_scale=experiment_spec.retinanet_config.max_scale,
                            aspect_ratios_global=aspect_ratios_global,
                            aspect_ratios_per_layer=aspect_ratios_per_layer,
                            two_boxes_for_ar1=experiment_spec.retinanet_config.two_boxes_for_ar1,
                            steps=steps,
                            n_anchor_levels=n_anchor_levels,
                            offsets=offsets,
                            clip_boxes=experiment_spec.retinanet_config.clip_boxes,
                            variances=variances,
                            nlayers=nlayers,
                            arch=arch,
                            n_kernels=n_kernels,
                            feature_size=feature_size,
                            input_tensor=input_tensor,
                            qat=use_qat)
    # Set learning to be 0 for generating eval graph
    K.set_learning_phase(0)
    model_eval = retinanet(image_size=(img_channels, img_height, img_width),
                           n_classes=n_classes,
                           kernel_regularizer=kernel_regularizer,
                           freeze_blocks=freeze_blocks,
                           freeze_bn=freeze_bn,
                           scales=scales,
                           min_scale=experiment_spec.retinanet_config.min_scale,
                           max_scale=experiment_spec.retinanet_config.max_scale,
                           aspect_ratios_global=aspect_ratios_global,
                           aspect_ratios_per_layer=aspect_ratios_per_layer,
                           two_boxes_for_ar1=experiment_spec.retinanet_config.two_boxes_for_ar1,
                           steps=steps,
                           n_anchor_levels=n_anchor_levels,
                           offsets=offsets,
                           clip_boxes=experiment_spec.retinanet_config.clip_boxes,
                           variances=variances,
                           nlayers=nlayers,
                           arch=arch,
                           n_kernels=n_kernels,
                           feature_size=feature_size,
                           input_tensor=None,
                           qat=use_qat)
    K.set_learning_phase(original_learning_phase)
    return model_train, model_eval
