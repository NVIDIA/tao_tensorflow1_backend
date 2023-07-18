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
"""EfficientNet (tf.keras) builder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import tensorflow.compat.v1 as tf

from nvidia_tao_tf1.core.templates.efficientnet_tf import EfficientNetB0, EfficientNetB1
from nvidia_tao_tf1.core.templates.efficientnet_tf import EfficientNetB2, EfficientNetB3
from nvidia_tao_tf1.core.templates.efficientnet_tf import EfficientNetB4, EfficientNetB5

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')
logger = logging.getLogger(__name__)

mappings = {
    'efficientdet-d0': [
        'block1a_project_bn', 'block2b_add', 'block3b_add', 'block5c_add', 'block7a_project_bn'],
    'efficientdet-d1': [
        'block1b_project_bn', 'block2c_add', 'block3c_add', 'block5d_add', 'block7b_project_bn'],
    'efficientdet-d2': [
        'block1b_project_bn', 'block2c_add', 'block3c_add', 'block5d_add', 'block7b_project_bn'],
    'efficientdet-d3': [
        'block1b_project_bn', 'block2c_add', 'block3c_add', 'block5e_add', 'block7b_project_bn'],
    'efficientdet-d4': [
        'block1b_project_bn', 'block2d_add', 'block3d_add', 'block5f_add', 'block7b_project_bn'],
    'efficientdet-d5': [
        'block1b_project_bn', 'block2e_add', 'block3e_add', 'block5g_add', 'block7c_project_bn'],
}


def swish(features, use_native=True, use_hard=False):
    """Computes the Swish activation function.

    We provide three alternatives:
        - Native tf.nn.swish, use less memory during training than composable swish.
        - Quantization friendly hard swish.
        - A composable swish, equivalent to tf.nn.swish, but more general for
            finetuning and TF-Hub.

    Args:
        features: A `Tensor` representing preactivation values.
        use_native: Whether to use the native swish from tf.nn that uses a custom
            gradient to reduce memory usage, or to use customized swish that uses
            default TensorFlow gradient computation.
        use_hard: Whether to use quantization-friendly hard swish.

    Returns:
        The activation value.
    """
    if use_native and use_hard:
        raise ValueError('Cannot specify both use_native and use_hard.')

    if use_native:
        return tf.nn.swish(features)

    if use_hard:
        return features * tf.nn.relu6(features + np.float32(3)) * (1. / 6.)

    features = tf.convert_to_tensor(features, name='features')
    return features * tf.nn.sigmoid(features)


def build_model_base(images, model_name='efficientdet-d0',
                     num_classes=2, freeze_blocks=None, freeze_bn=False):
    """Create a base feature network and return the features before pooling.

    Args:
        images: input images tensor.
        model_name: string, the predefined model name.

    Returns:
        features: base features before pooling.
        endpoints: the endpoints for each layer.

    Raises:
        When model_name specified an undefined model, raises NotImplementedError.
    """
    assert isinstance(images, tf.Tensor)
    supported_backbones = {
        'efficientdet-d0': EfficientNetB0,
        'efficientdet-d1': EfficientNetB1,
        'efficientdet-d2': EfficientNetB2,
        'efficientdet-d3': EfficientNetB3,
        'efficientdet-d4': EfficientNetB4,
        'efficientdet-d5': EfficientNetB5,
    }
    if model_name not in supported_backbones.keys():
        raise ValueError("{} is not a supported arch. \
            Please choose from `efficientdet-d0` to `efficientdet-d5`.")
    model = supported_backbones[model_name](
        add_head=False,
        input_tensor=images,
        classes=num_classes,
        data_format="channels_last",
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        use_bias=False,
        kernel_regularizer=None,
        bias_regularizer=None,
        stride16=False,
        activation_type=None)
    return [model.get_layer(fmap).output for fmap in mappings[model_name]]
