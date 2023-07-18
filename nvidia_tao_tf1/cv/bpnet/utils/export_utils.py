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
"""BpNet Export utils."""

import keras
from keras import backend as K
import numpy as np


def get_upsample_kernel(shape, dtype=np.float32):
    """Get a nearest neighbour upsampling kernel.

    Args:
        shape (tuple): shape of the upsampling kernel.
        dtype (np.dtype): kernel data type

    Returns:
        kernel: generated upsampling kernel.
    """
    kernel = np.zeros((shape), dtype=dtype)
    for i in range(shape[-1]):
        kernel[:, :, i, i] = np.ones((shape[0], shape[1]), dtype=dtype)

    return kernel


class CustomUpsampleKernelInitializer:
    """Class for upsample kernel initialization."""

    def __call__(self, shape, dtype=None):
        """Function to invoke kernel initializer."""
        return get_upsample_kernel(shape, dtype=None)


def update_model(model, sdk_compatible_model=False, upsample_ratio=4, use_conv_transpose=True):
    """Update the model with additonal/custom layers.

    Args:
        model (KerasModel): trained model
        upsample_ratio (int): specifies the upsampling ratio for the upsample layer

    Returns:
        model (KerasModel): update model
        custom_objects: Keras custom objects that are added to model
    """

    # Check if the model is a pruned model.
    # If given model has only 2 layers and one of the layer is an instance
    # of keras.engine.training.Model, it is a pruned model. If so, extract
    # the internal model for final export.
    num_layers = len(model.layers)
    if num_layers == 2:
        for layer in model.layers:
            if isinstance(layer, keras.engine.training.Model):
                model = layer

    num_stages = int(len(model.outputs) / 2)
    heatmap_out = model.outputs[num_stages - 1]
    paf_out = model.outputs[num_stages * 2 - 1]
    custom_objects = None

    if sdk_compatible_model:
        if K.image_data_format() == 'channels_first':
            num_paf_channels = paf_out.shape[1]
        else:
            num_paf_channels = int(paf_out.shape[-1])

        # Add upsampling layer for paf
        if use_conv_transpose:
            paf_out = keras.layers.Conv2DTranspose(
                        num_paf_channels,
                        (upsample_ratio, upsample_ratio),
                        strides=(upsample_ratio, upsample_ratio),
                        kernel_initializer=get_upsample_kernel,
                        padding='same'
                    )(paf_out)
            custom_objects = {
                'get_upsample_kernel': CustomUpsampleKernelInitializer
            }
        else:
            paf_out = keras.layers.UpSampling2D(
                        size=(upsample_ratio, upsample_ratio),
                        data_format=None,
                        interpolation='nearest',
                        name="paf_out"
                    )(paf_out)

    updated_model = keras.models.Model(inputs=model.inputs, outputs=[paf_out, heatmap_out])

    return updated_model, custom_objects
