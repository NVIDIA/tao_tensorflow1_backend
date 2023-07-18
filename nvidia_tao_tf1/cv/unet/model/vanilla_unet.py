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

"""Vanilla Unet model class that takes care of constructing and validating a model."""

import logging
import keras
from keras.layers import Input
from keras.models import Model
import tensorflow as tf
from nvidia_tao_tf1.cv.unet.model.layers import bottleneck, downsample_block, input_block, \
    output_block, upsample_block
from nvidia_tao_tf1.cv.unet.model.unet_model import UnetModel


logger = logging.getLogger(__name__)


class VanillaUnet(UnetModel):
    """VanillaUnet class."""

    def __init__(self, *args, **kwargs):
        """Init function.

        Args:
            num_layers (int): Number of layers for scalable feature extractors.
            use_pooling (bool): Whether to add pooling layers to the feature extractor.
            use_batch_norm (bool): Whether to add batch norm layers.
            dropout_rate (float): Fraction of the input units to drop. 0.0 means dropout is
                not used.
            target_class_names (list): A list of target class names.
            freeze_pretrained_layers (bool): Prevent updates to pretrained layers' parameters.
            allow_loaded_model_modification (bool): Allow loaded model modification.
            template (str): Model template to use for feature extractor.
            freeze_bn (bool): The boolean to freeze BN or not.
            load_graph (bool): The boolean to laod graph for phase 1.
        """
        super(VanillaUnet, self).__init__(*args, **kwargs)

    def construct_encoder_decoder_model(self, input_shape, mode, features, export):
        """Generate a encoder/ decoder model.

        Args:
            input_shape: model input shape (N,C,H,W). N is ignored.
            data_format: Order of the dimensions (C, H, W).
            kernel_regularizer: Keras regularizer to be applied to convolution
                                kernels.
            bias_regularizer: Keras regularizer to be applied to biases.
            features: Fed by the estimator during training.
        Raises:
            AssertionError: If the model is already constructed.
        Returns:
            model (keras.model): The model for feature extraction.
        """
        assert not self.constructed, "Model already constructed."
        skip_connections = []
        assert (input_shape[1] == input_shape[2] == 572), \
            "model_input_width and model_input_height should be set 572 \
            for Vanilla Unet in the spec."
        if features:
            inputs = Input(tensor=features, shape=input_shape)
        else:
            inputs = Input(shape=input_shape)
        out, skip = input_block(inputs, filters=64)

        skip_connections.append(skip)

        for filters in [128, 256, 512]:
            out, skip = downsample_block(out, filters=filters)
            skip_connections.append(skip)
        out = keras.layers.Conv2D(
                               filters=1024,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu)(out)
        out = keras.layers.Conv2D(
                               filters=1024,
                               kernel_size=(3, 3),
                               activation=tf.nn.relu)(out)

        out = bottleneck(out, filters_up=512, mode=mode)

        # Constructing the decoder
        for filters in [(512, 4, 256), (256, 16, 128), (128, 40, 64)]:
            out = upsample_block(out,
                                 residual_input=skip_connections.pop(),
                                 filters=filters)

        out = output_block(out, residual_input=skip_connections.pop(),
                           filters=64, n_classes=self.num_target_classes)
        if export:
            # When export the output is logits, applying softmax on it for final
            # output
            logger.debug("Building model for export")
            out = self.get_activation_for_export(out)

        model_unet_v1 = Model(inputs=inputs, outputs=out)
        return model_unet_v1
