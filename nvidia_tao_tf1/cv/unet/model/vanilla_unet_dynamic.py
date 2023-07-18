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

"""Vanilla Unet Dynamic model class that takes care of constructing decoder model."""

import logging
import keras
from keras.layers import Input
from keras.models import Model
from nvidia_tao_tf1.cv.unet.model.layers import conv_block, decoder_block, encoder_block
from nvidia_tao_tf1.cv.unet.model.unet_model import UnetModel


logger = logging.getLogger(__name__)


class VanillaUnetDynamic(UnetModel):
    """VanillaUnetDynamic class."""

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
        super(VanillaUnetDynamic, self).__init__(*args, **kwargs)

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
        if features:
            inputs = Input(tensor=features, shape=input_shape)
        else:
            inputs = Input(shape=input_shape)

        encoder0_pool, encoder0 = encoder_block(inputs, 32, 0, self.use_batch_norm,
                                                self.freeze_bn, self.initializer)  # 128
        encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64, 1,
                                                self.use_batch_norm, self.freeze_bn,
                                                self.initializer)  # 64
        encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128, 2,
                                                self.use_batch_norm, self.freeze_bn,
                                                self.initializer)  # 32
        encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256, 3,
                                                self.use_batch_norm, self.freeze_bn,
                                                self.initializer)  # 16
        encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512, 4,
                                                self.use_batch_norm, self.freeze_bn,
                                                self.initializer)  # 8

        center = conv_block(encoder4_pool, 1024, self.use_batch_norm,
                            self.freeze_bn)  # center

        decoder4 = decoder_block(center, encoder4, 512, 0, self.use_batch_norm,
                                 self.freeze_bn, self.initializer)  # 16
        decoder3 = decoder_block(decoder4, encoder3, 256, 1, self.use_batch_norm,
                                 self.freeze_bn, self.initializer)  # 32
        decoder2 = decoder_block(decoder3, encoder2, 128, 2, self.use_batch_norm,
                                 self.freeze_bn, self.initializer)  # 64
        decoder1 = decoder_block(decoder2, encoder1, 64, 3, self.use_batch_norm,
                                 self.freeze_bn, self.initializer)  # 128
        decoder0 = decoder_block(decoder1, encoder0, 32, 4, self.use_batch_norm,
                                 self.freeze_bn, self.initializer)  # 256

        out = keras.layers.Conv2D(self.num_target_classes, (1, 1))(decoder0)

        if export:
            logger.debug("Building model for export")
            out = self.get_activation_for_export(out)

        return Model(inputs=[inputs], outputs=[out])
