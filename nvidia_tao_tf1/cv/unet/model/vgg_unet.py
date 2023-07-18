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

"""VGG Unet model class that takes care of constructing and validating a model."""


import logging
import keras
from keras import backend as K
from keras.models import Model

from nvidia_tao_tf1.core.templates.vgg import VggNet
from nvidia_tao_tf1.cv.unet.model.unet_model import UnetModel

logger = logging.getLogger(__name__)

vgg_dict = {'vgg': ('block_1a_relu', 'block1_pool',
                    'block_2a_relu', 'block2_pool',
                    'block_3a_relu', 'block_3b_relu', 'block3_pool',
                    'block_4a_relu', 'block_4b_relu', 'block4_pool',
                    'block_5a_relu', 'block_5b_relu', 'block_5c_relu')}


class VggUnet(UnetModel):
    """VggUnet Unet class."""

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
        super(VggUnet, self).__init__(*args, **kwargs)

    def construct_decoder_model(self, encoder_model, export=False):
        """Construct the decoder for Unet with VGG as backbone.

        Args:
            encoder_model (keras.model): keras model type.
            export (bool): Set the inference flag to build the
                            inference model with softmax.
        Returns:
            model (keras.model): The entire Unet model with encoder and decoder.
        """
        vgg_layers = vgg_dict[self.template]
        conv1_1 = encoder_model.get_layer(vgg_layers[0]).output
        conv1_2 = encoder_model.get_layer(vgg_layers[1]).output
        conv2_2 = encoder_model.get_layer(vgg_layers[3]).output
        conv3_3 = encoder_model.get_layer(vgg_layers[6]).output
        conv4_3 = encoder_model.get_layer(vgg_layers[9]).output
        conv5_3 = encoder_model.get_layer(vgg_layers[12]).output
        conv5_3 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                                            data_format="channels_first")(conv5_3)
        # First Upsampling
        upscale1 = keras.layers.Conv2DTranspose(
            filters=K.int_shape(conv4_3)[1], kernel_size=(4, 4), strides=(2, 2),
            padding='same', data_format="channels_first")(conv5_3)
        concat_axis = 3 if K.image_data_format() == 'channels_last' else 1
        concat1 = keras.layers.Concatenate(axis=concat_axis)([upscale1, conv4_3])
        expand11 = keras.layers.Conv2D(
                                       filters=K.int_shape(conv4_3)[1],
                                       kernel_size=(3, 3), padding='same')(concat1)
        expand12 = keras.layers.Conv2D(
            filters=K.int_shape(conv4_3)[1], kernel_size=(3, 3),
            padding='same')(expand11)
        # Second Block Upsampling
        upscale2 = keras.layers.Conv2DTranspose(filters=K.int_shape(conv3_3)[1],
                                                kernel_size=(4, 4),
                                                strides=(2, 2),
                                                padding='same',
                                                data_format="channels_first")(expand12)
        concat2 = keras.layers.Concatenate(axis=concat_axis)([upscale2, conv3_3])
        expand21 = keras.layers.Conv2D(
                                       filters=K.int_shape(conv3_3)[1],
                                       kernel_size=(3, 3), padding='same')(concat2)
        expand22 = keras.layers.Conv2D(
            filters=K.int_shape(conv3_3)[1], kernel_size=(3, 3), padding='same')(expand21)

        # Third Block Upsampling
        upscale3 = keras.layers.Conv2DTranspose(
            filters=K.int_shape(conv2_2)[1], kernel_size=(4, 4), strides=(2, 2),
            padding='same', data_format="channels_first")(expand22)
        concat3 = keras.layers.Concatenate(axis=concat_axis)([upscale3, conv2_2])
        expand31 = keras.layers.Conv2D(
                                       filters=K.int_shape(conv2_2)[1],
                                       kernel_size=(3, 3), padding='same')(concat3)
        expand32 = keras.layers.Conv2D(
            filters=K.int_shape(conv2_2)[1], kernel_size=(3, 3), padding='same')(expand31)

        # Fourth Block Upsampling
        upscale4 = keras.layers.Conv2DTranspose(
            filters=K.int_shape(conv1_2)[1], kernel_size=(4, 4), strides=(2, 2),
            padding='same', data_format="channels_first")(expand32)
        concat4 = keras.layers.Concatenate(axis=concat_axis)([upscale4, conv1_2])
        expand41 = keras.layers.Conv2D(
                                       filters=K.int_shape(conv1_2)[1],
                                       kernel_size=(3, 3), padding='same')(concat4)
        expand42 = keras.layers.Conv2D(
            filters=K.int_shape(conv1_2)[1], kernel_size=(3, 3), padding='same')(expand41)
        # Fifth Block Upsampling
        upscale5 = keras.layers.Conv2DTranspose(
            filters=K.int_shape(conv1_1)[1], kernel_size=(4, 4), strides=(2, 2),
            padding='same', data_format="channels_first")(expand42)
        concat5 = keras.layers.Concatenate(axis=concat_axis)([upscale5, conv1_1])
        expand51 = keras.layers.Conv2D(
           filters=K.int_shape(conv1_1)[1], kernel_size=(3, 3), padding='same')(concat5)
        expand52 = keras.layers.Conv2D(
            filters=K.int_shape(conv1_1)[1], kernel_size=(3, 3),  padding='same')(expand51)

        # Output block
        out = keras.layers.Conv2D(filters=self.num_target_classes,
                                  kernel_size=(1, 1), padding='same')(expand52)
        if export:
            logger.debug("Building model for export")
            out = self.get_activation_for_export(out)

        model_unet = Model(inputs=encoder_model.input, outputs=out)
        return model_unet

    def get_base_model(self, args, kwargs):
        """Function to construct model specific backbone."""

        model_class = VggNet
        model = model_class(*args, **kwargs)
        return model
