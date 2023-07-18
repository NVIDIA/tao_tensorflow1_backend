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

"""ResNet Unet model class that takes care of constructing and validating a model."""

import logging
from keras.models import Model

from nvidia_tao_tf1.core.templates.resnet import ResNet
from nvidia_tao_tf1.cv.unet.model.layers import output_block_other, upsample_block_other
from nvidia_tao_tf1.cv.unet.model.unet_model import UnetModel

logger = logging.getLogger(__name__)

resnet_dict_gen = {'resnet': ('activation_1', 'block_1b_relu', 'block_2b_relu',
                              'block_3b_relu', 'block_4b_relu')}

resnet_10_dict = {'resnet': ('activation_1', 'block_1a_relu', 'block_2a_relu',
                             'block_3a_relu', 'block_4a_relu')}


class ResnetUnet(UnetModel):
    """ResnetUnet class."""

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
        super(ResnetUnet, self).__init__(*args, **kwargs)

    def construct_decoder_model(self, encoder_model, export=False):
        """Construct the decoder for Unet with Resnet as backbone.

        Args:
            encoder_model (keras.model): keras model type.
            export (bool): Set the inference flag to build the
                            inference model with softmax.
        Returns:
            model (keras.model): The entire Unet model with encoder and decoder.
        """
        if self.num_layers == 10:
            resnet_dict = resnet_10_dict
        else:
            resnet_dict = resnet_dict_gen

        B1, B2, B3, _, _ = resnet_dict[self.template]
        S1 = encoder_model.get_layer(B1).output
        S2 = encoder_model.get_layer(B2).output
        S3 = encoder_model.get_layer(B3).output

        skips = [S3, S2, S1]
        x = encoder_model.output
        decoder_filters = (256, 128, 64, 64)
        n_upsample_blocks = 4
        for i in range(n_upsample_blocks):
            if i < len(skips):
                skip = skips[i]
            else:
                skip = None
            x = upsample_block_other(x, skip, decoder_filters[i],
                                     use_batchnorm=self.use_batch_norm,
                                     freeze_bn=self.freeze_bn,
                                     initializer=self.initializer)

        out = output_block_other(x, n_classes=self.num_target_classes)

        if export:
            logger.debug("Building model for export")
            out = self.get_activation_for_export(out)

        model_unet = Model(inputs=encoder_model.input, outputs=out)

        return model_unet

    def get_base_model(self, args, kwargs):
        """Function to construct model specific backbone."""

        kwargs['use_pooling'] = False
        model_class = ResNet
        kwargs['all_projections'] = self.all_projections

        model = model_class(*args, **kwargs)
        return model
