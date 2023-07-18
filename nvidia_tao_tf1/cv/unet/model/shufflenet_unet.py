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

"""Skip Net decoder Unet model class that takes care of constructing and validating a model."""

import logging
import keras
from keras.models import Model

from nvidia_tao_tf1.core.templates.shufflenet import ShuffleNet
from nvidia_tao_tf1.cv.unet.model.layers import conv2D_bn, convTranspose2D_bn
from nvidia_tao_tf1.cv.unet.model.unet_model import UnetModel

logger = logging.getLogger(__name__)

eff_dict = {'shufflenet': ('stage4/block4/relu_out', 'stage3/block8/relu_out',
                           'stage2/block4/relu_out', 'maxpool1', 'conv1', 'score_fr')}


class ShuffleNetUnet(UnetModel):
    """Efficientnet Unet class."""

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
        super(ShuffleNetUnet, self).__init__(*args, **kwargs)

    def construct_decoder_model(self, encoder_model, export=False):
        """Construct the decoder for Unet with EfficientNet as backbone.

        Args:
            encoder_model (keras.model): keras model type.
            export (bool): Set the inference flag to build the
                            inference model with softmax.
        Returns:
            model (keras.model): The entire Unet model with encoder and decoder.
        """
        _, stage3, stage2, _, _, score_fr = eff_dict[self.template]
        stage3_out = encoder_model.get_layer(stage3).output
        stage2_out = encoder_model.get_layer(stage2).output
        score_fr_out = encoder_model.get_layer(score_fr).output

        upscore = convTranspose2D_bn(x=score_fr_out, filters=self.num_target_classes,
                                     freeze_bn=False, use_batchnorm=True)
        score_feed1 = conv2D_bn(x=stage3_out, filters=self.num_target_classes,
                                freeze_bn=False, kernel_size=(1, 1), use_batchnorm=True)
        fuse_feed1 = keras.layers.add([upscore, score_feed1])

        upscore4 = convTranspose2D_bn(x=fuse_feed1, filters=self.num_target_classes,
                                      freeze_bn=False, use_batchnorm=True)
        score_feed2 = conv2D_bn(x=stage2_out, filters=self.num_target_classes,
                                freeze_bn=False, kernel_size=(1, 1), use_batchnorm=True)
        fuse_feed2 = keras.layers.add([upscore4, score_feed2])

        out = keras.layers.Conv2DTranspose(
                                          filters=self.num_target_classes,
                                          kernel_size=(16, 16),
                                          strides=(8, 8),
                                          padding='same',
                                          activation=None,
                                          data_format="channels_first")(fuse_feed2)

        if export:
            logger.debug("Building model for export")
            out = keras.layers.Permute((2, 3, 1))(out)
            out = keras.layers.Softmax(axis=-1)(out)

        model_unet = Model(inputs=encoder_model.input, outputs=out)
        print(model_unet.summary())
        return model_unet

    def get_base_model(self, args, kwargs):
        """Function to construct model specific backbone."""

        model_class = ShuffleNet
        model = model_class(input_shape=kwargs['input_shape'], classes=self.num_target_classes)
        return model
