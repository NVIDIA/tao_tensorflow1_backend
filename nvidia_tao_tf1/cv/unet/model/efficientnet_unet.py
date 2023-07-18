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

"""EfficientNet Unet model class that takes care of constructing and validating a model."""

import logging
import keras
from keras.models import Model

from nvidia_tao_tf1.core.templates.efficientnet import EfficientNetB0
from nvidia_tao_tf1.cv.unet.model.layers import Conv2DTranspose_block
from nvidia_tao_tf1.cv.unet.model.unet_model import UnetModel

logger = logging.getLogger(__name__)


eff_dict = {'efficientnet_b0': ('block1a_project_bn', 'block2a_project_bn',
                                'block3a_project_bn', 'block5a_project_bn')}


class EfficientUnet(UnetModel):
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
        super(EfficientUnet, self).__init__(*args, **kwargs)

    def construct_decoder_model(self, encoder_model, export=False):
        """Construct the decoder for Unet with EfficientNet as backbone.

        Args:
            encoder_model (keras.model): keras model type.
            export (bool): Set the inference flag to build the
                            inference model with softmax.
        Returns:
            model (keras.model): The entire Unet model with encoder and decoder.
        """
        B1, B2, B3, B4 = eff_dict[self.template]
        S2 = encoder_model.get_layer(B1).output
        S3 = encoder_model.get_layer(B2).output
        S4 = encoder_model.get_layer(B3).output
        S5 = encoder_model.get_layer(B4).output
        skips = [S2, S3, S4, S5]
        out = encoder_model.output
        for filter_tmp in [512, 256, 128, 64, 32]:
                if skips:
                    skip_to_use = skips.pop()
                else:
                    skip_to_use = None
                out = Conv2DTranspose_block(input_tensor=out, filters=filter_tmp,
                                            initializer="glorot_uniform",
                                            skip=skip_to_use,
                                            use_batchnorm=self.use_batch_norm,
                                            freeze_bn=self.freeze_bn)

        out = keras.layers.Conv2D(self.num_target_classes, (1, 1), padding='same',
                                  data_format="channels_first")(out)
        if export:
            logger.debug("Building model for export")
            out = self.get_activation_for_export(out)

        model_unet = Model(inputs=encoder_model.input, outputs=out)
        return model_unet

    def get_base_model(self, args, kwargs):
        """Function to construct model specific backbone."""

        model_class = EfficientNetB0
        kwargs['add_head'] = False
        kwargs['input_tensor'] = args[1]
        kwargs['stride16'] = True
        while args:
            args.pop()

        model = model_class(*args, **kwargs)

        return model
