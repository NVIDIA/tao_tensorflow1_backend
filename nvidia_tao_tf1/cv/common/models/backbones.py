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

"""IVA RetinaNet model construction wrapper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.core.templates.cspdarknet import CSPDarkNet
from nvidia_tao_tf1.core.templates.cspdarknet_tiny import CSPDarkNetTiny
from nvidia_tao_tf1.core.templates.darknet import DarkNet
from nvidia_tao_tf1.core.templates.efficientnet import EfficientNetB0
from nvidia_tao_tf1.core.templates.googlenet import GoogLeNet
from nvidia_tao_tf1.core.templates.mobilenet import MobileNet, MobileNetV2
from nvidia_tao_tf1.core.templates.resnet import ResNet
from nvidia_tao_tf1.core.templates.squeezenet import SqueezeNet
from nvidia_tao_tf1.core.templates.vgg import VggNet


def get_efficientnet_b0(
    input_tensor=None,
    data_format='channels_first',
    nclasses=1000,
    use_bias=False,
    kernel_regularizer=None,
    bias_regularizer=None,
    use_imagenet_head=False,
    freeze_bn=False,
    freeze_blocks=None,
    stride16=False,
):
    """Get an EfficientNet B0 model."""
    base_model = EfficientNetB0(
        input_tensor=input_tensor,
        add_head=use_imagenet_head,
        data_format=data_format,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        classes=nclasses,
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        stride16=stride16,
        activation_type='relu'
    )

    return base_model


def get_cspdarknet(nlayers=19,
                   input_tensor=None,
                   data_format='channels_first',
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   freeze_bn=False,
                   freeze_blocks=None,
                   force_relu=False,
                   activation="leaky_relu"):
    """Wrapper to get CSPDarkNet model from IVA templates."""
    base_model = CSPDarkNet(nlayers=nlayers,
                            input_tensor=input_tensor,
                            data_format='channels_first',
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            use_batch_norm=True,
                            add_head=False,
                            use_bias=False,
                            freeze_blocks=freeze_blocks,
                            freeze_bn=freeze_bn,
                            force_relu=force_relu,
                            activation=activation)

    return base_model


def get_cspdarknet_tiny(
    input_tensor=None,
    data_format='channels_first',
    kernel_regularizer=None,
    bias_regularizer=None,
    freeze_bn=False,
    freeze_blocks=None,
    force_relu=False,
    activation="leaky_relu"
):
    """Wrapper to get CSPDarkNetTiny model from IVA templates."""
    base_model = CSPDarkNetTiny(
        input_tensor=input_tensor,
        data_format='channels_first',
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        use_batch_norm=True,
        add_head=False,
        use_bias=False,
        freeze_blocks=freeze_blocks,
        freeze_bn=freeze_bn,
        force_relu=force_relu,
        activation=activation
    )
    return base_model


def get_darknet(nlayers=19,
                input_tensor=None,
                data_format='channels_first',
                kernel_regularizer=None,
                bias_regularizer=None,
                freeze_bn=False,
                freeze_blocks=None,
                force_relu=False):
    """Wrapper to get DarkNet model from IVA templates."""
    base_model = DarkNet(nlayers=nlayers,
                         input_tensor=input_tensor,
                         data_format='channels_first',
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         use_batch_norm=True,
                         add_head=False,
                         use_bias=False,
                         freeze_blocks=freeze_blocks,
                         freeze_bn=freeze_bn,
                         force_relu=force_relu)

    return base_model


def get_googlenet(input_tensor=None,
                  data_format='channels_first',
                  kernel_regularizer=None,
                  bias_regularizer=None,
                  use_batch_norm=True,
                  use_bias=False,
                  freeze_bn=False,
                  freeze_blocks=None):
    """Wrapper to get GoogLeNet model from IVA templates."""
    base_model = GoogLeNet(inputs=input_tensor,
                           data_format=data_format,
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           use_batch_norm=use_batch_norm,
                           activation_type='relu',
                           freeze_bn=freeze_bn,
                           freeze_blocks=freeze_blocks,
                           use_bias=use_bias)
    return base_model


def get_resnet(nlayers=18,
               input_tensor=None,
               data_format='channels_first',
               kernel_regularizer=None,
               bias_regularizer=None,
               all_projections=True,
               use_batch_norm=True,
               use_pooling=False,
               use_bias=False,
               freeze_bn=False,
               freeze_blocks=None):
    """Wrapper to get ResNet model from IVA templates."""
    base_model = ResNet(nlayers=nlayers,
                        input_tensor=input_tensor,
                        data_format=data_format,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        use_batch_norm=use_batch_norm,
                        activation_type='relu',
                        all_projections=all_projections,
                        use_pooling=use_pooling,
                        freeze_blocks=freeze_blocks,
                        freeze_bn=freeze_bn,
                        use_bias=use_bias)
    return base_model


def get_vgg(nlayers=16,
            input_tensor=None,
            data_format="channels_first",
            kernel_regularizer=None,
            bias_regularizer=None,
            use_batch_norm=True,
            use_pooling=False,
            use_bias=False,
            freeze_bn=False,
            freeze_blocks=None):
    """Wrapper to get VGG model from IVA templates."""
    base_model = VggNet(nlayers=nlayers,
                        inputs=input_tensor,
                        data_format=data_format,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        use_batch_norm=use_batch_norm,
                        activation_type='relu',
                        use_pooling=use_pooling,
                        freeze_bn=freeze_bn,
                        freeze_blocks=freeze_blocks,
                        use_bias=use_bias)

    return base_model


def get_mobilenet(input_tensor=None,
                  data_format='channels_first',
                  kernel_regularizer=None,
                  bias_regularizer=None,
                  use_batch_norm=True,
                  use_bias=False,
                  freeze_bn=False,
                  freeze_blocks=None,
                  stride=16):
    """Wrapper to get MobileNet model from IVA templates."""
    base_model = MobileNet(inputs=input_tensor,
                           dropout=0.0,
                           data_format=data_format,
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           use_batch_norm=use_batch_norm,
                           use_bias=use_bias,
                           freeze_bn=freeze_bn,
                           freeze_blocks=freeze_blocks,
                           stride=stride,
                           add_head=False)
    return base_model


def get_mobilenet_v2(input_tensor=None,
                     data_format='channels_first',
                     kernel_regularizer=None,
                     bias_regularizer=None,
                     use_batch_norm=True,
                     all_projections=False,
                     use_bias=False,
                     freeze_bn=False,
                     freeze_blocks=None,
                     stride=16):
    """Wrapper to get MobileNet V2 model from IVA templates."""
    base_model = MobileNetV2(inputs=input_tensor,
                             data_format=data_format,
                             kernel_regularizer=kernel_regularizer,
                             bias_regularizer=bias_regularizer,
                             all_projections=all_projections,
                             use_batch_norm=use_batch_norm,
                             use_bias=use_bias,
                             freeze_bn=freeze_bn,
                             freeze_blocks=freeze_blocks,
                             stride=stride,
                             add_head=False)
    return base_model


def get_squeezenet(input_tensor=None,
                   data_format='channels_first',
                   dropout=1e-3,
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   freeze_blocks=None):
    """Wrapper to get SqueezeNet model from IVA templates."""
    base_model = SqueezeNet(inputs=input_tensor,
                            dropout=1e-3,
                            data_format=data_format,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            freeze_blocks=None)
    return base_model


model_choose = {"resnet": get_resnet,
                "vgg": get_vgg,
                "googlenet": get_googlenet,
                "mobilenet_v1": get_mobilenet,
                "mobilenet_v2": get_mobilenet_v2,
                "squeezenet": get_squeezenet,
                "darknet": get_darknet,
                'cspdarknet': get_cspdarknet,
                "cspdarknet_tiny": get_cspdarknet_tiny,
                "cspdarknet_tiny_3l": get_cspdarknet_tiny,
                "efficientnet_b0": get_efficientnet_b0}


def get_backbone(input_tensor,
                 backbone,
                 data_format=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 freeze_blocks=None,
                 **kwargs):
    """Wrapper to chose model defined in iva templates."""
    # defining model dictionary
    kwa = dict()
    if backbone == 'googlenet':
        kwa['use_batch_norm'] = kwargs['use_batch_norm']
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
    elif backbone in ['darknet', 'cspdarknet']:
        kwa['nlayers'] = kwargs['nlayers']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['force_relu'] = kwargs['force_relu']
        if backbone == "cspdarknet":
            kwa['activation'] = kwargs['activation']
    elif backbone == "cspdarknet_tiny":
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['force_relu'] = kwargs['force_relu']
        kwa['activation'] = kwargs['activation']
    elif backbone == "cspdarknet_tiny_3l":
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['force_relu'] = kwargs['force_relu']
        kwa['activation'] = kwargs['activation']
    elif backbone == 'resnet':
        kwa['nlayers'] = kwargs['nlayers']
        kwa['use_batch_norm'] = kwargs['use_batch_norm']
        kwa['use_pooling'] = kwargs['use_pooling']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['use_bias'] = kwargs['use_bias']
        kwa['all_projections'] = kwargs['all_projections']
    elif backbone == 'vgg':
        kwa['nlayers'] = kwargs['nlayers']
        kwa['use_batch_norm'] = kwargs['use_batch_norm']
        kwa['use_pooling'] = kwargs['use_pooling']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['use_bias'] = kwargs['use_bias']
    elif backbone == 'mobilenet_v1':
        kwa['use_batch_norm'] = kwargs['use_batch_norm']
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
    elif backbone == 'mobilenet_v2':
        kwa['use_batch_norm'] = kwargs['use_batch_norm']
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['all_projections'] = kwargs['all_projections']
    elif backbone == 'squeezenet':
        kwa['dropout'] = kwargs['dropout']
    elif backbone == "efficientnet_b0":
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
    else:
        raise ValueError('Unsupported backbone model: {}'.format(backbone))

    model = model_choose[backbone](input_tensor=input_tensor,
                                   data_format=data_format,
                                   kernel_regularizer=kernel_regularizer,
                                   bias_regularizer=bias_regularizer,
                                   freeze_blocks=freeze_blocks,
                                   **kwa)
    return model
