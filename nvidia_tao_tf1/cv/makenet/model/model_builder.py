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

"""IVA MakeNet model construction wrapper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import AveragePooling2D, Dense, Flatten
from keras.layers import Input
from keras.models import Model

from nvidia_tao_tf1.core.templates.alexnet import AlexNet
from nvidia_tao_tf1.core.templates.cspdarknet import CSPDarkNet
from nvidia_tao_tf1.core.templates.cspdarknet_tiny import CSPDarkNetTiny
from nvidia_tao_tf1.core.templates.darknet import DarkNet
from nvidia_tao_tf1.core.templates.efficientnet import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7
)
from nvidia_tao_tf1.core.templates.googlenet import GoogLeNet
from nvidia_tao_tf1.core.templates.mobilenet import MobileNet, MobileNetV2
from nvidia_tao_tf1.core.templates.resnet import ResNet
from nvidia_tao_tf1.core.templates.squeezenet import SqueezeNet
from nvidia_tao_tf1.core.templates.vgg import VggNet
from nvidia_tao_tf1.cv.makenet.utils.helper import model_io


SUPPORTED_ARCHS = [
    "resnet", "vgg", "alexnet", "googlenet",
    "mobilenet_v1", "mobilenet_v2", "squeezenet",
    "darknet", "efficientnet_b0", "efficientnet_b1",
    "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5",
    "efficientnet_b6", "efficientnet_b7",
    "cspdarknet", "cspdarknet_tiny"
]


def add_dense_head(nclasses, base_model, data_format, kernel_regularizer, bias_regularizer):
    """Wrapper to add dense head to the backbone structure."""
    output = base_model.output
    output_shape = output.get_shape().as_list()
    if data_format == 'channels_first':
        pool_size = (output_shape[-2], output_shape[-1])
    else:
        pool_size = (output_shape[-3], output_shape[-2])
    output = AveragePooling2D(pool_size=pool_size, name='avg_pool',
                              data_format=data_format, padding='valid')(output)
    output = Flatten(name='flatten')(output)
    output = Dense(nclasses, activation='softmax', name='predictions',
                   kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(output)
    final_model = Model(inputs=base_model.input, outputs=output, name=base_model.name)
    return final_model


def get_googlenet(input_shape=(3, 224, 224),
                  data_format='channels_first',
                  nclasses=1000,
                  kernel_regularizer=None,
                  bias_regularizer=None,
                  use_batch_norm=True,
                  retain_head=False,
                  use_bias=True,
                  freeze_bn=False,
                  freeze_blocks=None):
    """Wrapper to get GoogLeNet model from IVA templates."""
    input_image = Input(shape=input_shape)
    final_model = GoogLeNet(inputs=input_image,
                            data_format=data_format,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            use_batch_norm=use_batch_norm,
                            activation_type='relu',
                            add_head=retain_head,
                            nclasses=nclasses,
                            freeze_bn=freeze_bn,
                            freeze_blocks=freeze_blocks,
                            use_bias=use_bias)
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)
    return final_model


def get_alexnet(input_shape=(3, 224, 224),
                data_format='channels_first',
                nclasses=1000,
                kernel_regularizer=None,
                bias_regularizer=None,
                retain_head=False,
                freeze_blocks=None):
    """Wrapper to get AlexNet model from Maglev templates."""
    final_model = AlexNet(input_shape=input_shape,
                          data_format=data_format,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer,
                          add_head=retain_head,
                          nclasses=nclasses,
                          freeze_blocks=freeze_blocks)
    if not retain_head:
        final_model = add_dense_head(nclasses, final_model,
                                     data_format, kernel_regularizer,
                                     bias_regularizer)
    return final_model


def get_resnet(nlayers=18,
               input_shape=(3, 224, 224),
               data_format='channels_first',
               nclasses=1000,
               kernel_regularizer=None,
               bias_regularizer=None,
               all_projections=True,
               use_batch_norm=True,
               use_pooling=False,
               retain_head=False,
               use_bias=True,
               freeze_bn=False,
               freeze_blocks=None):
    """Wrapper to get ResNet model from Maglev templates."""
    input_image = Input(shape=input_shape)
    final_model = ResNet(nlayers=nlayers,
                         input_tensor=input_image,
                         data_format=data_format,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         use_batch_norm=use_batch_norm,
                         activation_type='relu',
                         all_projections=all_projections,
                         use_pooling=use_pooling,
                         add_head=retain_head,
                         nclasses=nclasses,
                         freeze_blocks=freeze_blocks,
                         freeze_bn=freeze_bn,
                         use_bias=use_bias)
    if not retain_head:
        final_model = add_dense_head(nclasses, final_model,
                                     data_format, kernel_regularizer,
                                     bias_regularizer)
    return final_model


def get_darknet(nlayers=19,
                input_shape=(3, 224, 224),
                data_format='channels_first',
                nclasses=1000,
                alpha=0.1,
                kernel_regularizer=None,
                bias_regularizer=None,
                use_batch_norm=True,
                retain_head=False,
                use_bias=False,
                freeze_bn=False,
                freeze_blocks=None):
    """Wrapper to get DarkNet model."""
    input_image = Input(shape=input_shape)
    final_model = DarkNet(nlayers=nlayers,
                          input_tensor=input_image,
                          data_format=data_format,
                          alpha=alpha,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer,
                          use_batch_norm=use_batch_norm,
                          add_head=retain_head,
                          nclasses=nclasses,
                          freeze_blocks=freeze_blocks,
                          freeze_bn=freeze_bn,
                          use_bias=use_bias)
    if not retain_head:
        final_model = add_dense_head(nclasses, final_model,
                                     data_format, kernel_regularizer,
                                     bias_regularizer)
    return final_model


def get_cspdarknet(nlayers=19,
                   input_shape=(3, 224, 224),
                   data_format='channels_first',
                   nclasses=1000,
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   use_batch_norm=True,
                   retain_head=False,
                   use_bias=False,
                   freeze_bn=False,
                   freeze_blocks=None,
                   activation="leaky_relu"):
    """Wrapper to get CSPDarkNet model."""
    input_image = Input(shape=input_shape)
    final_model = CSPDarkNet(nlayers=nlayers,
                             input_tensor=input_image,
                             data_format=data_format,
                             kernel_regularizer=kernel_regularizer,
                             bias_regularizer=bias_regularizer,
                             use_batch_norm=use_batch_norm,
                             add_head=retain_head,
                             nclasses=nclasses,
                             freeze_blocks=freeze_blocks,
                             freeze_bn=freeze_bn,
                             use_bias=use_bias,
                             activation=activation)
    if not retain_head:
        final_model = add_dense_head(nclasses, final_model,
                                     data_format, kernel_regularizer,
                                     bias_regularizer)
    return final_model


def get_cspdarknet_tiny(
    input_shape=(3, 224, 224),
    data_format='channels_first',
    nclasses=1000,
    kernel_regularizer=None,
    bias_regularizer=None,
    use_batch_norm=True,
    retain_head=False,
    use_bias=False,
    freeze_bn=False,
    freeze_blocks=None,
    activation="leaky_relu",
):
    """Wrapper to get CSPDarkNetTiny model."""
    input_image = Input(shape=input_shape)
    final_model = CSPDarkNetTiny(
        input_tensor=input_image,
        data_format=data_format,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        use_batch_norm=use_batch_norm,
        add_head=retain_head,
        nclasses=nclasses,
        freeze_blocks=freeze_blocks,
        freeze_bn=freeze_bn,
        use_bias=use_bias,
        activation=activation
    )
    if not retain_head:
        final_model = add_dense_head(nclasses, final_model,
                                     data_format, kernel_regularizer,
                                     bias_regularizer)
    return final_model


def get_vgg(nlayers=16,
            input_shape=(3, 224, 224),
            data_format="channels_first",
            nclasses=1000,
            kernel_regularizer=None,
            bias_regularizer=None,
            use_batch_norm=True,
            use_pooling=False,
            retain_head=False,
            use_bias=True,
            freeze_bn=False,
            freeze_blocks=None,
            dropout=0.5):
    """Wrapper to get VGG model from IVA templates."""
    input_image = Input(shape=input_shape)
    final_model = VggNet(nlayers=nlayers,
                         inputs=input_image,
                         data_format=data_format,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         use_batch_norm=use_batch_norm,
                         activation_type='relu',
                         use_pooling=use_pooling,
                         add_head=retain_head,
                         nclasses=nclasses,
                         freeze_bn=freeze_bn,
                         freeze_blocks=freeze_blocks,
                         use_bias=use_bias,
                         dropout=dropout)
    if not retain_head:
        final_model = add_dense_head(nclasses, final_model,
                                     data_format, kernel_regularizer,
                                     bias_regularizer)
    return final_model


def get_mobilenet(input_shape=None,
                  data_format='channels_first',
                  nclasses=1000,
                  use_batch_norm=None,
                  kernel_regularizer=None,
                  bias_regularizer=None,
                  retain_head=False,
                  use_bias=True,
                  freeze_bn=False,
                  freeze_blocks=None,
                  stride=32):
    """Wrapper to get MobileNet model from IVA templates."""
    input_image = Input(shape=input_shape)
    final_model = MobileNet(inputs=input_image,
                            input_shape=input_shape,
                            dropout=0.0,
                            add_head=retain_head,
                            stride=stride,
                            data_format=data_format,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            nclasses=nclasses,
                            use_batch_norm=use_batch_norm,
                            use_bias=use_bias,
                            freeze_bn=freeze_bn,
                            freeze_blocks=freeze_blocks)
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)
    return final_model


def get_mobilenet_v2(input_shape=None,
                     data_format='channels_first',
                     nclasses=1000,
                     use_batch_norm=None,
                     kernel_regularizer=None,
                     bias_regularizer=None,
                     retain_head=False,
                     all_projections=False,
                     use_bias=True,
                     freeze_bn=False,
                     freeze_blocks=None,
                     stride=32):
    """Wrapper to get MobileNet V2 model from IVA templates."""
    input_image = Input(shape=input_shape)
    final_model = MobileNetV2(inputs=input_image,
                              input_shape=input_shape,
                              add_head=retain_head,
                              stride=stride,
                              data_format=data_format,
                              kernel_regularizer=kernel_regularizer,
                              bias_regularizer=bias_regularizer,
                              all_projections=all_projections,
                              nclasses=nclasses,
                              use_batch_norm=use_batch_norm,
                              use_bias=use_bias,
                              freeze_bn=freeze_bn,
                              freeze_blocks=freeze_blocks)

    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


def get_squeezenet(input_shape=None,
                   data_format='channels_first',
                   nclasses=1000,
                   dropout=1e-3,
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   retain_head=False,
                   freeze_blocks=None):
    """Wrapper to get SqueezeNet model from IVA templates."""
    input_image = Input(shape=input_shape)
    final_model = SqueezeNet(inputs=input_image,
                             input_shape=input_shape,
                             dropout=1e-3,
                             add_head=retain_head,
                             data_format=data_format,
                             kernel_regularizer=kernel_regularizer,
                             bias_regularizer=bias_regularizer,
                             nclasses=nclasses,
                             freeze_blocks=freeze_blocks)

    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


def get_efficientnet_b0(
    input_shape=None,
    data_format='channels_first',
    nclasses=1000,
    use_bias=False,
    kernel_regularizer=None,
    bias_regularizer=None,
    retain_head=False,
    freeze_bn=False,
    freeze_blocks=None,
    stride16=False,
    activation_type=None
):
    """Get an EfficientNet B0 model."""
    input_image = Input(shape=input_shape)
    final_model = EfficientNetB0(
        input_tensor=input_image,
        input_shape=input_shape,
        add_head=retain_head,
        data_format=data_format,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        classes=nclasses,
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        stride16=stride16,
        activation_type=activation_type
    )
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


def get_efficientnet_b1(
    input_shape=None,
    data_format='channels_first',
    nclasses=1000,
    use_bias=False,
    kernel_regularizer=None,
    bias_regularizer=None,
    retain_head=False,
    freeze_bn=False,
    freeze_blocks=None,
    stride16=False,
    activation_type=None
):
    """Get an EfficientNet B1 model."""
    input_image = Input(shape=input_shape)
    final_model = EfficientNetB1(
        input_tensor=input_image,
        input_shape=input_shape,
        add_head=retain_head,
        data_format=data_format,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        classes=nclasses,
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        stride16=stride16,
        activation_type=activation_type
    )
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


def get_efficientnet_b2(
    input_shape=None,
    data_format='channels_first',
    nclasses=1000,
    use_bias=False,
    kernel_regularizer=None,
    bias_regularizer=None,
    retain_head=False,
    freeze_bn=False,
    freeze_blocks=None,
    stride16=False,
    activation_type=None
):
    """Get an EfficientNet B2 model."""
    input_image = Input(shape=input_shape)
    final_model = EfficientNetB2(
        input_tensor=input_image,
        input_shape=input_shape,
        add_head=retain_head,
        data_format=data_format,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        classes=nclasses,
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        stride16=stride16,
        activation_type=activation_type
    )
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


def get_efficientnet_b3(
    input_shape=None,
    data_format='channels_first',
    nclasses=1000,
    use_bias=False,
    kernel_regularizer=None,
    bias_regularizer=None,
    retain_head=False,
    freeze_bn=False,
    freeze_blocks=None,
    stride16=False,
    activation_type=None
):
    """Get an EfficientNet B3 model."""
    input_image = Input(shape=input_shape)
    final_model = EfficientNetB3(
        input_tensor=input_image,
        input_shape=input_shape,
        add_head=retain_head,
        data_format=data_format,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        classes=nclasses,
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        stride16=stride16,
        activation_type=activation_type
    )
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


def get_efficientnet_b4(
    input_shape=None,
    data_format='channels_first',
    nclasses=1000,
    use_bias=False,
    kernel_regularizer=None,
    bias_regularizer=None,
    retain_head=False,
    freeze_bn=False,
    freeze_blocks=None,
    stride16=False,
    activation_type=None
):
    """Get an EfficientNet B4 model."""
    input_image = Input(shape=input_shape)
    final_model = EfficientNetB4(
        input_tensor=input_image,
        input_shape=input_shape,
        add_head=retain_head,
        data_format=data_format,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        classes=nclasses,
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        stride16=stride16,
        activation_type=activation_type
    )
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


def get_efficientnet_b5(
    input_shape=None,
    data_format='channels_first',
    nclasses=1000,
    use_bias=False,
    kernel_regularizer=None,
    bias_regularizer=None,
    retain_head=False,
    freeze_bn=False,
    freeze_blocks=None,
    stride16=False,
    activation_type=None
):
    """Get an EfficientNet B5 model."""
    input_image = Input(shape=input_shape)
    final_model = EfficientNetB5(
        input_tensor=input_image,
        input_shape=input_shape,
        add_head=retain_head,
        data_format=data_format,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        classes=nclasses,
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        stride16=stride16,
        activation_type=activation_type
    )
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


def get_efficientnet_b6(
    input_shape=None,
    data_format='channels_first',
    nclasses=1000,
    use_bias=False,
    kernel_regularizer=None,
    bias_regularizer=None,
    retain_head=False,
    freeze_bn=False,
    freeze_blocks=None,
    stride16=False,
    activation_type=None
):
    """Get an EfficientNet B6 model."""
    input_image = Input(shape=input_shape)
    final_model = EfficientNetB6(
        input_tensor=input_image,
        input_shape=input_shape,
        add_head=retain_head,
        data_format=data_format,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        classes=nclasses,
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        stride16=stride16,
        activation_type=activation_type
    )
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


def get_efficientnet_b7(
    input_shape=None,
    data_format='channels_first',
    nclasses=1000,
    use_bias=False,
    kernel_regularizer=None,
    bias_regularizer=None,
    retain_head=False,
    freeze_bn=False,
    freeze_blocks=None,
    stride16=False,
    activation_type=None
):
    """Get an EfficientNet B7 model."""
    input_image = Input(shape=input_shape)
    final_model = EfficientNetB7(
        input_tensor=input_image,
        input_shape=input_shape,
        add_head=retain_head,
        data_format=data_format,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        classes=nclasses,
        freeze_bn=freeze_bn,
        freeze_blocks=freeze_blocks,
        stride16=stride16,
        activation_type=activation_type
    )
    if not retain_head:
        final_model = add_dense_head(nclasses,
                                     final_model,
                                     data_format,
                                     kernel_regularizer,
                                     bias_regularizer)

    return final_model


# defining model dictionary
model_choose = {"resnet": get_resnet,
                "darknet": get_darknet,
                "cspdarknet": get_cspdarknet,
                "cspdarknet_tiny": get_cspdarknet_tiny,
                "vgg": get_vgg,
                "googlenet": get_googlenet,
                "alexnet": get_alexnet,
                "mobilenet_v1": get_mobilenet,
                "mobilenet_v2": get_mobilenet_v2,
                "squeezenet": get_squeezenet,
                "efficientnet_b0": get_efficientnet_b0,
                "efficientnet_b1": get_efficientnet_b1,
                "efficientnet_b2": get_efficientnet_b2,
                "efficientnet_b3": get_efficientnet_b3,
                "efficientnet_b4": get_efficientnet_b4,
                "efficientnet_b5": get_efficientnet_b5,
                "efficientnet_b6": get_efficientnet_b6,
                "efficientnet_b7": get_efficientnet_b7}


def get_model(arch="resnet",
              input_shape=(3, 224, 224),
              data_format=None,
              nclasses=1000,
              kernel_regularizer=None,
              bias_regularizer=None,
              retain_head=False,
              freeze_blocks=None,
              **kwargs):
    """Wrapper to chose model defined in iva templates."""

    kwa = dict()
    if arch == 'googlenet':
        kwa['use_batch_norm'] = kwargs['use_batch_norm']
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
    elif arch == 'alexnet':
        pass
    elif arch == 'resnet':
        kwa['nlayers'] = kwargs['nlayers']
        kwa['use_batch_norm'] = kwargs['use_batch_norm']
        kwa['use_pooling'] = kwargs['use_pooling']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['use_bias'] = kwargs['use_bias']
        kwa['all_projections'] = kwargs['all_projections']
    elif arch in ['darknet', 'cspdarknet']:
        kwa['nlayers'] = kwargs['nlayers']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['use_bias'] = kwargs['use_bias']
        kwa['use_batch_norm'] = kwargs['use_batch_norm']
        if arch == "cspdarknet":
            kwa["activation"] = kwargs["activation"].activation_type
    elif arch in ["cspdarknet_tiny"]:
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['use_bias'] = kwargs['use_bias']
        kwa['use_batch_norm'] = kwargs['use_batch_norm']
        kwa["activation"] = kwargs["activation"].activation_type
    elif arch == 'vgg':
        kwa['nlayers'] = kwargs['nlayers']
        kwa['use_batch_norm'] = kwargs['use_batch_norm']
        kwa['use_pooling'] = kwargs['use_pooling']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['use_bias'] = kwargs['use_bias']
        kwa['dropout'] = kwargs['dropout']
    elif arch == 'mobilenet_v1':
        kwa['use_batch_norm'] = kwargs['use_batch_norm']
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
    elif arch == 'mobilenet_v2':
        kwa['use_batch_norm'] = kwargs['use_batch_norm']
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['all_projections'] = kwargs['all_projections']
    elif arch == 'squeezenet':
        kwa['dropout'] = kwargs['dropout']
    elif arch == "efficientnet_b0":
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['activation_type'] = kwargs['activation'].activation_type
    elif arch == "efficientnet_b1":
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['activation_type'] = kwargs['activation'].activation_type
    elif arch == "efficientnet_b2":
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['activation_type'] = kwargs['activation'].activation_type
    elif arch == "efficientnet_b3":
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['activation_type'] = kwargs['activation'].activation_type
    elif arch == "efficientnet_b4":
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['activation_type'] = kwargs['activation'].activation_type
    elif arch == "efficientnet_b5":
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['activation_type'] = kwargs['activation'].activation_type
    elif arch == "efficientnet_b6":
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['activation_type'] = kwargs['activation'].activation_type
    elif arch == "efficientnet_b7":
        kwa['use_bias'] = kwargs['use_bias']
        kwa['freeze_bn'] = kwargs['freeze_bn']
        kwa['activation_type'] = kwargs['activation'].activation_type
    else:
        raise ValueError('Unsupported architecture: {}'.format(arch))

    model = model_choose[arch](input_shape=input_shape,
                               nclasses=nclasses,
                               data_format=data_format,
                               kernel_regularizer=kernel_regularizer,
                               bias_regularizer=bias_regularizer,
                               retain_head=retain_head,
                               freeze_blocks=freeze_blocks,
                               **kwa)
    return model
