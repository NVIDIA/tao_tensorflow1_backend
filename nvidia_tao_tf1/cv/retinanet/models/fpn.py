"""IVA RetinaNet Feature Pyramid Generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.layers import Conv2D, ReLU, UpSampling2D

from nvidia_tao_tf1.cv.common.models.backbones import get_backbone

fpn_dict = {'vgg16': ('block_3c_relu', 'block_4c_relu', 'block_5c_relu'),
            'vgg19': ('block_3d_relu', 'block_4d_relu', 'block_5d_relu'),
            'resnet10': ('block_1a_relu', 'block_2a_relu', 'block_4a_relu'),
            'resnet18': ('block_1b_relu', 'block_2b_relu', 'block_4b_relu'),
            'resnet34': ('block_1c_relu', 'block_2d_relu', 'block_4c_relu'),
            'resnet50': ('block_1c_relu', 'block_2d_relu', 'block_4c_relu'),
            'resnet101': ('block_1c_relu', 'block_2d_relu', 'block_4c_relu'),
            'googlenet': ('activation_3', 'inception_3b_output', 'inception_5b_output'),
            'mobilenet_v1': ('conv_pw_relu_3', 'conv_pw_relu_5', 'conv_pw_relu_11'),
            'mobilenet_v2': ('re_lu_4', 're_lu_7', 'block_12_add'),
            'squeezenet': ('fire4', 'fire8', 'fire9'),
            'darknet19': ('b3_conv3_lrelu', 'b4_conv5_lrelu', 'b5_conv5_lrelu'),
            'darknet53': ('b3_add8', 'b4_add8', 'b5_add4'),
            'efficientnet_b0': ('block4a_expand_activation',
                                'block6a_expand_activation',
                                'top_activation')}


class FPN:
    """Class for generating feature pyramid."""

    def __init__(self,
                 input_tensor,
                 model_name,
                 **kwargs):
        """Initialize input and backbone."""
        self.input_tensor = input_tensor
        if model_name in ['vgg', 'resnet', 'darknet']:
            self.nlayers = kwargs['nlayers']
            self.model_name = model_name + str(self.nlayers)
        else:
            self.model_name = model_name
            self.nlayers = None
        self.backbone = get_backbone(input_tensor, model_name, **kwargs)

    def generate(self, feature_size, kernel_regularizer):
        """Return a list of feature maps in FPN."""
        options = {
            'padding' : 'same',
            'kernel_initializer' : 'he_normal',
            'kernel_regularizer' : kernel_regularizer,
            # 'use_bias' : False
        }

        if 'darknet' in self.model_name:
            B1, B2, B3 = fpn_dict[self.model_name]
            C3 = self.backbone.get_layer(B1).output
            C4 = self.backbone.get_layer(B2).output
            C5 = self.backbone.get_layer(B3).output
            expand1 = Conv2D(feature_size,
                             kernel_size=1,
                             strides=1,
                             name='expand_conv1',
                             **options)(C5)
            C5 = ReLU(name='expand1_relu')(expand1)
        elif 'efficientnet' in self.model_name:
            B1, B2, B3 = fpn_dict[self.model_name]
            C3 = self.backbone.get_layer(B1).output
            C4 = self.backbone.get_layer(B2).output
            C5 = self.backbone.get_layer(B3).output
        else:
            _, B2, B3 = fpn_dict[self.model_name]
            C3 = self.backbone.get_layer(B2).output
            C4 = self.backbone.get_layer(B3).output
            expand1 = Conv2D(feature_size,
                             kernel_size=3,
                             strides=2,
                             name='expand_conv1',
                             **options)(C4)
            C5 = ReLU(name='expand1_relu')(expand1)

        # Extra feature maps
        P5 = Conv2D(feature_size, kernel_size=1, strides=1,
                    name='C5_reduced', **options)(C5)
        P5_upsampled = UpSampling2D(size=(2, 2),
                                    data_format='channels_first',
                                    name='P5_upsampled')(P5)
        P5 = Conv2D(feature_size, kernel_size=3, strides=1,
                    name='P5', **options)(P5)
        P5 = ReLU(name='P5_relu')(P5)
        # add P5 elementwise to C4
        P4 = Conv2D(feature_size, kernel_size=1, strides=1,
                    name='C4_reduced', **options)(C4)
        P4 = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
        P4_upsampled = UpSampling2D(size=(2, 2),
                                    data_format='channels_first',
                                    name='P4_upsampled')(P4)
        P4 = Conv2D(feature_size, kernel_size=3, strides=1,
                    name='P4', **options)(P4)
        P4 = ReLU(name='P4_relu')(P4)

        # add P4 elementwise to C3
        P3 = Conv2D(feature_size, kernel_size=1, strides=1,
                    name='C3_reduced', **options)(C3)
        P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
        P3 = Conv2D(feature_size, kernel_size=3, strides=1,
                    name='P3', **options)(P3)
        P3 = ReLU(name='P3_relu')(P3)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        P6 = Conv2D(feature_size, kernel_size=3, strides=2,
                    name='P6', **options)(C5)
        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        P6 = ReLU(name='P6_relu')(P6)
        P7 = Conv2D(feature_size, kernel_size=3, strides=2,
                    name='P7', **options)(P6)
        P7 = ReLU(name='P7_relu')(P7)

        return P3, P4, P5, P6, P7
