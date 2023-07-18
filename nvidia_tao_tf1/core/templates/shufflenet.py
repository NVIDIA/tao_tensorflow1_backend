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

"""Shufflenet Encoder model template class."""

import keras
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, Concatenate, Input
from keras.layers import AveragePooling2D, BatchNormalization, Conv2D, MaxPooling2D
from keras.layers import DepthwiseConv2D
from keras.models import Model
from keras_applications.imagenet_utils import _obtain_input_shape
import numpy as np

K.set_image_data_format('channels_first')


def ShuffleNet(include_top=False, input_tensor=None, scale_factor=1.0, pooling='max',
               input_shape=(224, 224, 3), groups=1, load_model=None, bottleneck_ratio=0.25,
               classes=1000):
    """ShuffleNet implementation.

    ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
    Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun
    https://arxiv.org/pdf/1707.01083.pdf
    Note that only TensorFlow is supported for now, therefore it only works
    with the data format `image_data_format='channels_last'` in your Keras
    config at `~/.keras/keras.json`.
    Args:
        include_top: bool(True)
             whether to include the fully-connected layer at the top of the network.
        input_tensor:
            optional Keras tensor (i.e. output of `layers.Input()`) to use as
            image input for the model.
        scale_factor:
            scales the number of output channels
        input_shape:
        pooling:
            Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        groups: int
            number of groups per channel
        num_shuffle_units: list([3,7,3])
            number of stages (list length) and the number of shufflenet units in a
            stage beginning with stage 2 because stage 1 is fixed
            e.g. idx 0 contains 3 + 1 (first shuffle unit in each stage differs)
            shufflenet units for stage 2 idx 1 contains 7 + 1 Shufflenet Units
            for stage 3 and idx 2 contains 3 + 1 Shufflenet Units
        bottleneck_ratio:
            bottleneck ratio implies the ratio of bottleneck channels to output channels.
            For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
            the width of the bottleneck feature map.
        classes: int(1000)
            number of classes to predict
    """

    if K.backend() != 'tensorflow':
        raise RuntimeError('Only TensorFlow backend is currently supported, '
                           'as other backends do not support ')

    num_shuffle_units = [3, 7, 3]
    name = "ShuffleNet_%.2gX_g%d_br_%.2g_%s" % (scale_factor, groups, bottleneck_ratio,
                                                "".join([str(x) for x in num_shuffle_units]))
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=28,
                                      require_flatten=include_top,
                                      data_format='channels_first')

    out_dim_stage_two = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}
    if groups not in out_dim_stage_two:
        raise ValueError("Invalid number of groups.")

    if pooling not in ['max', 'avg']:
        raise ValueError("Invalid value for pooling.")

    if not (float(scale_factor) * 4).is_integer():
        raise ValueError("Invalid value for scale_factor. Should be x over 4.")

    exp = np.insert(np.arange(0, len(num_shuffle_units), dtype=np.float32), 0, 0)
    out_channels_in_stage = 2 ** exp
    out_channels_in_stage *= out_dim_stage_two[groups]  # calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # create shufflenet architecture
    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same',
               use_bias=False, strides=(2, 2), activation="relu", name="conv1",
               data_format='channels_first')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same',
                     name="maxpool1", data_format='channels_first')(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(0, len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = _block(x, out_channels_in_stage, repeat=repeat,
                   bottleneck_ratio=bottleneck_ratio,
                   groups=groups, stage=stage + 2)

    x = keras.layers.Conv2D(filters=classes, kernel_size=(1, 1), name="score_fr",
                            data_format="channels_first")(x)
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs=inputs, outputs=x, name=name)

    if load_model is not None:
        model.load_weights('', by_name=True)

    return model


def _block(x, channel_map, bottleneck_ratio, repeat=1, groups=1, stage=1):
    """Creates a bottleneck block containing `repeat + 1` shuffle units.

    Args:
        x:
            Input tensor of with `channels_last` data format
        channel_map: list
            list containing the number of output channels for a stage
        repeat: int(1)
            number of repetitions for a shuffle unit with stride 1
        groups: int(1)
            number of groups per channel
        bottleneck_ratio: float
            bottleneck ratio implies the ratio of bottleneck channels to output channels.
            For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
            the width of the bottleneck feature map.
        stage: int(1)
            stage number

    """
    x = _shuffle_unit(x, in_channels=channel_map[stage - 2],
                      out_channels=channel_map[stage - 1], strides=2,
                      groups=groups, bottleneck_ratio=bottleneck_ratio,
                      stage=stage, block=1)

    for i in range(1, repeat + 1):
        x = _shuffle_unit(x, in_channels=channel_map[stage - 1],
                          out_channels=channel_map[stage - 1], strides=1,
                          groups=groups, bottleneck_ratio=bottleneck_ratio,
                          stage=stage, block=(i + 1))

    return x


def _shuffle_unit(inputs, in_channels, out_channels, groups, bottleneck_ratio,
                  strides=2, stage=1, block=1):
    """Creates a shuffleunit.

    Args:
        inputs:
            Input tensor of with `channels_last` data format
        in_channels:
            number of input channels
        out_channels:
            number of output channels
        strides:
            An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
        groups: int(1)
            number of groups per channel
        bottleneck_ratio: float
            bottleneck ratio implies the ratio of bottleneck channels to output channels.
            For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
            the width of the bottleneck feature map.
        stage: int(1)
            stage number
        block: int(1)
            block number
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    prefix = 'stage%d/block%d' % (stage, block)

    # default: 1/4 of the output channel of a ShuffleNet Unit
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    groups = (1 if stage == 2 and block == 1 else groups)

    x = _group_conv(inputs, in_channels, out_channels=bottleneck_channels,
                    groups=(1 if stage == 2 and block == 1 else groups),
                    name='%s/1x1_gconv_1' % prefix)
    x = BatchNormalization(axis=bn_axis, name='%s/bn_gconv_1' % prefix)(x)
    x = Activation('relu', name='%s/relu_gconv_1' % prefix)(x)

    x = ChannelShuffle(groups=groups)(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=False,
                        strides=strides, data_format="channels_first",
                        name='%s/1x1_dwconv_1' % prefix)(x)
    x = BatchNormalization(axis=bn_axis, name='%s/bn_dwconv_1' % prefix)(x)

    x = _group_conv(x, bottleneck_channels, out_channels=out_channels
                    if strides == 1 else out_channels - in_channels,
                    groups=groups, name='%s/1x1_gconv_2' % prefix)
    x = BatchNormalization(axis=bn_axis, name='%s/bn_gconv_2' % prefix)(x)

    if strides < 2:
        ret = Add(name='%s/add' % prefix)([x, inputs])
    else:
        avg = AveragePooling2D(pool_size=3, strides=2, padding='same',
                               name='%s/avg_pool' % prefix)(inputs)
        ret = Concatenate(bn_axis, name='%s/concat' % prefix)([x, avg])

    ret = Activation('relu', name='%s/relu_out' % prefix)(ret)

    return ret


class GroupLayer(keras.layers.Layer):
    """Group Layer Class."""

    def __init__(self, offset=0, ig=0, **kwargs):
        """Init function.

        Args:
            offset (int): Offset to sample the input.
            ig (int): Number of input channels per groups.
        """
        self.ig = ig
        self.offset = offset
        super(GroupLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """Function to construct the input."""

        return inputs[:, self.offset: self.offset + self.ig, :, :]


class ChannelShuffle(keras.layers.Layer):
    """Channel Shuffle Class."""

    def __init__(self, groups=1, **kwargs):
        """Init function.

        Args:
            groups (int): No. of groups for the group convolution.
        """
        self.groups = groups
        super(ChannelShuffle, self).__init__(**kwargs)

    def call(self, inputs):
        """Function to Shuffle the channels in the input."""

        x = K.permute_dimensions(inputs, (0, 2, 3, 1))  # Made tensor channel last
        height, width, in_channels = x.shape.as_list()[1:]
        channels_per_group = in_channels // self.groups

        x = K.reshape(x, [-1, height, width, self.groups, channels_per_group])
        x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
        x = K.reshape(x, [-1, height, width, in_channels])  # bs x h x w x c
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        return x


def _group_conv(x, in_channels, out_channels, groups, kernel=1, stride=1, name=''):
    """Grouped convolution.

    Args:
        x:
            Input tensor of with `channels_last` data format
        in_channels:
            number of input channels
        out_channels:
            number of output channels
        groups:
            number of groups per channel
        kernel: int(1)
            An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        stride: int(1)
            An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for all spatial dimensions.
        name: str
            A string to specifies the layer name
    """
    if groups == 1:
        return Conv2D(filters=out_channels, kernel_size=kernel, padding='same',
                      use_bias=False, strides=stride,
                      name=name, data_format="channels_first")(x)

    # number of intput channels per group
    ig = in_channels // groups
    assert out_channels % groups == 0
    offset = groups[0] * ig
    cat = GroupLayer(offset=offset, ig=ig)(x)
    for i in range(groups[1:]):
        offset = i * ig
        group = GroupLayer(offset=offset, ig=ig)(x)
        cat = Concatenate(name='%s/concat' % name)([cat, group])

    return cat


def channel_shuffle(x, groups):
    """Shuffle the Channels by grouping.

    Args:
        x:
            Input tensor of with `channels_last` data format
        groups: int
            number of groups per channel
    Returns:
        channel shuffled output tensor
    Examples:
        Example for a 1D Array with 3 groups
        >>> d = np.array([0,1,2,3,4,5,6,7,8])
        >>> x = np.reshape(d, (3,3))
        >>> x = np.transpose(x, [1,0])
        >>> x = np.reshape(x, (9,))
        '[0 1 2 3 4 5 6 7 8] --> [0 3 6 1 4 7 2 5 8]'
    """
    x = K.permute_dimensions(x, (0, 2, 3, 1))  # Made tensor channel last
    height, width, in_channels = x.shape.as_list()[1:]
    channels_per_group = in_channels // groups

    x = K.reshape(x, [-1, height, width, groups, channels_per_group])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
    x = K.reshape(x, [-1, height, width, in_channels])  # bs x h x w x c
    x = K.permute_dimensions(x, (0, 3, 1, 2))

    return x
