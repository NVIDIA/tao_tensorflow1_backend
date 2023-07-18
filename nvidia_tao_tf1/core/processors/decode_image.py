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

"""Decode Image Processor."""

import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.processors import Processor
from nvidia_tao_tf1.core.types import data_format as modulus_data_format, DataFormat


class DecodeImage(Processor):
    """Processor for decoding and reshaping an input to an image tensor.

    Args:
        encoding (str): The expected encoding ('fp16', 'jpg', 'png').
        shape (tuple): An optional explicit reshape after loading. Especially relevant for loading
            from raw.
        data_format (str): A string representing the dimension ordering of the input data.
            Must be one of 'channels_last' or 'channels_first'. If ``None`` (default), the
            modulus global default will be used.
        channels (int): Amount of channels of the image. Only relevant for compressed formats.
        normalize (float): An optional normalization factor for color/pixel values.  When set, all
            pixels will be divided by this value.
        uint8_precision (bool): Whether or not cast the input image so that the precision of the
            output image is uint8, but data type is still fp32. This is used when training images
            consist of both .png/.jpeg and .fp16 images.
        kwargs (dict): keyword arguments passed to parent class.

    Raises:
        NotImplementedError: if ``data_format`` is not in ['channels_first', 'channels_last'], or
            if ``encoding`` is not one of ['fp16', 'jpg', 'jpeg', 'png'].
    """

    @save_args
    def __init__(
        self,
        encoding,
        shape=None,
        data_format=None,
        channels=3,
        normalize=None,
        uint8_precision=False,
        **kwargs
    ):
        """__init__ method."""
        self.encoding = encoding.lower()
        self.shape = shape
        self.data_format = (
            data_format if data_format is not None else modulus_data_format()
        )
        self.channels = channels
        self.normalize = normalize
        self.uint8_precision = uint8_precision

        if self.data_format not in [
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_LAST,
        ]:
            raise NotImplementedError(
                "Data format not supported, must be 'channels_first' or "
                "'channels_last', given {}.".format(self.data_format)
            )

        if self.encoding not in ["fp16", "jpg", "jpeg", "png"]:
            raise NotImplementedError(
                "Encoding not supported, must be one of ['fp16', 'jpg', 'jpeg', 'png'], "
                "given {}.".format(self.encoding)
            )

        super(DecodeImage, self).__init__(**kwargs)

    def call(self, data):
        """call method.

        Args:
            data (tensor): Tensor to be loaded (and decoded).

        Returns:
            tensor: Decoded image.
        """
        if self.encoding == "fp16":
            img = tf.io.decode_raw(data, tf.float16, little_endian=None)
            if self.uint8_precision:  # change float16 images' precision to uint8
                img *= 255
                img = tf.cast(img, tf.uint8)
                self.normalize = 255.0
        elif self.encoding in ["jpg", "jpeg"]:
            img = tf.image.decode_jpeg(
                data, channels=self.channels
            )  # (H, W, C) [C:RGB]
        elif self.encoding == "png":
            img = tf.image.decode_png(data, channels=self.channels)  # (H, W, C) [C:RGB]
        img = tf.cast(img, tf.float32)
        if self.shape:
            img = tf.reshape(img, self.shape)
        if self.data_format == DataFormat.CHANNELS_FIRST and self.encoding != "fp16":
            img = DataFormat.convert(
                img, DataFormat.CHANNELS_LAST, DataFormat.CHANNELS_FIRST
            )
        if self.normalize:
            img /= self.normalize
        return img
