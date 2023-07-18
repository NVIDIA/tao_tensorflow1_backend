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

"""License plate tensorboard visualization callback."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
from PIL import Image
import tensorflow as tf


def make_image(tensor):
    """Convert an numpy representation image to Image protobuf."""
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


class LPRNetTensorBoardImage(tf.keras.callbacks.Callback):
    """Add the augmented images to Tensorboard."""

    def __init__(self, log_dir, num_imgs=3):
        """Init."""
        super(LPRNetTensorBoardImage, self).__init__()

        self.img = tf.Variable(0., collections=[tf.GraphKeys.LOCAL_VARIABLES], validate_shape=False)
        self.writer = tf.summary.FileWriter(log_dir)
        self.num_imgs = num_imgs

    def on_batch_end(self, batch, logs=None):
        """on_batch_end method."""
        augmented_imgs = tf.keras.backend.get_value(self.img)
        # scale to uint8 255:
        augmented_imgs = (augmented_imgs * 255.0).astype("uint8")
        # NCHW -> NHWC:
        augmented_imgs = augmented_imgs.transpose(0, 2, 3, 1)
        # BGR -> RGB:
        augmented_imgs = augmented_imgs[:, :, :, ::-1]
        cnt = 0
        summary_values = []
        for img in augmented_imgs:
            if cnt >= self.num_imgs:
                break
            tb_img = make_image(img)
            summary_values.append(tf.Summary.Value(tag=f"batch_imgs/{cnt}", image=tb_img))
            cnt += 1
        summary = tf.Summary(value=summary_values)
        self.writer.add_summary(summary, batch)
        self.writer.flush()

    def on_train_end(self, *args, **kwargs):
        """on_train_end method."""
        self.writer.close()
