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

"""SSD tensorboard visualization callback."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import keras
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

from nvidia_tao_tf1.cv.common.utils import summary_from_value, TensorBoard


class SSDTensorBoard(TensorBoard):
    """Override the on_epoch_end."""

    def on_epoch_end(self, epoch, logs=None):
        """on_epoch_end method."""
        for name, value in logs.items():
            if "AP" in name or "validation_loss" in name:
                summary = summary_from_value(name, value.item())
                self.writer.add_summary(summary, epoch)
                self.writer.flush()


def make_image_bboxes(tensor, bboxes):
    """Convert an numpy representation image to Image protobuf."""
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle(
            ((bbox[0], bbox[1]), (bbox[2], bbox[3])),
            outline="red"
        )
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


class SSDTensorBoardImage(keras.callbacks.Callback):
    """Add the augmented images to Tensorboard and draw bboxes on the images."""

    def __init__(self, log_dir, experiment_spec, variances, num_imgs=3):
        """Init the TensorBoardImage."""
        super(SSDTensorBoardImage, self).__init__()

        self.img = tf.Variable(0., collections=[tf.GraphKeys.LOCAL_VARIABLES], validate_shape=False)
        self.label = tf.Variable(0., collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                 validate_shape=False)
        self.writer = tf.summary.FileWriter(log_dir)
        self.num_imgs = num_imgs

        # init the de-mean parameter
        augmentation_config = experiment_spec.augmentation_config
        bb, gg, rr = 103.939, 116.779, 123.68
        img_mean = augmentation_config.image_mean
        if img_mean:
            if augmentation_config.output_channel == 3:
                bb, gg, rr = img_mean['b'], img_mean['g'], img_mean['r']
            else:
                bb, gg, rr = img_mean['l'], img_mean['l'], img_mean['l']

        if augmentation_config.output_channel == 3:
            self.img_mean = np.array([[[[bb]], [[gg]], [[rr]]]])
        else:
            g_mean = bb * 0.1140 + gg * 0.5870 + rr * 0.2990
            self.img_mean = np.array([[[[g_mean]]]])

        # init the label decode parameter:
        self.variances = np.array(variances)
        self.img_height = experiment_spec.augmentation_config.output_height
        self.img_width = experiment_spec.augmentation_config.output_width

    def _deprocess_imgs(self, augmented_imgs):
        """Deprocess the images."""
        # add mean and cast to uint8
        augmented_imgs = (augmented_imgs + self.img_mean).astype("uint8")
        # NCHW -> NHWC:
        augmented_imgs = augmented_imgs.transpose(0, 2, 3, 1)
        # BGR -> RGB:
        augmented_imgs = augmented_imgs[:, :, :, ::-1]

        return augmented_imgs

    def _decode_label(self, labels):
        """Decode the labels."""
        # labels shape: [batch_size, num_anchors, encoded_label]
        final_gt_anchors = []
        for label_per_img in labels:
            # pick up the highest IOU anchors
            gt_anchors = label_per_img[label_per_img[:, -1] == 1024.0, :]
            cx_gt = gt_anchors[:, -12]
            cy_gt = gt_anchors[:, -11]
            w_gt = gt_anchors[:, -10]
            h_gt = gt_anchors[:, -9]
            cx_anchor = gt_anchors[:, -8]
            cy_anchor = gt_anchors[:, -7]
            w_anchor = gt_anchors[:, -6]
            h_anchor = gt_anchors[:, -5]
            cx_variance = self.variances[np.newaxis, -4]
            cy_variance = self.variances[np.newaxis, -3]
            variance_w = self.variances[np.newaxis, -2]
            variance_h = self.variances[np.newaxis, -1]
            # Convert anchor box offsets to image offsets.
            cx = cx_gt * cx_variance * w_anchor + cx_anchor
            cy = cy_gt * cy_variance * h_anchor + cy_anchor
            w = np.exp(w_gt * variance_w) * w_anchor
            h = np.exp(h_gt * variance_h) * h_anchor

            # Convert 'centroids' to 'corners'.
            xmin = cx - 0.5 * w
            ymin = cy - 0.5 * h
            xmax = cx + 0.5 * w
            ymax = cy + 0.5 * h
            xmin = np.expand_dims(xmin * self.img_width, axis=-1)
            ymin = np.expand_dims(ymin * self.img_height, axis=-1)
            xmax = np.expand_dims(xmax * self.img_width, axis=-1)
            ymax = np.expand_dims(ymax * self.img_height, axis=-1)

            decoded_anchors = np.concatenate((xmin, ymin, xmax, ymax), axis=-1)
            decoded_anchors.astype(np.int32)
            final_gt_anchors.append(decoded_anchors)

        return final_gt_anchors

    def on_batch_end(self, batch, logs=None):
        """On batch end method."""
        if batch != 0:
            return
        augmented_imgs = keras.backend.get_value(self.img)
        labels = keras.backend.get_value(self.label)
        augmented_imgs = self._deprocess_imgs(augmented_imgs)
        labels = self._decode_label(labels)
        cnt = 0
        summary_values = []
        for img, label in zip(augmented_imgs, labels):
            if cnt >= self.num_imgs:
                break
            tb_img = make_image_bboxes(img, label)
            summary_values.append(tf.Summary.Value(tag=f"batch_imgs/{cnt}", image=tb_img))
            cnt += 1
        summary = tf.Summary(value=summary_values)
        self.writer.add_summary(summary, batch)
        self.writer.flush()

    def on_train_end(self, *args, **kwargs):
        """on_train_end method."""
        self.writer.close()
