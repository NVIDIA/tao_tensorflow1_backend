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
"""Data source for image files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import lru_cache
import os

import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader import types
from nvidia_tao_tf1.blocks.multi_source_loader.sources import data_source
from nvidia_tao_tf1.core.coreobject import save_args


class ImageDataSource(data_source.DataSource):
    """DataSource to read a list of images."""

    @save_args
    def __init__(self, image_paths, extension, image_shape, **kwargs):
        """
        Initialize the image paths, preprocessing Pipeline object, encoding and image shape.

        Args:
            image_paths (list(str)): List of paths to the image files.
            extension (str): Image file extension (encoding) ('.fp16' or '.jpg').
            image_shape (`FrameShape`): Shape of the image in the form HWC.
        """
        super(ImageDataSource, self).__init__(**kwargs)
        self.extension = extension
        self._image_paths = image_paths
        self._image_shape = image_shape

    # TODO(mlehr): refactor the method to ImageDecoder class.
    def _decode_function(self, image, image_id):
        if self.extension == ".fp16":
            # Raw images have pixels in the [0, 1] range and do not contain shape information.
            # Networks are optimized for GPUs and expect CHW ordered inputs.
            # Raw images are stored in CHW - no need to reorder dimensions.
            image_shape = [
                self._image_shape.channels,
                self._image_shape.height,
                self._image_shape.width,
            ]
            decoded_image = tf.io.decode_raw(image, tf.float16, little_endian=None)
            casted_image = tf.cast(decoded_image, tf.float32)
            processed_image = tf.reshape(casted_image, image_shape)
        elif self.extension in [".jpg", ".jpeg"]:
            decoded_image = tf.image.decode_jpeg(image, channels=3)  # (H, W, C) [C:RGB]
            decoded_image.set_shape(
                [
                    self._image_shape.height,
                    self._image_shape.width,
                    self._image_shape.channels,
                ]
            )
            processed_image = tf.cast(
                tf.transpose(a=decoded_image, perm=[2, 0, 1]), tf.float32
            )
            processed_image /= 255.0
        else:
            raise ValueError("Unrecognized image extension: {}".format(self.extension))
        return processed_image, image_id

    @lru_cache()
    def __len__(self):
        """Return the number of images to visualize."""
        return len(self._image_paths)

    def get_image_properties(self):
        """Returns the width and height of the images for this source."""
        return self._image_shape.width, self._image_shape.height

    def call(self):
        """Return a tf.data.Dataset for this data source."""

        def generator():
            for image_path in self._image_paths:
                # image_path --> for reading the file. 2nd element is the frame id.
                yield image_path, os.path.splitext(os.path.basename(image_path))[0]

        dataset = tf.data.Dataset.from_generator(
            generator=generator, output_types=tf.string
        )

        # x[0] = image_path, x[1] = frame_id
        dataset = dataset.map(map_func=lambda x: (tf.io.read_file(x[0]), x[1]))
        dataset = dataset.map(self._decode_function)

        return dataset.map(
            lambda frame, frame_id: types.SequenceExample(
                instances={
                    types.FEATURE_CAMERA: types.Images2D(
                        images=frame,
                        canvas_shape=types.Canvas2D(
                            height=self._image_shape.height,
                            width=self._image_shape.width,
                        ),
                    ),
                    types.FEATURE_SESSION: types.Session(
                        uuid=tf.constant("session_uuid"),
                        camera_name=tf.constant("camera_name"),
                        frame_number=tf.constant(0),
                    ),
                },
                labels={},
            )
        )
