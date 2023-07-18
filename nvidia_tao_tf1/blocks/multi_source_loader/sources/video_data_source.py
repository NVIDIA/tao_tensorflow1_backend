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
"""Data source for video files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import lru_cache

import cv2
import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader import types
from nvidia_tao_tf1.blocks.multi_source_loader.sources import data_source
from nvidia_tao_tf1.core.coreobject import save_args


class VideoDataSource(data_source.DataSource):
    """DataSource for reading frames from video files."""

    @save_args
    def __init__(self, video_path, **kwargs):
        """
        Initialize video path, and preprocessing Pipeline object.

        Args:
            video_path (str): Absolute path to the video file.
        """
        super(VideoDataSource, self).__init__(**kwargs)
        self._video_path = video_path
        vid = cv2.VideoCapture(video_path)
        self._height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._fps = vid.get(cv2.CAP_PROP_FPS)
        vid.release()

    @lru_cache()
    def __len__(self):
        """Return the number of frames."""
        frame_counter = 0
        vid = cv2.VideoCapture(self._video_path)
        while vid.isOpened():
            success, _ = vid.read()
            if not success:
                break
            frame_counter += 1
        vid.release()
        return frame_counter

    def _generator(self):
        vid = cv2.VideoCapture(self._video_path)
        while True:
            success, frame = vid.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0
            yield frame
        vid.release()

    @property
    def height(self):
        """Return video frame height."""
        return self._height

    @property
    def width(self):
        """Return video frame width."""
        return self._width

    def get_image_properties(self):
        """Returns the width and height of the frames within the video."""
        return self.width, self.height

    @property
    def frames_per_second(self):
        """Return frames per second."""
        return self._fps

    def call(self):
        """Return a tf.data.Dataset for this data source."""
        dataset = tf.data.Dataset.from_generator(
            generator=self._generator,
            output_types=tf.float32,
            output_shapes=tf.TensorShape([self.height, self.width, 3]),
        )
        return dataset.map(
            lambda frame: types.SequenceExample(
                instances={
                    types.FEATURE_CAMERA: types.Images2D(
                        images=tf.transpose(a=frame, perm=[2, 0, 1]),
                        canvas_shape=types.Canvas2D(
                            height=self.height, width=self.width
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

    @property
    def image_dtype(self):
        """Return the dtype of this data source."""
        return tf.float32
