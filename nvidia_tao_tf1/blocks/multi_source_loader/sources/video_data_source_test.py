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

"""Main test for video_data_source.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader import data_loader
from nvidia_tao_tf1.blocks.multi_source_loader import types
from nvidia_tao_tf1.blocks.multi_source_loader.sources.video_data_source import (
    VideoDataSource,
)
from nvidia_tao_tf1.core.coreobject import deserialize_tao_object


def _create_h264_video(video_path, frame_count, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    i = 0
    while i < frame_count:
        x = np.random.randint(255, size=(480, 640, 3)).astype("uint8")
        video_writer.write(x)
        i = i + 1


def test_video_data_source(tmpdir_factory):
    tmp_dir = str(tmpdir_factory.mktemp("video"))
    video_path = os.path.join(tmp_dir, "video1.h264")
    _create_h264_video(video_path, 5, 25, 640, 480)
    video_source = VideoDataSource(video_path)
    assert len(video_source) == 5
    assert video_source.height == 480
    assert video_source.width == 640
    assert video_source.frames_per_second == 25

    data_loader_ = data_loader.DataLoader(
        [video_source], augmentation_pipeline=[], batch_size=1
    )
    data_loader_.set_shard()
    batch = data_loader_()
    assert len(data_loader_) == 5
    with tf.compat.v1.Session() as sess:
        for _ in range(len(data_loader_)):
            sess.run(tf.compat.v1.get_collection("iterator_init"))
            example = sess.run(batch)
            image = example.instances[types.FEATURE_CAMERA].images
            assert image.shape == (1, 3, 480, 640)


def test_serialization_and_deserialization(tmpdir_factory):
    tmp_dir = str(tmpdir_factory.mktemp("video"))
    video_path = os.path.join(tmp_dir, "video1.h264")
    _create_h264_video(video_path, 5, 25, 640, 480)
    source = VideoDataSource(video_path)
    source_dict = source.serialize()
    deserialized_source = deserialize_tao_object(source_dict)
    assert source._video_path == deserialized_source._video_path
    assert source._height == deserialized_source._height
    assert source._width == deserialized_source._width
    assert source._fps == deserialized_source._fps
