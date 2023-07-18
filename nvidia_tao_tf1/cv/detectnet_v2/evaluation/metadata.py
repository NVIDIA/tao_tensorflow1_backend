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

"""Gather camera metadata for evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple
import warnings

import numpy as np

from nvidia_tao_tf1.blocks.multi_source_loader.types import Bbox2DLabel

FRAME_ID_KEY = 'frame/id'
CAMERA_LOCATION_KEY = 'frame/camera_location'
IMAGE_DIMENSIONS_KEY = 'frame/image_dimensions'
VALID_METADATA_KEYS = {FRAME_ID_KEY, CAMERA_LOCATION_KEY, IMAGE_DIMENSIONS_KEY}

Frame = namedtuple("Frame", [
    'frame_id',
    'dimensions',
    'camera',
])


def get_metadata_from_batch_ground_truth(batch_data, num_frames):
    """Parse a batch of metadata.

    Args:
        batch_data: A list of dict of lists containing the ground truth and possible metadata.
        num_frames (int): Number of frames seen so far. The frame number is set as the frame id.

    Return:
        metadata: Metadata for the current minibatch. Keys are frame number (integer) and values
            are tuple (frame_id, camera_location, image_dimension). Values for camera_location and
            image_dimension are None if not defined, frame_id is frame_idx when not defined.
    """
    if isinstance(batch_data, list):
        missing_keys = VALID_METADATA_KEYS - set(batch_data[0].keys())
        if len(missing_keys) > 0:
            warnings.warn("One or more metadata field(s) are missing from ground_truth batch_data, "
                          "and will be replaced with defaults: %s" % list(missing_keys))

        metadata = []

        for frame_idx, frame_data in enumerate(batch_data, num_frames):
            frame_id = frame_data.get(FRAME_ID_KEY, (frame_idx,))[0]
            camera_location = frame_data.get(CAMERA_LOCATION_KEY, (None,))[0]
            image_dimensions = tuple(frame_data[IMAGE_DIMENSIONS_KEY][0]) \
                if IMAGE_DIMENSIONS_KEY in frame_data else None

            metadata.append((frame_id, camera_location, image_dimensions))
    elif isinstance(batch_data, Bbox2DLabel):
        metadata = [Frame(
            np.squeeze(batch_data.frame_id[i]).flatten()[
                0].decode().replace("/", "_"),
            (batch_data.vertices.canvas_shape.width[i].size,
             batch_data.vertices.canvas_shape.height[i].size),
            None)
            for i in range(batch_data.object_class.dense_shape[0])]
    else:
        raise NotImplementedError("Unhandled batch data of type: {}".format(type(Bbox2DLabel)))

    return metadata
