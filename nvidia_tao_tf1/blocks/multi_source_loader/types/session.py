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
"""Structure for representing recording session and temporal ordering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


"""
Information related to the recording session from which the example was generated from.

Args:
    uuid (tf.Tensor): Tensor of type tf.string representing the unique identifier of a recording
        session.
    camera_name (tf.Tensor): Tensor of type tf.string representing the name of the camera with which
        the session was recorded.
    frame_number (tf.Tensor): Tensor of type tf.int32 representing the position of a frame within
        a recording session (~= video.) This is equivalent to the timestep within a sequence. The
        higher the frame number the newer the example/frame
"""
Session = collections.namedtuple("Session", ["uuid", "camera_name", "frame_number"])
