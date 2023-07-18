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

"""Generate square bounding boxes for face and eyes."""

import abc
from six import add_metaclass


@add_metaclass(abc.ABCMeta)
class BoundingBoxStrategy(object):
    """Abstract class for creating square bounding boxes."""

    @abc.abstractmethod
    def get_square_bbox(self):
        """Abstract method for retrieving square bounding box."""
        pass


class FaceBboxStrategy(BoundingBoxStrategy):
    """Generate face bounding box."""

    def __init__(self, frame_width, frame_height, x1, y1, w, h, scale_factor=1.3):
        """Initialize frame_width, frame_height, w, h, top left coordinate, and scale factor."""
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._x1 = x1
        self._y1 = y1
        self._w = w
        self._h = h
        self._scale_factor = scale_factor

    def get_square_bbox(self):
        """Get square face bounding box."""
        center = [self._x1 + self._w/2, self._y1 + self._h/2]
        w = self._w * self._scale_factor
        h = self._h * self._scale_factor
        side_len = max(w, h)

        upper = int(center[1] - side_len/2)
        lower = int(center[1] + side_len/2)

        left = int(center[0] - side_len/2)
        right = int(center[0] + side_len/2)

        if left < 0:
            dx = 0 - left
            left = 0
            right += dx
        if right > self._frame_width:
            dx = right - self._frame_width
            right = self._frame_width
            left = left - dx
        if upper < 0:
            dx = 0 - upper
            upper = 0
            lower += dx
        if lower > self._frame_height:
            dx = lower - self._frame_height
            lower = self._frame_height
            upper = upper - dx

        upper = max(upper, 0)
        lower = min(lower, self._frame_height)
        left = max(left, 0)
        right = min(right, self._frame_width)

        side_len = min(right - left, lower - upper)
        lower = upper + side_len
        right = left + side_len

        return list(map(int, [left, upper, side_len]))


class EyeBboxStrategy(BoundingBoxStrategy):
    """Generate eye bounding box."""

    def __init__(self, frame_width, frame_height, coords, scale_factor=1.1):
        """Initialize frame_width, frame_height, coordinates and scale factor."""
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._coords = coords
        self._scale_factor = scale_factor

    def get_square_bbox(self):
        """Get square eye bounding box."""
        x1, y1, x2, y2 = self._coords

        w = self._scale_factor * (x2 - x1)
        h = self._scale_factor * (y2 - y1)
        side_len = max(w, h)

        center = [0.5*(x1 + x2), 0.5*(y1 + y2)]
        upper = center[1] - side_len/2
        lower = center[1] + side_len/2
        left = center[0] - side_len/2
        right = center[0] + side_len/2

        upper = max(upper, 0)
        lower = min(lower, self._frame_height)
        left = max(left, 0)
        right = min(right, self._frame_width)

        side_len = int(min(right - left, lower - upper))
        lower = int(upper) + side_len
        right = int(left) + side_len

        if lower - upper <= 0 and right - left <= 0:
            return [-1, -1, -1, -1]

        return list(map(int, [left, upper, right - left, lower - upper]))
