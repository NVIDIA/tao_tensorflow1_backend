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

"""Script to calculate YOLOv4 anchor config."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import build_command_line_parser as this is needed by entrypoint
from nvidia_tao_tf1.cv.yolo_v3.scripts.kmeans import build_command_line_parser  # noqa pylint: disable=W0611
from nvidia_tao_tf1.cv.yolo_v3.scripts.kmeans import main

if __name__ == "__main__":
    main()
