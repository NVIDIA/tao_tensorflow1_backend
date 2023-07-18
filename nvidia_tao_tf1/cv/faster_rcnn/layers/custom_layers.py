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
"""FasterRCNN custom keras layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Group all the custom layers here so that we can import them
# from here with a single import statement
from nvidia_tao_tf1.cv.faster_rcnn.layers.CropAndResize import CropAndResize
from nvidia_tao_tf1.cv.faster_rcnn.layers.NmsInputs import NmsInputs
from nvidia_tao_tf1.cv.faster_rcnn.layers.OutputParser import OutputParser
from nvidia_tao_tf1.cv.faster_rcnn.layers.Proposal import Proposal
from nvidia_tao_tf1.cv.faster_rcnn.layers.ProposalTarget import ProposalTarget
from nvidia_tao_tf1.cv.faster_rcnn.layers.TFReshape import TFReshape


__all__ = (
    'CropAndResize',
    'NmsInputs',
    'OutputParser',
    'Proposal',
    'ProposalTarget',
    'TFReshape',
)
