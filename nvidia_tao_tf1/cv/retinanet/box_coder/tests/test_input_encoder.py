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
"""test input encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from nvidia_tao_tf1.cv.retinanet.box_coder.input_encoder import InputEncoder


def test_input_encoder():
    encoder = InputEncoder(img_height=300,
                           img_width=300,
                           n_classes=3,
                           predictor_sizes=[(1, 2), (1, 2)],
                           scales=[0.1, 0.88, 1.05],
                           aspect_ratios_per_layer=[[1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                    [1.0, 2.0, 0.5]],
                           two_boxes_for_ar1=True,
                           steps=None,
                           offsets=None,
                           clip_boxes=False,
                           variances=[0.1, 0.1, 0.2, 0.2],
                           pos_iou_threshold=0.5,
                           neg_iou_limit=0.5,
                           normalize_coords=True)

    gt = np.array([[0, 10, 10, 100, 100], [1, 2, 3, 6, 8]])
    assert encoder(gt).shape == (60, 16)


def test_input_encoder_multimatch():
    encoder = InputEncoder(img_height=300,
                           img_width=300,
                           n_classes=3,
                           predictor_sizes=[(1, 10), (1, 10)],
                           scales=[0.1, 0.88, 1.05],
                           aspect_ratios_per_layer=[[2.0], [2.0]],
                           two_boxes_for_ar1=True,
                           steps=None,
                           offsets=None,
                           clip_boxes=False,
                           variances=[0.1, 0.1, 0.2, 0.2],
                           pos_iou_threshold=0.01,
                           neg_iou_limit=0.01,
                           normalize_coords=True)

    raw_gt = '''np.array(
      [[1, 0, 139, 36, 171],
       [1, 23, 139, 66, 171],
       [0, 50, 139, 306, 171]])'''

    gt = eval(raw_gt)
    assert encoder(gt).shape == (60, 16)
