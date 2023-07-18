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
"""test ssd arch builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.cv.ssd.architecture.ssd_arch import ssd
import nvidia_tao_tf1.cv.ssd.models.patch_keras
nvidia_tao_tf1.cv.ssd.models.patch_keras.patch()


def test_arch():
    model = ssd((3, 300, 300),
                3, True,
                kernel_regularizer=None,
                freeze_blocks=[0],
                freeze_bn=None,
                min_scale=0.1,
                max_scale=0.8,
                scales=None,
                aspect_ratios_global=[1, 0.5, 2],
                aspect_ratios_per_layer=None,
                two_boxes_for_ar1=False,
                steps=None,
                offsets=None,
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                confidence_thresh=0.01,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400,
                arch="resnet",
                nlayers=10,
                input_tensor=None,
                qat=False)
    assert model.get_layer('conv1').trainable is False
    assert model.get_layer('ssd_predictions').output_shape[-2:] == (5829, 15)
    model = ssd((3, 300, 300),
                3, False,
                kernel_regularizer=None,
                freeze_blocks=[0],
                freeze_bn=None,
                min_scale=0.1,
                max_scale=0.8,
                scales=None,
                aspect_ratios_global=[1, 0.5, 2],
                aspect_ratios_per_layer=None,
                two_boxes_for_ar1=False,
                steps=None,
                offsets=None,
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                confidence_thresh=0.01,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400,
                arch="resnet",
                nlayers=10,
                input_tensor=None,
                qat=True)
    assert model.get_layer('conv1').trainable is False
    assert model.get_layer('ssd_predictions').output_shape[-2:] == (5829, 15)
