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
"""Data loader and processing test cases."""
import pytest
import tensorflow as tf
from nvidia_tao_tf1.cv.efficientdet.backbone import backbone_factory


@pytest.mark.parametrize("model_name",
                         [('efficientdet-d0'),
                          ('efficientdet-d2'),
                          ('efficientdet-d3'),
                          ('efficientdet-d4'),
                          ('efficientdet-d5')])
def test_backbone(model_name):
    builder = backbone_factory.get_model_builder(model_name)
    inputs = tf.keras.layers.Input(shape=(512, 512, 3), batch_size=1)
    fmaps = builder.build_model_base(inputs, model_name)
    assert len(fmaps) == 5
