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
"""EfficientDet arch tests."""
import os
import pytest
import tensorflow as tf

from nvidia_tao_tf1.cv.efficientdet.models import efficientdet_arch
from nvidia_tao_tf1.cv.efficientdet.utils import hparams_config, spec_loader


@pytest.mark.parametrize("file_name",
                         [('d1.txt'),
                          ('d2.txt'),
                          ('d3.txt'),
                          ('d4.txt'),
                          ('d5.txt')])
def test_arch(tmpdir, file_name):
    file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_spec_path = os.path.join(file_path, '../experiment_specs', file_name)
    spec = spec_loader.load_experiment_spec(default_spec_path, merge_from_default=False)
    MODE = 'train'
    # Parse and override hparams
    config = hparams_config.get_detection_config(spec.model_config.model_name)
    params = spec_loader.generate_params_from_spec(config, spec, MODE)
    config.update(params)
    config.model_dir = tmpdir

    inputs = tf.keras.layers.Input(shape=(512, 512, 3), batch_size=4)
    class_output, box_outputs = efficientdet_arch.efficientdet(inputs, None, config=config)

    assert len(class_output) == config.max_level - config.min_level + 1
    assert len(box_outputs) == config.max_level - config.min_level + 1
