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
"""Tests for FpeNet Inferencer Class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import keras.backend as K
import numpy as np
import pytest
import tensorflow as tf
import yaml

from nvidia_tao_tf1.cv.fpenet.inferencer.fpenet_inferencer import FpeNetInferencer

@pytest.mark.skipif(
    os.getenv("RUN_ON_CI", "0") == "1",
    reason="Temporarily skipping on the CI."
)
def test_inferencer(tmpdir):
    """
    Test the inferencer.

    Args:
        tmpdir (str): Temporary path for checkpoint directory.
    Returns:
        None
    """
    K.clear_session()
    key = '0'
    with open('nvidia_tao_tf1/cv/fpenet/experiment_specs/default.yaml', 'r') as yaml_file:
        spec = yaml.load(yaml_file.read())

    inferencer = FpeNetInferencer(
        experiment_spec=spec,
        data_path='nvidia_tao_tf1/cv/fpenet/dataloader/testdata/datafactory.json',
        output_folder=str(tmpdir),
        model_path=str(tmpdir)+'/temp_model.tlt',
        image_root_path='',
        key=key)
    inferencer.is_trt_model = False

    # save temporary model for testing purpose
    inferencer._model.build(
        input_images=tf.convert_to_tensor(np.ones((1, 1, 80, 80)).astype(np.float32)))
    inferencer._model.save_model(inferencer._model_path, key)
    # run inference
    inferencer.infer_model()
    # save inference results
    inferencer.save_results()

    assert os.path.exists(str(tmpdir)+'/result.txt')
