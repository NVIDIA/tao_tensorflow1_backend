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
import tensorflow as tf

from nvidia_tao_tf1.cv.fpenet.exporter.fpenet_exporter import FpeNetExporter
from nvidia_tao_tf1.cv.fpenet.models.fpenet_basemodel import FpeNetBaseModel


def test_exporter(tmpdir):
    """
    Test the exporter.

    Args:
        tmpdir (str): Temporary path for checkpoint directory.
    Returns:
        None
    """
    K.clear_session()
    key = 'test'

    model_parameters = {
        'beta': 0.01,
        'dropout_rate': 0.5,
        'freeze_Convlayer': None,
        'pretrained_model_path': None,
        'regularizer_type': 'l2',
        'regularizer_weight': 1.0e-05,
        'type': 'FpeNet_public',
        'visualization_parameters': None,
    }

    model = FpeNetBaseModel(model_parameters)
    model.build(input_images=tf.convert_to_tensor(np.ones((1, 1, 80, 80)).astype(np.float32)))

    # save temporary files for test case purpose
    model.save_model(str(tmpdir)+'/temp_model.hdf5', key)

    # Build exporter instance
    exporter = FpeNetExporter(str(tmpdir)+'/temp_model.hdf5',
                              key,
                              backend='tfonnx',
                              data_type='fp32')

    output_filename = str(tmpdir)+'/temp_model.onnx'
    engine_file_name = str(tmpdir)+'/temp_model.engine'
    # Export the model to etlt file and build the TRT engine.
    exporter.export([1, 80, 80],
                    output_filename,
                    backend='tfonnx',
                    static_batch_size=-1,
                    save_engine=True,
                    engine_file_name=engine_file_name,
                    target_opset=10,
                    validate_trt_engine=True)

    assert os.path.exists(output_filename), (
        f"Output model file doesn't exist at {output_filename}"
    )
    assert os.path.exists(engine_file_name), (
        f"Output engine file doesn't exist at {engine_file_name}"
    )
