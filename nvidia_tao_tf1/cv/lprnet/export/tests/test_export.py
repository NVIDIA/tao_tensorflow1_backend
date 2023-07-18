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
"""test lprnet export."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import pytest
import tensorflow as tf
import nvidia_tao_tf1.cv.lprnet.models.model_builder as model_builder
from nvidia_tao_tf1.cv.lprnet.utils.model_io import save_model
from nvidia_tao_tf1.cv.lprnet.utils.spec_loader import load_experiment_spec


backbone_configs = [('baseline', 10, "fp32"),
                    ('baseline', 18, "fp16")]


@pytest.fixture
def _spec_file():
    '''default spec file.'''
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(parent_dir, 'experiment_specs/default_spec.txt')


@pytest.fixture
def spec():
    experiment_spec = load_experiment_spec(merge_from_default=True)
    label = "abcdefg"
    with open("tmp_ch_list.txt", "w") as f:
        for ch in label:
            f.write(ch + "\n")
    experiment_spec.dataset_config.characters_list_file = "tmp_ch_list.txt"
    yield experiment_spec
    os.remove("tmp_ch_list.txt")


@pytest.mark.skipif(os.getenv("RUN_ON_CI", "0") == "1", reason="Cannot be run on CI")
@pytest.mark.script_launch_mode('subprocess')
@pytest.mark.parametrize("model_type, nlayers, data_type",
                         backbone_configs)
def test_export(script_runner, spec, _spec_file, model_type,
                nlayers, data_type):
    spec.lpr_config.arch = model_type
    spec.lpr_config.nlayers = nlayers
    # pin GPU ID 0 so it uses the newest GPU ARCH for INT8
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    enc_key = 'nvidia_tlt'
    tf.keras.backend.clear_session()

    model, _, _ = model_builder.build(spec)

    os_handle, tmp_keras_model = tempfile.mkstemp(suffix=".hdf5")
    os.close(os_handle)
    save_model(model, tmp_keras_model, enc_key.encode())
    os_handle, tmp_exported_model = tempfile.mkstemp(suffix=".onnx")
    os.close(os_handle)
    os.remove(tmp_exported_model)
    tf.reset_default_graph()
    del model
    # export to etlt model
    script = 'nvidia_tao_tf1/cv/lprnet/scripts/export.py'
    env = os.environ.copy()
    # 1. export in FP32 mode
    if data_type == "fp32":
        args = ['-m', tmp_keras_model,
                '-k', enc_key,
                '--experiment_spec', _spec_file,
                '-o', tmp_exported_model]
        ret = script_runner.run(script, env=env, *args)
        # before abort, remove the created temp files when exception raises
        try:
            assert ret.success
            assert os.path.isfile(tmp_exported_model)
            if os.path.exists(tmp_exported_model):
                os.remove(tmp_exported_model)
        except AssertionError:
            # if the script runner failed, the tmp_exported_model may not be created at all
            if os.path.exists(tmp_exported_model):
                os.remove(tmp_exported_model)
            os.remove(tmp_keras_model)
            raise(AssertionError(ret.stdout + ret.stderr))

    # 2. export in FP16 mode
    if data_type == "fp16":
        args = ['-m', tmp_keras_model,
                '-k', enc_key,
                '--experiment_spec', _spec_file,
                '-o', tmp_exported_model,
                '--data_type', 'fp16']
        ret = script_runner.run(script, env=env, *args)
        try:
            assert ret.success
            assert os.path.isfile(tmp_exported_model)
            if os.path.exists(tmp_exported_model):
                os.remove(tmp_exported_model)
        except AssertionError:
            if os.path.exists(tmp_exported_model):
                os.remove(tmp_exported_model)
            os.remove(tmp_keras_model)
            raise(AssertionError(ret.stdout + ret.stderr))

    # clear the tmp files
    if os.path.exists(tmp_exported_model):
        os.remove(tmp_exported_model)
    if os.path.exists(tmp_keras_model):
        os.remove(tmp_keras_model)
