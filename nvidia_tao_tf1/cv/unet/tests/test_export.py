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
"""Test Export script."""

import json
import os
import tempfile
import pytest
import shutil
from nvidia_tao_tf1.cv.unet.distribution import distribution
from nvidia_tao_tf1.cv.unet.model.build_unet_model import build_model, select_model_proto
from nvidia_tao_tf1.cv.unet.model.utilities import build_target_class_list, \
    get_train_class_mapping, initialize_params
from nvidia_tao_tf1.cv.unet.model.utilities import get_pretrained_ckpt, \
    get_pretrained_model_path, initialize, update_model_params
from nvidia_tao_tf1.cv.unet.spec_handler.spec_loader import load_experiment_spec

data_types = [("fp32"), ("int8")]


@pytest.mark.skipif(os.getenv("RUN_ON_CI", "0") == "1", reason="Cannot be run on CI")
@pytest.mark.script_launch_mode('subprocess')
@pytest.mark.parametrize("data_type",
                         data_types)
def test_export(script_runner, data_type):
    file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    spec_path = os.path.join(file_path, 'experiment_specs/default1.txt')
    experiment_spec = load_experiment_spec(spec_path, merge_from_default=False)
    key = "nvidia-tao"
    # Initialize the environment
    initialize(experiment_spec)
    # Initialize Params
    params = initialize_params(experiment_spec)
    results_dir = tempfile.mkdtemp()
    if distribution.get_distributor().is_master():
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
    target_classes = build_target_class_list(
        experiment_spec.dataset_config.data_class_config)
    target_classes_train_mapping = get_train_class_mapping(target_classes)
    with open(os.path.join(results_dir, 'target_class_id_mapping.json'), 'w') as fp:
        json.dump(target_classes_train_mapping, fp)
    # Build run config
    model_config = select_model_proto(experiment_spec)
    unet_model = build_model(m_config=model_config,
                             target_class_names=target_classes,
                             seed=params["seed"])
    pretrained_model_file = None
    custom_objs = None
    input_model_file_name = get_pretrained_model_path(pretrained_model_file)
    pre_trained_weights, model_json, _ = \
        get_pretrained_ckpt(input_model_file_name, key, custom_objs=custom_objs)
    ckpt_dir = os.path.split(os.path.abspath(pre_trained_weights))[0]
    img_height, img_width, img_channels = experiment_spec.model_config.model_input_height, \
        experiment_spec.model_config.model_input_width, \
        experiment_spec.model_config.model_input_channels
    params = update_model_params(params=params, unet_model=unet_model,
                                 experiment_spec=experiment_spec, key=key,
                                 results_dir=results_dir,
                                 target_classes=target_classes,
                                 model_json=model_json,
                                 custom_objs=custom_objs
                                 )
    unet_model.construct_model(
        input_shape=(img_channels, img_height, img_width),
        pretrained_weights_file=params.pretrained_weights_file,
        enc_key=key, model_json=params.model_json,
        features=None)

    ckzip_file = ckpt_dir
    tmp_onnx_model = os.path.join(file_path, "tmp.onnx")
    if os.path.exists(tmp_onnx_model):
        os.remove(tmp_onnx_model)
    env = os.environ.copy()
    script = "nvidia_tao_tf1/cv/unet/scripts/export.py"
    if data_type == "fp32":
        args = ['-m', ckzip_file,
                '-k', key,
                '--experiment_spec', spec_path,
                '-o', tmp_onnx_model]
        ret = script_runner.run(script, env=env, *args)
        # before abort, remove the created temp files when exception raises
        try:
            assert ret.success
            assert os.path.isfile(tmp_onnx_model)
            if os.path.exists(tmp_onnx_model):
                os.remove(tmp_onnx_model)
        finally:
            # if the script runner failed, the tmp_onnx_model may not be created at all
            if os.path.exists(tmp_onnx_model):
                os.remove(tmp_onnx_model)
            shutil.rmtree(ckzip_file)

    os_handle, tmp_data_file = tempfile.mkstemp()
    os.close(os_handle)
    os.remove(tmp_data_file)
    os_handle, tmp_cache_file = tempfile.mkstemp()
    os.close(os_handle)
    os.remove(tmp_cache_file)

    if data_type == "int8":
        args = ['-m', ckzip_file,
                '-k', key,
                '--experiment_spec', spec_path,
                '-o', tmp_onnx_model,
                '--data_type', 'int8',
                '--cal_cache_file', tmp_cache_file,
                "--cal_data_file", tmp_data_file
                ]
        ret = script_runner.run(script, env=env, *args)
        # before abort, remove the created temp files when exception raises
        try:
            assert ret.success
            assert os.path.isfile(tmp_onnx_model), (
                f"Output model wasn't generated: {tmp_onnx_model}"
            )
        except AssertionError:
            raise AssertionError(ret.stdout + ret.stderr)
        finally:
            # if the script runner failed, the tmp_onnx_model may not be created at all
            if os.path.exists(tmp_onnx_model):
                os.remove(tmp_onnx_model)
            shutil.rmtree(ckzip_file)

    if os.path.exists(ckzip_file):
        shutil.rmtree(ckzip_file)
