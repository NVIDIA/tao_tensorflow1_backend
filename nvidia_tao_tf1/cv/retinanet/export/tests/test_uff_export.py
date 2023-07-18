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
'''Unit test for RetinaNet model export functionality.'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import keras
import pytest
import tensorflow as tf

from nvidia_tao_tf1.cv.common.utils import encode_from_keras
from nvidia_tao_tf1.cv.retinanet.architecture.retinanet import retinanet
from nvidia_tao_tf1.cv.retinanet.utils.helper import eval_str
from nvidia_tao_tf1.cv.retinanet.utils.spec_loader import load_experiment_spec


backbone_configs = [
    ('resnet', 10, False, "fp32"),
    ('resnet', 50, True, "int8"),
    ('resnet', 18, False, "int8")]
keras.backend.set_image_data_format('channels_first')


@pytest.fixture
def _spec_file():
    '''Default spec path.'''
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(parent_dir, 'experiment_specs/default_spec.txt')


@pytest.fixture
def spec():
    '''Default spec.'''
    experiment_spec = load_experiment_spec(merge_from_default=True)
    return experiment_spec


@pytest.mark.skipif(os.getenv("RUN_ON_CI", "0") == "1", reason="Cannot be run on CI")
@pytest.mark.script_launch_mode('subprocess')
@pytest.mark.parametrize("model_type, nlayers, qat, data_type",
                         backbone_configs)
def test_export_uff(script_runner, spec, _spec_file, model_type,
                    nlayers, qat, data_type):
    '''test to make sure the export works and uff model can be parsed without issues.'''
    # pin GPU ID 0 so it uses the newest GPU ARCH for INT8
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    enc_key = 'nvidia_tlt'
    cls_mapping = spec.dataset_config.target_class_mapping
    classes = sorted({str(x) for x in cls_mapping.values()})
    # n_classes + 1 for background class
    n_classes = len(classes) + 1
    scales = eval_str(spec.retinanet_config.scales)
    aspect_ratios_global = eval_str(
        spec.retinanet_config.aspect_ratios_global)
    aspect_ratios_per_layer = eval_str(
        spec.retinanet_config.aspect_ratios)
    steps = eval_str(spec.retinanet_config.steps)
    offsets = eval_str(spec.retinanet_config.offsets)
    variances = eval_str(spec.retinanet_config.variances)
    freeze_blocks = eval_str(spec.retinanet_config.freeze_blocks)
    freeze_bn = eval_str(spec.retinanet_config.freeze_bn)
    keras.backend.clear_session()
    model = retinanet(
        image_size=(3, 608, 608),
        n_classes=n_classes,
        nlayers=nlayers,
        kernel_regularizer=None,
        freeze_blocks=freeze_blocks,
        freeze_bn=freeze_bn,
        scales=scales,
        min_scale=spec.retinanet_config.min_scale,
        max_scale=spec.retinanet_config.max_scale,
        aspect_ratios_global=aspect_ratios_global,
        aspect_ratios_per_layer=aspect_ratios_per_layer,
        two_boxes_for_ar1=spec.retinanet_config.two_boxes_for_ar1,
        steps=steps,
        offsets=offsets,
        clip_boxes=spec.retinanet_config.clip_boxes,
        variances=variances,
        arch=model_type,
        input_tensor=None,
        qat=qat)

    os_handle, tmp_keras_model = tempfile.mkstemp(suffix=".tlt")
    os.close(os_handle)
    encode_from_keras(model, tmp_keras_model, enc_key.encode())
    os_handle, tmp_exported_model = tempfile.mkstemp(suffix=".onnx")
    os.close(os_handle)
    os.remove(tmp_exported_model)
    del model
    # export to etlt model
    script = 'nvidia_tao_tf1/cv/retinanet/scripts/export.py'
    env = os.environ.copy()
    # 1. export in FP32 mode
    if data_type == "fp32":
        args = ['-m', tmp_keras_model,
                '-k', enc_key,
                '--experiment_spec', _spec_file,
                '-o', tmp_exported_model,
                '--static_batch_size', "1"]
        keras.backend.clear_session()
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
                '--data_type', 'fp16',
                '--static_batch_size', "1"]
        keras.backend.clear_session()
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
    # 3. export in INT8 mode with random data for calibration
    # 4. export in INT8 mode with tensor_scale_dict
    os_handle, tmp_data_file = tempfile.mkstemp()
    os.close(os_handle)
    os.remove(tmp_data_file)
    os_handle, tmp_cache_file = tempfile.mkstemp()
    os.close(os_handle)
    os.remove(tmp_cache_file)
    if data_type == "int8":
        if qat:
            args = ['-m', tmp_keras_model,
                    '-k', enc_key,
                    '--experiment_spec', _spec_file,
                    '-o', tmp_exported_model,
                    '--data_type', 'int8',
                    '--cal_cache_file', tmp_cache_file,
                    '--static_batch_size', "1"]
            keras.backend.clear_session()
            ret = script_runner.run(script, env=env, *args)
            try:
                assert ret.success
                assert os.path.isfile(tmp_exported_model)
                assert os.path.isfile(tmp_cache_file)
            except AssertionError:
                raise AssertionError(ret.stdout + ret.stderr)
        else:
            args = ['-m', tmp_keras_model,
                    '-k', enc_key,
                    '--experiment_spec', _spec_file,
                    '-o', tmp_exported_model,
                    '--data_type', 'int8',
                    '--cal_data_file', tmp_data_file,
                    '--cal_image_dir', "",
                    '--batches', '1',
                    '--batch_size', '1',
                    '--cal_cache_file', tmp_cache_file,
                    '--static_batch_size', "1"]
            keras.backend.clear_session()
            ret = script_runner.run(script, env=env, *args)
            try:
                # this is the last export, retain the etlt model for following check
                assert ret.success
                assert os.path.isfile(tmp_exported_model)
                if os.path.exists(tmp_data_file):
                    os.remove(tmp_data_file)
                if os.path.exists(tmp_cache_file):
                    os.remove(tmp_cache_file)
            except AssertionError:
                raise(AssertionError(ret.stdout + ret.stderr))

    # clear the tmp files
    if os.path.exists(tmp_exported_model):
        os.remove(tmp_exported_model)
    if os.path.exists(tmp_cache_file):
        os.remove(tmp_cache_file)
    if os.path.exists(tmp_keras_model):
        os.remove(tmp_keras_model)
