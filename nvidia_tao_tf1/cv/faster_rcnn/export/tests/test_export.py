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
'''Unit test for FasterRCNN model export functionality.'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import keras
from keras.layers import Input
import pytest
import tensorflow as tf

from nvidia_tao_tf1.cv.common.utils import encode_from_keras
from nvidia_tao_tf1.cv.faster_rcnn.models.darknets import DarkNet
from nvidia_tao_tf1.cv.faster_rcnn.models.googlenet import GoogleNet
from nvidia_tao_tf1.cv.faster_rcnn.models.iva_vgg import IVAVGG
from nvidia_tao_tf1.cv.faster_rcnn.models.mobilenet_v1 import MobileNetV1
from nvidia_tao_tf1.cv.faster_rcnn.models.mobilenet_v2 import MobileNetV2
from nvidia_tao_tf1.cv.faster_rcnn.models.resnets import ResNet
from nvidia_tao_tf1.cv.faster_rcnn.models.vgg16 import VGG16
from nvidia_tao_tf1.cv.faster_rcnn.spec_loader import spec_loader, spec_wrapper


backbone_configs_all = [
    (ResNet, 10, False, False),
    (ResNet, 10, True, True),
    (ResNet, 10, False, True),
    (ResNet, 18, True, False),
    (ResNet, 18, False, False),
    (ResNet, 18, True, True),
    (ResNet, 18, False, True),
    (ResNet, 34, True, False),
    (ResNet, 34, False, False),
    (ResNet, 34, True, True),
    (ResNet, 34, False, True),
    (ResNet, 50, True, False),
    (ResNet, 50, False, False),
    (ResNet, 50, True, True),
    (ResNet, 50, False, True),
    (ResNet, 101, True, False),
    (ResNet, 101, False, False),
    (ResNet, 101, True, True),
    (ResNet, 101, False, True),
    (VGG16, None, False, True),
    (IVAVGG, 16, False, True),
    (IVAVGG, 19, False, True),
    (GoogleNet, None, False, True),
    (MobileNetV1, None, False, False),
    (MobileNetV2, None, False, False),
    (MobileNetV2, None, True, False),
    (DarkNet, 19, None, None),
    (DarkNet, 53, None, None),
]

backbone_configs_subset = [
    (ResNet, 18, True, False),
    (ResNet, 50, True, False),
]

if int(os.getenv("TLT_TF_CI_TEST_LEVEL", "0")) > 0:
    backbone_configs = backbone_configs_all
else:
    backbone_configs = backbone_configs_subset

keras.backend.set_image_data_format('channels_first')
output_nodes = ['NMS']


@pytest.fixture()
def _spec_file():
    '''default spec file.'''
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(parent_dir, 'experiment_spec/default_spec_ci.txt')


@pytest.fixture()
def spec(_spec_file):
    '''spec.'''
    return spec_wrapper.ExperimentSpec(spec_loader.load_experiment_spec(_spec_file))


@pytest.mark.skipif(os.getenv("RUN_ON_CI", "0") == "1", reason="Cannot be run on CI")
@pytest.mark.script_launch_mode('subprocess')
@pytest.mark.parametrize("model_type, nlayers, all_projections, use_pooling",
                         backbone_configs)
def test_export(script_runner, tmpdir, spec, _spec_file, model_type,
                nlayers, all_projections, use_pooling):
    '''test export with FP32/FP16.'''
    keras.backend.clear_session()
    gpu_options = tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=0.33,
        allow_growth=True
    )
    device_count = {'GPU': 0, 'CPU': 1}
    session_config = tf.compat.v1.ConfigProto(
        gpu_options=gpu_options,
        device_count=device_count
    )
    session = tf.compat.v1.Session(config=session_config)
    keras.backend.set_session(session)
    model = model_type(nlayers, spec.batch_size_per_gpu,
                       spec.rpn_stride, spec.reg_type,
                       spec.weight_decay, spec.freeze_bn, spec.freeze_blocks,
                       spec.dropout_rate, spec.drop_connect_rate,
                       spec.conv_bn_share_bias, all_projections,
                       use_pooling, spec.anchor_sizes, spec.anchor_ratios,
                       spec.roi_pool_size, spec.roi_pool_2x, spec.num_classes,
                       spec.std_scaling, spec.rpn_pre_nms_top_N, spec.rpn_post_nms_top_N,
                       spec.rpn_nms_iou_thres, spec.gt_as_roi,
                       spec.rcnn_min_overlap, spec.rcnn_max_overlap, spec.rcnn_train_bs,
                       spec.rcnn_regr_std, spec.rpn_train_bs, spec.lambda_rpn_class,
                       spec.lambda_rpn_regr, spec.lambda_cls_class, spec.lambda_cls_regr,
                       f"frcnn_{spec._backbone.replace(':', '_')}", tmpdir,
                       spec.enc_key, spec.lr_scheduler)
    img_input = Input(shape=spec.input_dims, name='input_image')
    gt_cls_input = Input(shape=(None,), name='input_gt_cls')
    gt_bbox_input = Input(shape=(None, 4), name='input_gt_bbox')
    model.build_keras_model(img_input, gt_cls_input, gt_bbox_input)
    os_handle, tmp_keras_model = tempfile.mkstemp()
    os.close(os_handle)
    encode_from_keras(model.keras_model, tmp_keras_model, spec.enc_key.encode())
    os_handle, tmp_onnx_model = tempfile.mkstemp(suffix=".onnx")
    os.close(os_handle)
    os.remove(tmp_onnx_model)
    # export to etlt model
    script = 'nvidia_tao_tf1/cv/faster_rcnn/scripts/export.py'
    env = os.environ.copy()
    # 1. export in FP32 mode
    args = ['-m', tmp_keras_model,
            '-k', spec.enc_key,
            '--experiment_spec', _spec_file,
            '-o', tmp_onnx_model]
    keras.backend.clear_session()
    ret = script_runner.run(script, env=env, *args)
    # before abort, remove the created temp files when exception raises
    try:
        assert ret.success
        assert os.path.isfile(tmp_onnx_model)
        if os.path.exists(tmp_onnx_model):
            os.remove(tmp_onnx_model)
    except AssertionError:
        # if the script runner failed, the tmp_onnx_model may not be created at all
        if os.path.exists(tmp_onnx_model):
            os.remove(tmp_onnx_model)
        os.remove(tmp_keras_model)
        raise(AssertionError(ret.stdout + ret.stderr))
    # 2. export in FP16 mode
    args = ['-m', tmp_keras_model,
            '-k', spec.enc_key,
            '--experiment_spec', _spec_file,
            '-o', tmp_onnx_model,
            '--data_type', 'fp16']
    keras.backend.clear_session()
    ret = script_runner.run(script, env=env, *args)
    try:
        assert ret.success
        assert os.path.isfile(tmp_onnx_model)
        if os.path.exists(tmp_onnx_model):
            os.remove(tmp_onnx_model)
    except AssertionError:
        if os.path.exists(tmp_onnx_model):
            os.remove(tmp_onnx_model)
        os.remove(tmp_keras_model)
        raise(AssertionError(ret.stdout + ret.stderr))
    finally:
        if os.path.exists(tmp_onnx_model):
            os.remove(tmp_onnx_model)
        if os.path.exists(tmp_keras_model):
            os.remove(tmp_keras_model)
