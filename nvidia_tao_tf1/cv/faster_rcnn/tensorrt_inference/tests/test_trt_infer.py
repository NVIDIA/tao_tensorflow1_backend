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
'''Unit test for FasterRCNN model TensorRT inference functionality.'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import tempfile

import keras
from keras.layers import Input
import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.cv.common.utils import encode_from_keras
from nvidia_tao_tf1.cv.faster_rcnn.models.resnets import ResNet
from nvidia_tao_tf1.cv.faster_rcnn.models.utils import build_inference_model
from nvidia_tao_tf1.cv.faster_rcnn.spec_loader import spec_loader, spec_wrapper
from nvidia_tao_tf1.cv.faster_rcnn.tensorrt_inference.tensorrt_model import TrtModel

backbone_configs = [
    (ResNet, 10, False, False),
]

keras.backend.set_image_data_format('channels_first')


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
def test_trt_inference(script_runner, tmpdir, spec, _spec_file,
                       model_type, nlayers, all_projections,
                       use_pooling):
    '''test to make sure the trt inference works well.'''
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
    config_override = {
        'pre_nms_top_N': spec.infer_rpn_pre_nms_top_N,
        'post_nms_top_N': spec.infer_rpn_post_nms_top_N,
        'nms_iou_thres': spec.infer_rpn_nms_iou_thres,
        'bs_per_gpu': spec.infer_batch_size
    }
    os_handle, tmp_keras_model = tempfile.mkstemp()
    os.close(os_handle)
    # export will convert train model to infer model, so we still encode train
    # model here
    encode_from_keras(model.keras_model, tmp_keras_model, spec.enc_key.encode())
    os_handle, tmp_onnx_model = tempfile.mkstemp(suffix=".onnx")
    os.close(os_handle)
    os.remove(tmp_onnx_model)
    # export
    script = 'nvidia_tao_tf1/cv/faster_rcnn/scripts/export.py'
    env = os.environ.copy()
    os_handle, tmp_engine_file = tempfile.mkstemp()
    os.close(os_handle)
    os.remove(tmp_engine_file)
    args = ['faster_rcnn',
            '-m', tmp_keras_model,
            '-k', spec.enc_key,
            '--experiment_spec', _spec_file,
            '-o', tmp_onnx_model,
            '--engine_file', tmp_engine_file]
    keras.backend.clear_session()
    ret = script_runner.run(script, env=env, *args)
    try:
        assert ret.success, print(ret.stdout + ret.stderr)
        assert os.path.isfile(tmp_onnx_model)
        assert os.path.isfile(tmp_engine_file)
        if os.path.exists(tmp_keras_model):
            os.remove(tmp_keras_model)
    except AssertionError:
        if os.path.exists(tmp_onnx_model):
            os.remove(tmp_onnx_model)
        if os.path.exists(tmp_engine_file):
            os.remove(tmp_engine_file)
        if os.path.exists(tmp_keras_model):
            os.remove(tmp_keras_model)
        raise(AssertionError(ret.stdout + ret.stderr))
    trt_model = TrtModel(
        tmp_engine_file,
        spec.batch_size_per_gpu,
        spec.image_h,
        spec.image_w
    )
    trt_model.build_or_load_trt_engine()
    # do prediction
    image_shape = (
        spec.batch_size_per_gpu,
        spec.image_c,
        spec.image_h,
        spec.image_w
    )
    # random image
    random_input = np.random.random(size=image_shape) * 255.
    # due to plugins, we cannot compare keras output with tensorrt output
    # instead, we just make sure the inference can work
    trt_model.predict(random_input)
