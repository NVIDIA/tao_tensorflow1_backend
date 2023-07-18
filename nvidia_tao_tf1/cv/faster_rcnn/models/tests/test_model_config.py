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
"""Unit test for FasterRCNN model parameter configurations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

# Forcing this test to use GPU 0. For some reason
# after QAT patch, tensorflow seems to be looking for
# GPU id 1.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras.backend as K
from keras.layers import Input
import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.cv.common.utils import hvd_keras
from nvidia_tao_tf1.cv.detectnet_v2.proto.regularizer_config_pb2 import RegularizerConfig
from nvidia_tao_tf1.cv.faster_rcnn.models.utils import (
    build_inference_model,
    select_model_type,
)
from nvidia_tao_tf1.cv.faster_rcnn.spec_loader import spec_loader, spec_wrapper
from nvidia_tao_tf1.cv.faster_rcnn.utils import utils


np.random.seed(42)
tf.set_random_seed(42)
logger = logging.getLogger(__name__)
hvd = None


class TestModelConfig(object):
    '''Main class to test model parameter configurations.'''
    def _spec_file(self):
        '''default spec file.'''
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        return os.path.join(parent_dir, 'experiment_spec/default_spec_ci.txt')

    def _spec(self):
        '''spec.'''
        return spec_wrapper.ExperimentSpec(spec_loader.load_experiment_spec(self._spec_file()))

    def config(self):
        '''Configuration.'''
        K.clear_session()
        global hvd  # noqa pylint: disable=W0603
        hvd = hvd_keras()
        hvd.init()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=config))
        K.set_image_data_format('channels_first')
        self.session = K.get_session()
        self.spec = self._spec()

    def build_model(self, results_dir, config_override=None):
        '''Build a model from spec with possibly outside config for overriding.'''
        spec = self.spec
        if config_override is None:
            config_override = dict()
        for k in config_override.keys():
            assert hasattr(spec, k)
            setattr(spec, k, config_override[k])
        self.override_spec = spec
        img_input = Input(shape=spec.input_dims, name='input_image')
        gt_cls_input = Input(shape=(None,), name='input_gt_cls')
        gt_bbox_input = Input(shape=(None, 4), name='input_gt_bbox')
        # build the model
        model_type = select_model_type(spec._backbone)
        self.model = model_type(
            spec.nlayers, spec.batch_size_per_gpu,
            spec.rpn_stride, spec.reg_type,
            spec.weight_decay, spec.freeze_bn, spec.freeze_blocks,
            spec.dropout_rate, spec.drop_connect_rate,
            spec.conv_bn_share_bias, spec.all_projections,
            spec.use_pooling, spec.anchor_sizes, spec.anchor_ratios,
            spec.roi_pool_size, spec.roi_pool_2x, spec.num_classes,
            spec.std_scaling, spec.rpn_pre_nms_top_N, spec.rpn_post_nms_top_N,
            spec.rpn_nms_iou_thres, spec.gt_as_roi,
            spec.rcnn_min_overlap, spec.rcnn_max_overlap, spec.rcnn_train_bs,
            spec.rcnn_regr_std, spec.rpn_train_bs, spec.lambda_rpn_class,
            spec.lambda_rpn_regr, spec.lambda_cls_class, spec.lambda_cls_regr,
            "frcnn_"+spec._backbone.replace(":", "_"), results_dir,
            spec.enc_key, spec.lr_scheduler,
            spec.enable_qat,
            activation_type=spec.activation_type
        )
        self.model.build_keras_model(img_input, gt_cls_input, gt_bbox_input)
        return self.model

    def test_input_shape(self, tmpdir):
        '''Check the model input shape is the same as in spec file.'''
        self.config()
        K.set_learning_phase(1)
        self.build_model(tmpdir)
        input_image_shape = self.model.keras_model.get_layer('input_image').output_shape
        assert input_image_shape == (None,
                                     self.override_spec.image_c,
                                     self.override_spec.image_h,
                                     self.override_spec.image_w)

        input_gt_bbox_shape = self.model.keras_model.get_layer('input_gt_bbox').output_shape
        assert input_gt_bbox_shape == (None,
                                       None,
                                       4)
        input_cls_shape = self.model.keras_model.get_layer('input_gt_cls').output_shape
        assert input_cls_shape == (None,
                                   None)

    def test_anchor_boxes(self, tmpdir):
        '''Check the anchor boxes.'''
        self.config()
        K.set_learning_phase(1)
        self.build_model(tmpdir)
        # check anchors for Proposal layer
        proposal = None
        for l in self.model.keras_model.layers:
            if l.name.startswith('proposal') and not l.name.startswith('proposal_target'):
                proposal = l
                break
        assert proposal is not None
        assert proposal.anchor_sizes == self.override_spec.anchor_sizes
        ar = [np.sqrt(r) for r in self.override_spec.anchor_ratios]
        assert proposal.anchor_ratios == ar

    def test_rpn_nms_params(self, tmpdir):
        '''Check the RPN NMS parameters.'''
        self.config()
        K.set_learning_phase(1)
        self.build_model(tmpdir)
        K.get_session().run(utils.get_init_ops())
        proposal = None
        for l in self.model.keras_model.layers:
            if l.name.startswith('proposal_') and not l.name.startswith('proposal_target_'):
                proposal = l
                break
        assert proposal is not None
        proposal.pre_nms_top_N == self.override_spec.rpn_pre_nms_top_N
        proposal.post_nms_top_N == self.override_spec.rpn_post_nms_top_N
        proposal.nms_iou_thres == self.override_spec.rpn_nms_iou_thres

    def test_rcnn_iou_thres(self, tmpdir):
        '''Check the RCNN IoU thresholds used to generate the RCNN target tensors.'''
        self.config()
        K.set_learning_phase(1)
        self.build_model(tmpdir)
        K.get_session().run(utils.get_init_ops())
        pt_layer = None
        for l in self.model.keras_model.layers:
            if l.name.startswith('proposal_target_'):
                pt_layer = l
                break
        assert pt_layer is not None
        assert pt_layer.iou_high_thres == self.override_spec.rcnn_max_overlap
        assert pt_layer.iou_low_thres == self.override_spec.rcnn_min_overlap

    def test_regularizers(self, tmpdir):
        '''Check the regularizers.'''
        self.config()
        K.set_learning_phase(1)
        model = self.build_model(tmpdir)
        K.get_session().run(utils.get_init_ops())
        mconfig = model.keras_model.get_config()
        reg_type = self.override_spec.reg_type
        if reg_type == RegularizerConfig.L1:
            reg_type = 'l1'
        elif reg_type == RegularizerConfig.L2:
            reg_type = 'l2'
        else:
            ValueError(
                "Should use either L1 or L2 regularizer for test_regularizers."
                " Got {}".format(reg_type)
            )
        # Obtain type and scope of the regularizer
        for layer, layer_config in zip(model.keras_model.layers, mconfig['layers']):
            # Regularizer settings
            if hasattr(layer, 'kernel_regularizer'):
                assert layer_config['config']['kernel_regularizer']['config'][reg_type] == \
                    self.override_spec.weight_decay

    def test_inference_model_config(self, tmpdir):
        '''Check the model config for inference model.'''
        self.config()
        K.set_learning_phase(1)
        model = self.build_model(tmpdir)
        K.get_session().run(utils.get_init_ops())
        train_model = model.keras_model
        config_override = {'pre_nms_top_N': self.override_spec.infer_rpn_pre_nms_top_N,
                           'post_nms_top_N': self.override_spec.infer_rpn_post_nms_top_N,
                           'nms_iou_thres': self.override_spec.infer_rpn_nms_iou_thres,
                           'bs_per_gpu': 1}
        infer_model = build_inference_model(train_model, config_override)
        proposal = None
        for l in infer_model.layers:
            if l.name.startswith('proposal_') and not l.name.startswith('proposal_target_'):
                proposal = l
                break
        assert proposal is not None
        assert proposal.pre_nms_top_N == self.override_spec.infer_rpn_pre_nms_top_N
        assert proposal.post_nms_top_N == self.override_spec.infer_rpn_post_nms_top_N
        assert proposal.nms_iou_thres == self.override_spec.infer_rpn_nms_iou_thres

    def test_eval_model_config(self, tmpdir):
        '''Check the model config for evaluation model.'''
        self.config()
        K.set_learning_phase(1)
        model = self.build_model(tmpdir)
        K.get_session().run(utils.get_init_ops())
        train_model = model.keras_model
        config_override = {'pre_nms_top_N': self.override_spec.eval_rpn_pre_nms_top_N,
                           'post_nms_top_N': self.override_spec.eval_rpn_post_nms_top_N,
                           'nms_iou_thres': self.override_spec.eval_rpn_nms_iou_thres,
                           'bs_per_gpu': 1}
        eval_model = build_inference_model(train_model, config_override)
        proposal = None
        for l in eval_model.layers:
            if l.name.startswith('proposal_') and not l.name.startswith('proposal_target_'):
                proposal = l
                break
        assert proposal is not None
        assert proposal.pre_nms_top_N == self.override_spec.eval_rpn_pre_nms_top_N
        assert proposal.post_nms_top_N == self.override_spec.eval_rpn_post_nms_top_N
        assert proposal.nms_iou_thres == self.override_spec.eval_rpn_nms_iou_thres
