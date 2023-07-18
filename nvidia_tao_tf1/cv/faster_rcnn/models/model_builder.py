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
'''Base class to implement the FasterRCNN model builder.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging
import os
import shutil
import sys
import tempfile

import keras
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras.layers import Conv2D, Dense, Input, TimeDistributed
from keras.regularizers import l1, l2
from keras.utils.generic_utils import CustomObjectScope
import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.cv.common import utils as iva_utils
from nvidia_tao_tf1.cv.common.callbacks.loggers import TAOStatusLogger
from nvidia_tao_tf1.cv.common.model_parallelism.parallelize_model import find_segment_idx
from nvidia_tao_tf1.cv.common.utils import (
    CUSTOM_OBJS,
    MultiGPULearningRateScheduler,
    StepLRScheduler,
    TensorBoard
)

from nvidia_tao_tf1.cv.detectnet_v2.proto.regularizer_config_pb2 import RegularizerConfig
from nvidia_tao_tf1.cv.faster_rcnn.callbacks.callbacks import ModelSaver, ValidationCallback
from nvidia_tao_tf1.cv.faster_rcnn.layers.custom_layers import (
    CropAndResize, OutputParser, Proposal, ProposalTarget
)
from nvidia_tao_tf1.cv.faster_rcnn.losses import losses
from nvidia_tao_tf1.cv.faster_rcnn.patched_keras import saving
from nvidia_tao_tf1.cv.faster_rcnn.qat._quantized import check_for_quantized_layers
from nvidia_tao_tf1.cv.faster_rcnn.qat.quantize_keras_model import create_quantized_keras_model


# Patch keras.engine.saving so that we can load weights for TimeDistributed layer from
# classification backbones.
saving.patch()


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')
logger = logging.getLogger(__name__)


class FrcnnModel(object):
    """Model builder for FasterRCNN model.

    This is the base class implementing the FasterRCNN model builder. The FasterRCNN model
    includes several major building blocks: backbone, RPN, proposal, proposal_target,
    crop_and_resize, and rcnn. It builds the FasterRCNN model architecture with these building
    blocks and encapsulate it as a keras model. Besides, it also handles the checkpoint saving and
    loading, regularizers updating, overriding custom layer parameters, etc. It is a high level
    abstraction of the FasterRCNN model that covers the whole life time of it: from training to
    inference and test.
    """

    def __init__(self, nlayers, batch_size_per_gpu,
                 rpn_stride, regularizer_type,
                 weight_decay, freeze_bn, freeze_blocks,
                 dropout_rate, drop_connect_rate,
                 conv_bn_share_bias, all_projections,
                 use_pooling, anchor_sizes, anchor_ratios,
                 roi_pool_size, roi_pool_2x, num_classes,
                 std_scaling, rpn_pre_nms_top_N, rpn_post_nms_top_N,
                 rpn_nms_iou_thres, gt_as_roi, rcnn_min_overlap,
                 rcnn_max_overlap, rcnn_train_bs, rcnn_bbox_std,
                 rpn_train_bs, lambda_rpn_class, lambda_rpn_regr,
                 lambda_rcnn_class, lambda_rcnn_regr,
                 backbone, results_dir, enc_key,
                 lr_config, enable_qat=False,
                 activation_type=None,
                 early_stopping=None):
        '''Initialize the FasterRCNN model architecture.

        Args:
            nlayers(int/str): the number of layers in the backbone.
            batch_size_per_gpu(int): the image batch size per GPU.
            rpn_stride(int): the RPN stride relative to input images(16).
            regularizer_type(str): regularizer type in string.
            weight_decay(float): weight decay factor.
            freeze_bn(bool): Whether or not to freeze the BatchNorm layer during training.
                Usually, especially for small batch size, BN layer should be frozen during
                the training.
            freeze_blocks(list): the list of backbone block IDs to freeze during training.
            dropout_rate(float): The dropout rate for Dropout layer.
            drop_connect_rate(float): The drop connect rate in EfficientNet.
            conv_bn_share_bias(bool): whether or not to share bias between conv2d and BN layer.
                If BN layer is frozen during training, then setting this option to False will
                allow the conv2d layers to have bias and can be learnt from training data. In this
                case, set it to False will benifit accuracy.
            all_projections(bool): whether or not to use all_projections for shortcut connections.
                This is useful for ResNets and MobileNet V2.
            use_pooling(bool): use pooling or strided conv2d in the backbone.
            anchor_sizes(list): the list of anchor box sizes, at the input image scale.
            anchor_ratios(list): the list of anchor box ratios.
            roi_pool_size(int): the output feature map spatial size for CropAndResize layer.
            roi_pool_2x(bool): whether or not to double the roi_pool_size and apply a pooling or
                a stride-2 conv2d after CropAndResize.
            num_classes(int): the number of classes in the dataset(including background).
            std_scaling(float): a scaling factor appied to the RPN deltas output.
            rpn_pre_nms_top_N(int): the number of bboxes to retain before doing NMS for RPN.
            rpn_post_nms_top_N(int): the number of bboxes to retain after doing NMS for RPN.
            rpn_nms_iou_thres(float): the IoU threshold used in the NMS for RPN.
            gt_as_roi(bool): whether or not to use the groundtruth boxes as RoIs for training RCNN.
            rcnn_min_overlap(float): the lower IoU threshold below which we regard RoI as negative
                when generating the target tensors for RCNN.
            rcnn_max_overlap(float): thw higher IoU threshold above which we regard RoI as positive
                when generating the target tensors for RCNN.
            rcnn_train_bs(int): RoI batch size per image for training RCNN.
            lambda_rpn_class(float): scaling factor for RPN classification loss.
            lambda_rpn_regr(float): scaling factor for RPN regression loss.
            lambda_rcnn_class(float): scaling factor for RCNN classification loss.
            lambda_rcnn_regr(float): scaling factor for RCNN regression loss.
            backbone(str): backbone chosen.
            results_dir(str): folder to save training checkpoints.
            enc_key(str): the encoding key.
            lr_config(proto): the learning rate scheduler config proto.
            enable_qat(bool): enable the QAT(quantization-aware training) or not.
            activation_type(str): type of activation function. For overriding EfficientNet
                swish to relu.
            early_stopping(proto): Config for early stopping.
        '''

        self.nlayers = nlayers
        self.batch_size_per_gpu = batch_size_per_gpu
        self.rpn_stride = rpn_stride
        if regularizer_type == RegularizerConfig.L1:
            self.regularizer_type = l1
        elif regularizer_type == RegularizerConfig.L2:
            self.regularizer_type = l2
        else:
            self.regularizer_type = None
        self.weight_decay = weight_decay
        if self.regularizer_type is not None:
            self.kernel_reg = self.regularizer_type(self.weight_decay)
        else:
            self.kernel_reg = None
        self.freeze_bn = freeze_bn
        if freeze_blocks is None:
            freeze_blocks = []
        self.freeze_blocks = freeze_blocks
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.conv_bn_share_bias = conv_bn_share_bias
        self.all_projections = all_projections
        self.use_pooling = use_pooling
        self.anchor_sizes = anchor_sizes
        self.anchor_ratios = anchor_ratios
        self.num_anchors = len(self.anchor_sizes) * len(anchor_ratios)
        self.roi_pool_size = roi_pool_size
        self.roi_pool_2x = roi_pool_2x
        self.num_classes = num_classes
        self.std_scaling = std_scaling
        self.rpn_pre_nms_top_N = rpn_pre_nms_top_N
        self.rpn_post_nms_top_N = rpn_post_nms_top_N
        self.rpn_nms_iou_thres = rpn_nms_iou_thres
        self.gt_as_roi = gt_as_roi
        self.rcnn_min_overlap = rcnn_min_overlap
        self.rcnn_max_overlap = rcnn_max_overlap
        self.rcnn_train_bs = rcnn_train_bs
        self.rcnn_bbox_std = rcnn_bbox_std
        self.rpn_train_bs = rpn_train_bs
        self.lambda_rpn_class = lambda_rpn_class
        self.lambda_rpn_regr = lambda_rpn_regr
        self.lambda_rcnn_class = lambda_rcnn_class
        self.lambda_rcnn_regr = lambda_rcnn_regr
        self.output_model = os.path.join(results_dir, f"{backbone}.hdf5")
        self.enc_key = enc_key
        self.lr_config = lr_config
        self.callbacks = []
        self.losses = None
        self.optimizer = None
        self.target_tensors = None
        self.qat = enable_qat
        self.activation_type = activation_type
        self.early_stopping = early_stopping

    @property
    def prop_config(self):
        '''proposal layer config.

        This config is used to override the proposal layers config.
        '''
        config = {'anchor_sizes': self.anchor_sizes,
                  'anchor_ratios': self.anchor_ratios,
                  'std_scaling': self.std_scaling,
                  'rpn_stride': self.rpn_stride,
                  'pre_nms_top_N': self.rpn_pre_nms_top_N,
                  'post_nms_top_N': self.rpn_post_nms_top_N,
                  'nms_iou_thres': self.rpn_nms_iou_thres,
                  'activation_type': 'sigmoid',
                  'bs_per_gpu': self.batch_size_per_gpu}
        return config

    @property
    def propt_config(self):
        '''proposal_target layers config.

        This config is used to override the proposal_target layers config.
        '''
        config = {'gt_as_roi': self.gt_as_roi,
                  'iou_high_thres': self.rcnn_max_overlap,
                  'iou_low_thres': self.rcnn_min_overlap,
                  'roi_train_bs': self.rcnn_train_bs,
                  'roi_positive_ratio': 0.25,
                  'deltas_scaling': self.rcnn_bbox_std,
                  'bg_class_id': self.num_classes-1,
                  'bs_per_gpu': self.batch_size_per_gpu}
        return config

    @property
    def cr_config(self):
        '''CropAndResize layers config.

        This config is used to override the crop_and_resize layer config.
        '''
        if self.roi_pool_2x:
            _pool_size = self.roi_pool_size * 2
        else:
            _pool_size = self.roi_pool_size
        config = {'pool_size': _pool_size}
        return config

    def backbone(self, input_images):
        '''backbone, implemented in derived classes.'''
        raise NotImplementedError('backbone is not implemented in FrcnnModel base class.')

    def rpn(self, backbone_featuremap):
        '''RPN.'''
        x = Conv2D(512, (3, 3), padding='same',
                   activation='relu', name='rpn_conv1',
                   kernel_regularizer=self.kernel_reg,
                   bias_regularizer=None)(backbone_featuremap)

        x_class = Conv2D(self.num_anchors, (1, 1),
                         activation='sigmoid',
                         name='rpn_out_class',
                         kernel_regularizer=self.kernel_reg,
                         bias_regularizer=None)(x)
        x_regr = Conv2D(self.num_anchors * 4, (1, 1),
                        activation='linear',
                        name='rpn_out_regress',
                        kernel_regularizer=self.kernel_reg,
                        bias_regularizer=None)(x)
        return [x_class, x_regr]

    def proposals(self, rpn_score_head, rpn_deltas_head, input_image):
        '''proposal layer.'''
        rois = Proposal(self.anchor_sizes,
                        self.anchor_ratios,
                        self.std_scaling,
                        self.rpn_stride,
                        self.rpn_pre_nms_top_N,
                        self.rpn_post_nms_top_N,
                        self.rpn_nms_iou_thres,
                        'sigmoid',
                        self.batch_size_per_gpu)([rpn_score_head, rpn_deltas_head, input_image])
        return rois

    def proposals_val(self, spec, rpn_score_head, rpn_deltas_head, input_image):
        '''proposal layer for validation model.'''
        rois = Proposal(self.anchor_sizes,
                        self.anchor_ratios,
                        self.std_scaling,
                        self.rpn_stride,
                        spec.eval_rpn_pre_nms_top_N,
                        spec.eval_rpn_post_nms_top_N,
                        spec.eval_rpn_nms_iou_thres,
                        'sigmoid',
                        spec.eval_batch_size)([rpn_score_head, rpn_deltas_head, input_image])
        return rois

    def proposal_targets(self, rois, input_gt_class, input_gt_bboxes):
        '''proposal target layer.'''
        proposal_targets_out = ProposalTarget(self.gt_as_roi,
                                              self.rcnn_max_overlap,
                                              self.rcnn_min_overlap,
                                              self.num_classes-1,
                                              self.rcnn_train_bs,
                                              0.25,
                                              self.rcnn_bbox_std,
                                              self.batch_size_per_gpu)([rois,
                                                                        input_gt_class,
                                                                        input_gt_bboxes])
        return proposal_targets_out

    def crop_and_resize(self, backbone_featuremap, total_rois, input_image):
        '''CropAndResize layer.'''
        if self.roi_pool_2x:
            _pool_size = self.roi_pool_size * 2
        else:
            _pool_size = self.roi_pool_size
        crop_and_resize_layer = CropAndResize(_pool_size)
        roi_crop = crop_and_resize_layer([backbone_featuremap, total_rois, input_image])
        return roi_crop

    def rcnn(self, roi_crop):
        '''RCNN layer.'''
        out = self.rcnn_body(roi_crop)
        out_class = TimeDistributed(Dense(self.num_classes, activation='softmax',
                                          name='dense_class',
                                          kernel_regularizer=self.kernel_reg,
                                          bias_regularizer=None),
                                    name='dense_class_td')(out)
        # note: no regression target for bg class
        out_regr = TimeDistributed(Dense(4 * (self.num_classes-1),
                                         activation='linear',
                                         name='dense_regress',
                                         kernel_regularizer=self.kernel_reg,
                                         bias_regularizer=None),
                                   name='dense_regress_td')(out)
        return out_class, out_regr

    def rcnn_body(self, x):
        '''RCNN body.'''
        raise NotImplementedError('rcnn_body is not implemented in FrcnnModel base class.')

    def build_keras_model(self, input_images, input_gt_class, input_gt_bbox):
        '''build keras model with these building blocks.'''
        backbone_featuremap = self.backbone(input_images)
        rpn_out_class, rpn_out_regress = self.rpn(backbone_featuremap)
        rois = self.proposals(rpn_out_class, rpn_out_regress, input_images)
        total_rois = self.proposal_targets(rois, input_gt_class, input_gt_bbox)[0]
        roi_crop = self.crop_and_resize(backbone_featuremap, total_rois, input_images)
        rcnn_out_class, rcnn_out_regress = self.rcnn(roi_crop)
        self.inputs = [input_images, input_gt_class, input_gt_bbox]
        self.outputs = [rpn_out_class, rpn_out_regress,
                        rcnn_out_class, rcnn_out_regress]
        self.keras_model = keras.models.Model(inputs=self.inputs,
                                              outputs=self.outputs)
        # Fake quantize the keras model if QAT is enabled
        if self.qat:
            self.keras_model = create_quantized_keras_model(
                self.keras_model,
                freeze_bn=self.freeze_bn,
                training=True
            )
            self.inputs = self.keras_model.inputs
            self.outputs = self.keras_model.outputs

    def build_keras_validation_model(self, spec, input_images):
        '''build unpruned validation keras model with these building blocks.'''
        backbone_featuremap = self.backbone(input_images)
        rpn_out_class, rpn_out_regress = self.rpn(backbone_featuremap)
        rois = self.proposals_val(spec, rpn_out_class, rpn_out_regress, input_images)
        roi_crop = self.crop_and_resize(backbone_featuremap, rois, input_images)
        rcnn_out_class, rcnn_out_regress = self.rcnn(roi_crop)
        inputs = [input_images]
        outputs = [rois, rcnn_out_class, rcnn_out_regress]
        val_model = keras.models.Model(inputs=inputs,
                                       outputs=outputs)
        # Fake quantize the keras model if QAT is enabled
        if self.qat:
            val_model = create_quantized_keras_model(val_model,
                                                     freeze_bn=self.freeze_bn)
        return val_model

    def build_validation_model_unpruned(
        self,
        spec,
        max_box_num=100,
        regr_std_scaling=(10.0, 10.0, 5.0, 5.0),
        iou_thres=0.5,
        score_thres=0.0001
    ):
        """Build the validation model for online validation during training."""
        # tune to inference phase to build the validation model
        prev_lp = keras.backend.learning_phase()
        keras.backend.set_learning_phase(0)
        input_image = Input(shape=spec.input_dims, name='input_image')
        val_model = self.build_keras_validation_model(spec, input_image)
        # attach OutputParser layer
        parser_outputs = OutputParser(max_box_num, list(regr_std_scaling), iou_thres, score_thres)(
            val_model.outputs + val_model.inputs
        )
        val_model = keras.models.Model(
            inputs=val_model.inputs,
            outputs=parser_outputs,
            name=val_model.name
        )
        keras.backend.set_learning_phase(prev_lp)
        return val_model

    def build_validation_model(
        self,
        model,
        config_override,
        max_box_num=100,
        regr_std_scaling=(10.0, 10.0, 5.0, 5.0),
        iou_thres=0.5,
        score_thres=0.0001,
        eval_rois=300
    ):
        """Build the validation model for online validation during training."""
        # clone the training model so it does not use the input tensors
        model_config = model.get_config()
        with CustomObjectScope(CUSTOM_OBJS):
            model = keras.models.Model.from_config(model_config)
        # tune to inference phase to build the validation model
        prev_lp = keras.backend.learning_phase()
        keras.backend.set_learning_phase(0)
        # build a validation model out of the cloned training model
        _explored_layers = dict()
        for l in model.layers:
            _explored_layers[l.name] = [False, None]
        input_layer = [l for l in model.layers if (type(l) == keras.layers.InputLayer)]
        layers_to_explore = input_layer
        model_outputs = {}
        # Loop until we reach the last layer.
        while layers_to_explore:
            layer = layers_to_explore.pop(0)
            # Skip layers that may be revisited in the graph to prevent duplicates.
            if not _explored_layers[layer.name][0]:
                # Check if all inbound layers explored for given layer.
                if not all([
                        _explored_layers[l.name][0]
                        for n in layer._inbound_nodes
                        for l in n.inbound_layers
                        ]):
                    continue
                outputs = None
                # Visit input layer.
                if type(layer) == keras.layers.InputLayer and layer.name == 'input_image':
                    # Re-use the existing InputLayer.
                    outputs = layer.output
                    new_layer = layer
                elif type(layer) == keras.layers.InputLayer:
                    # skip the input_class_ids and input_gt_boxes
                    # mark them as visited but do nothing essential
                    _explored_layers[layer.name][0] = True
                    _explored_layers[layer.name][1] = None
                    layers_to_explore.extend([
                        node.outbound_layer for node in layer._outbound_nodes
                    ])
                    continue
                # special handling for ProposalTarget layer.
                elif type(layer) == ProposalTarget:
                    # get ROIs data.
                    for node in layer._inbound_nodes:
                        prev_outputs = []
                        # only use the first Input: input_rois
                        for idx, l in enumerate(node.inbound_layers[:1]):
                            keras_layer = _explored_layers[l.name][1]
                            prev_outputs.append(keras_layer.get_output_at(node.node_indices[idx]))
                        assert prev_outputs, "Expected non-input layer to have inputs."
                        # remember it
                        if len(prev_outputs) == 1:
                            prev_outputs = prev_outputs[0]
                        proposal_outputs = prev_outputs
                    _explored_layers[layer.name][0] = True
                    _explored_layers[layer.name][1] = None
                    layers_to_explore.extend([
                        node.outbound_layer for node in layer._outbound_nodes
                    ])
                    continue
                # special handling of CropAndResize to skip the ProposalTarget layer.
                elif type(layer) == CropAndResize:
                    # Create new layer.
                    layer_config = layer.get_config()
                    new_layer = type(layer).from_config(layer_config)
                    outputs = []
                    for node in layer._inbound_nodes:
                        prev_outputs = []
                        for idx, l in enumerate(node.inbound_layers):
                            # skip ProposalTarget(idx==1) because it doesn't exist
                            # in validation model. Use None as a placeholder for it
                            # will update the None later
                            if idx == 1:
                                prev_outputs.append(None)
                                continue
                            keras_layer = _explored_layers[l.name][1]
                            prev_outputs.append(keras_layer.get_output_at(node.node_indices[idx]))
                        assert prev_outputs, "Expected non-input layer to have inputs."
                        # replace None with the proposal_outputs
                        prev_outputs[1] = proposal_outputs
                        if len(prev_outputs) == 1:
                            prev_outputs = prev_outputs[0]
                        outputs.append(new_layer(prev_outputs))
                    if len(outputs) == 1:
                        outputs = outputs[0]
                elif ("pre_pool_reshape" in layer.name and type(layer) == keras.layers.Reshape):
                    H, W = layer._inbound_nodes[0].inbound_layers[0].output_shape[3:]
                    new_layer = keras.layers.Reshape((-1, H, W), name=layer.name)
                    outputs = []
                    for node in layer._inbound_nodes:
                        prev_outputs = []
                        for idx, l in enumerate(node.inbound_layers):
                            keras_layer = _explored_layers[l.name][1]
                            prev_outputs.append(keras_layer.get_output_at(node.node_indices[idx]))
                        assert prev_outputs, "Expected non-input layer to have inputs."
                        if len(prev_outputs) == 1:
                            prev_outputs = prev_outputs[0]
                        outputs.append(new_layer(prev_outputs))
                    if len(outputs) == 1:
                        outputs = outputs[0]
                elif ("post_pool_reshape" in layer.name and type(layer) == keras.layers.Reshape):
                    new_layer = keras.layers.Reshape(
                        (eval_rois, -1, 1, 1),
                        name=layer.name
                    )
                    outputs = []
                    for node in layer._inbound_nodes:
                        prev_outputs = []
                        for idx, l in enumerate(node.inbound_layers):
                            keras_layer = _explored_layers[l.name][1]
                            prev_outputs.append(keras_layer.get_output_at(node.node_indices[idx]))
                        assert prev_outputs, "Expected non-input layer to have inputs."
                        if len(prev_outputs) == 1:
                            prev_outputs = prev_outputs[0]
                        outputs.append(new_layer(prev_outputs))
                    if len(outputs) == 1:
                        outputs = outputs[0]
                else:
                    # Create new layer.
                    layer_config = layer.get_config()
                    # override config for Proposal layer for test graph
                    if type(layer) == Proposal:
                        layer_config.update(config_override)
                    new_layer = type(layer).from_config(layer_config)

                    # Add to model.
                    outputs = []
                    for node in layer._inbound_nodes:
                        prev_outputs = []
                        for idx, l in enumerate(node.inbound_layers):
                            keras_layer = _explored_layers[l.name][1]
                            prev_outputs.append(keras_layer.get_output_at(node.node_indices[idx]))
                        assert prev_outputs, "Expected non-input layer to have inputs."
                        if len(prev_outputs) == 1:
                            prev_outputs = prev_outputs[0]
                        outputs.append(new_layer(prev_outputs))
                    if len(outputs) == 1:
                        outputs = outputs[0]
                    weights = layer.get_weights()
                    if weights is not None:
                        new_layer.set_weights(weights)
                outbound_nodes = layer._outbound_nodes
                # RPN outputs will be excluded since it has outbound nodes.
                if not outbound_nodes:
                    model_outputs[layer.output.name] = outputs
                layers_to_explore.extend([node.outbound_layer for node in outbound_nodes])
                # Mark current layer as visited and assign output nodes to the layer.
                _explored_layers[layer.name][0] = True
                _explored_layers[layer.name][1] = new_layer
            else:
                continue
        # Create new keras model object from pruned specifications.
        # only use input_image as Model Input.
        output_tensors = [model_outputs[l.name] for l in model.outputs if l.name in model_outputs]
        output_tensors = [proposal_outputs] + output_tensors
        new_model = keras.models.Model(inputs=model.inputs[:1],
                                       outputs=output_tensors,
                                       name=model.name)
        # attach OutputParser layer
        parser_outputs = OutputParser(max_box_num, list(regr_std_scaling), iou_thres, score_thres)(
            new_model.outputs + new_model.inputs
        )
        new_model = keras.models.Model(
            inputs=new_model.inputs,
            outputs=parser_outputs,
            name=new_model.name
        )
        keras.backend.set_learning_phase(prev_lp)
        return new_model

    def summary(self):
        '''print keras model summary.'''
        self.keras_model.summary()

    def load_weights(self, weights_path, key, logger):
        '''loading pretrained weights for initialization.'''
        assert os.path.isfile(weights_path), '''
        pretrained weights file not found: {}'''.format(weights_path)
        logger.info('Loading pretrained weights '
                    'from {}'.format(weights_path))
        weights_format = weights_path.split('.')[-1]
        # remember the old weights and then compare with new weights after loading
        # to see which layers' weights has been loaded(changed) succesfully.
        old_weights = self.keras_model.get_weights()
        if weights_format == 'tlt':
            # first, convert tlt model to weights
            _model = iva_utils.decode_to_keras(str(weights_path),
                                               str.encode(key),
                                               compile_model=False)
            os_handle, tmp_weights_file = tempfile.mkstemp(suffix='.h5')
            os.close(os_handle)
            _model.save_weights(tmp_weights_file)
            # then load the weights
            self.keras_model.load_weights(str(tmp_weights_file),
                                          by_name=True)
            os.remove(tmp_weights_file)
        elif weights_format == 'hdf5':
            # unencoded keras models from classification network
            with CustomObjectScope(CUSTOM_OBJS):
                _model = keras.models.load_model(str(weights_path),
                                                 compile=False)
            os_handle, tmp_weights_file = tempfile.mkstemp(suffix='.h5')
            os.close(os_handle)
            _model.save_weights(tmp_weights_file)
            # then load the weights
            self.keras_model.load_weights(str(tmp_weights_file),
                                          by_name=True)
            os.remove(tmp_weights_file)
        elif weights_format == 'h5':
            self.keras_model.load_weights(str(weights_path),
                                          by_name=True)
        else:
            raise ValueError('''Unrecognized pretrained
                              weights format {}'''.format(weights_format))
        new_weights = self.keras_model.get_weights()
        self._validate_loaded_weights(old_weights, new_weights)
        logger.info('Pretrained weights loaded!')

    def _validate_loaded_weights(self, old, new):
        _summary = OrderedDict()
        idx = 0
        for layer in self.keras_model.layers:
            if len(layer.weights) == 0:
                # this layer has no weights
                _summary[layer.name] = None
            else:
                # layer have weights
                num_weights = len(layer.weights)
                if self._weights_equal(old[idx:idx+num_weights], new[idx:idx+num_weights]):
                    # weights was not updated
                    _summary[layer.name] = False
                else:
                    # weights was updated
                    _summary[layer.name] = True
                idx += num_weights
        print('='*99)
        print('Pretrained weights loading status summary:')
        print('None: layer has no weights at all.')
        print('Yes: layer has weights and loaded successfully by name.')
        print('No: layer has weights but names not match, skipped.')
        print('='*99)
        print(self._left_align(90, 'Layer(Type):') + self._left_align(9, 'Status:'))
        print('-'*99)
        for l_name in _summary:
            l_type = '({})'.format(type(self.keras_model.get_layer(l_name)).__name__)
            if _summary[l_name] is None:
                _stat = 'None'
            elif _summary[l_name]:
                _stat = 'Yes'
            else:
                _stat = 'No'
            print(self._left_align(90, l_name + l_type) +
                  self._left_align(9, _stat))
            print('-'*99)

    def _weights_equal(self, old, new):
        for idx, w in enumerate(old):
            if not np.array_equal(w, new[idx]):
                return False
        return True

    def _left_align(self, l, s):
        s_len = len(s)
        return s + ' '*(l-s_len)

    @property
    def model_format(self):
        '''format string for output model path.'''
        model_path = str(self.output_model).split('.')
        model_path.insert(-1, 'epoch_{}')
        return '.'.join(model_path)

    def _build_rpn_class_loss(self):
        '''build RPN classification loss.'''
        self.rpn_class_loss = losses._build_rpn_class_loss(self.num_anchors,
                                                           self.lambda_rpn_class,
                                                           self.rpn_train_bs)

    def _build_rpn_bbox_loss(self):
        '''build RPN bbox loss.'''
        self.rpn_bbox_loss = losses._build_rpn_bbox_loss(self.num_anchors,
                                                         self.lambda_rpn_regr,
                                                         self.rpn_train_bs)

    def _build_rcnn_class_loss(self):
        '''build RCNN classification loss.'''
        self.rcnn_class_loss = losses._build_rcnn_class_loss(self.lambda_rcnn_class,
                                                             self.rcnn_train_bs)

    def _build_rcnn_bbox_loss(self):
        '''build RCNN bbox loss.'''
        self.rcnn_bbox_loss = losses._build_rcnn_bbox_loss(self.num_classes,
                                                           self.lambda_rcnn_regr,
                                                           self.rcnn_train_bs)

    def build_losses(self):
        '''build all the losses(totally 4 losses) and remember them.'''
        if self.losses is not None:
            return
        self._build_rpn_class_loss()
        self._build_rpn_bbox_loss()
        self._build_rcnn_class_loss()
        self._build_rcnn_bbox_loss()
        self.losses = [self.rpn_class_loss, self.rpn_bbox_loss,
                       self.rcnn_class_loss, self.rcnn_bbox_loss]

    def build_lr_scheduler(self, max_iters, hvd_size, initial_step=0):
        '''build learning rate scheduler.'''
        if self.lr_config.WhichOneof("lr_config") == 'soft_start':
            lr_config = self.lr_config.soft_start
            scheduler = MultiGPULearningRateScheduler(max_iters,
                                                      lr_config.start_lr*hvd_size,
                                                      lr_config.base_lr*hvd_size,
                                                      lr_config.soft_start,
                                                      lr_config.annealing_points,
                                                      lr_config.annealing_divider)
        elif self.lr_config.WhichOneof("lr_config") == 'step':
            lr_config = self.lr_config.step
            scheduler = StepLRScheduler(lr_config.base_lr*hvd_size,
                                        lr_config.gamma,
                                        lr_config.step_size,
                                        max_iters)
        else:
            raise ValueError('Invalid learning rate config.')
        scheduler.reset(initial_step)
        self.lr_scheduler = scheduler
        self.callbacks.append(self.lr_scheduler)

    def build_checkpointer(self, interval=1):
        '''build tlt encoded model checkpointer.'''
        self.checkpointer = ModelSaver(self.model_format,
                                       self.enc_key,
                                       interval)
        self.callbacks.append(self.checkpointer)

    def set_target_tensors(self, rpn_score_tensor, rpn_deltas_tensor):
        '''setup target tensors for RPN and RCNN.'''
        if self.target_tensors is not None:
            return
        pt_outputs = None
        for l in self.keras_model.layers:
            if type(l) == ProposalTarget:
                pt_outputs = l.output
                break
        assert pt_outputs is not None, "Cannot find ProposalTarget output tensors in Keras model."
        self.target_tensors = [rpn_score_tensor, rpn_deltas_tensor] + \
            pt_outputs[1:3]

    def set_optimizer(self, opt, hvd):
        '''setup optimizer.'''
        if self.optimizer is not None:
            return
        self.optimizer = hvd.DistributedOptimizer(opt)

    def set_hvd_callbacks(self, hvd):
        '''setup horovod callbacks.'''
        self.callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        self.callbacks.append(hvd.callbacks.MetricAverageCallback())
        self.callbacks.append(TerminateOnNaN())

    def build_validation_callback(self, val_data_loader, spec):
        """Build the validation callback for online validation."""
        logger.info("Building validation model, may take a while...")
        if spec.pretrained_model or spec.resume_from_model:
            config_override = {'pre_nms_top_N': spec.eval_rpn_pre_nms_top_N,
                               'post_nms_top_N': spec.eval_rpn_post_nms_top_N,
                               'nms_iou_thres': spec.eval_rpn_nms_iou_thres,
                               'bs_per_gpu': spec.eval_batch_size}
            val_model = self.build_validation_model(
                self.keras_model,
                config_override,
                max_box_num=spec.eval_rcnn_post_nms_top_N,
                regr_std_scaling=spec.rcnn_regr_std,
                iou_thres=spec.eval_rcnn_nms_iou_thres,
                score_thres=spec.eval_confidence_thres,
                eval_rois=spec.eval_rpn_post_nms_top_N
            )
        else:
            val_model = self.build_validation_model_unpruned(
                spec,
                max_box_num=spec.eval_rcnn_post_nms_top_N,
                regr_std_scaling=spec.rcnn_regr_std,
                iou_thres=spec.eval_rcnn_nms_iou_thres,
                score_thres=spec.eval_confidence_thres
            )
        logger.info("Validation model built successfully!")
        val_model.summary()
        validation_callback = ValidationCallback(
            val_model,
            val_data_loader,
            spec.validation_period,
            spec.eval_batch_size,
            spec.eval_confidence_thres,
            spec.use_voc07_metric,
            spec.id_to_class,
            spec.eval_gt_matching_iou_list,
        )
        self.callbacks.append(validation_callback)

    def build_early_stopping_callback(self):
        """Setup early stopping callback."""
        # If early stopping is enabled...
        if self.early_stopping is not None:
            callback = EarlyStopping(
                monitor=self.early_stopping.monitor,
                min_delta=self.early_stopping.min_delta,
                patience=self.early_stopping.patience,
                verbose=True
            )
            self.callbacks.append(callback)

    def build_tensorboard_callback(self):
        """Build TensorBoard callback for visualization."""
        tb_path = os.path.join(
            os.path.dirname(self.output_model),
            "logs"
        )
        if os.path.exists(tb_path) and os.path.isdir(tb_path):
            shutil.rmtree(tb_path)
        if not os.path.exists(tb_path):
            os.makedirs(tb_path)
        tb_callback = TensorBoard(
            log_dir=tb_path,
            write_graph=False,
            weight_hist=False
        )
        self.callbacks.append(tb_callback)

    def build_status_logging_callback(self, results_dir, num_epochs, is_master):
        """Build status logging for TAO API."""
        status_logger = TAOStatusLogger(
            results_dir,
            append=True,
            num_epochs=num_epochs,
            is_master=is_master,
        )
        self.callbacks.append(status_logger)

    def compile(self):
        '''compile the keras model.'''
        self.build_losses()
        self.keras_model.compile(optimizer=self.optimizer,
                                 loss=self.losses,
                                 target_tensors=self.target_tensors)

    def train(self, epochs, steps_per_epoch, initial_epoch):
        '''train the keras model with dataset.'''
        self.keras_model.fit(epochs=epochs,
                             steps_per_epoch=steps_per_epoch,
                             callbacks=self.callbacks,
                             initial_epoch=initial_epoch)

    @staticmethod
    def apply_model_to_new_inputs(model, tf_inputs, freeze_bn=False):
        '''Apply keras model to new input tensors, it will avoid nested model.'''
        # set training=False for BN layers if freeze_bn=True
        def compose_call(prev_call_method):
            def call(self, inputs, training=False):
                return prev_call_method(self, inputs, training)

            return call

        prev_batchnorm_call = keras.layers.normalization.BatchNormalization.call
        prev_td_call = keras.layers.wrappers.TimeDistributed.call
        if freeze_bn:
            keras.layers.normalization.BatchNormalization.call = compose_call(
                prev_batchnorm_call
            )
            keras.layers.wrappers.TimeDistributed.call = compose_call(
                prev_td_call
            )
        _explored_layers = dict()
        for l in model.layers:
            _explored_layers[l.name] = [False, None]
        input_layer = [l for l in model.layers if (type(l) == keras.layers.InputLayer)]
        layers_to_explore = input_layer
        model_outputs = {}
        # Loop until we reach the last layer.
        while layers_to_explore:
            layer = layers_to_explore.pop(0)
            # Skip layers that may be revisited in the graph to prevent duplicates.
            if not _explored_layers[layer.name][0]:
                # Check if all inbound layers explored for given layer.
                if not all([
                        _explored_layers[l.name][0]
                        for n in layer._inbound_nodes
                        for l in n.inbound_layers
                        ]):
                    continue
                outputs = None
                # Visit input layer.
                if type(layer) == keras.layers.InputLayer:  # noqa pylint: disable = R1724
                    # skip input layer and use outside input tensors intead.
                    _explored_layers[layer.name][0] = True
                    _explored_layers[layer.name][1] = None
                    layers_to_explore.extend([node.outbound_layer for
                                              node in layer._outbound_nodes])
                    continue
                elif type(layer) == CropAndResize:
                    # Create new layer.
                    layer_config = layer.get_config()
                    new_layer = type(layer).from_config(layer_config)
                    outputs = []
                    for node in layer._inbound_nodes:
                        prev_outputs = []
                        for idx, l in enumerate(node.inbound_layers):
                            keras_layer = _explored_layers[l.name][1]
                            if keras_layer is not None:
                                # _explored_layers[l.name][1] is None for input image
                                _tmp_outputs = keras_layer.get_output_at(node.node_indices[idx])
                                # ProposalTarget has 4 outputs,
                                # only use the first one for CropAndResize,
                                # i.e., ROIs.
                                if type(l) == ProposalTarget:
                                    _tmp_outputs = _tmp_outputs[0]
                                prev_outputs.append(_tmp_outputs)
                            else:
                                prev_outputs.append(None)
                        assert prev_outputs, "Expected non-input layer to have inputs."
                        # update the input image
                        prev_outputs[-1] = tf_inputs[0]
                        if len(prev_outputs) == 1:
                            prev_outputs = prev_outputs[0]
                        outputs.append(new_layer(prev_outputs))
                    if len(outputs) == 1:
                        outputs = outputs[0]
                else:
                    # Create new layer.
                    layer_config = layer.get_config()
                    new_layer = type(layer).from_config(layer_config)
                    # Add to model.
                    outputs = []
                    for node in layer._inbound_nodes:
                        prev_outputs = []
                        for idx, l in enumerate(node.inbound_layers):
                            if l.name == 'input_image':
                                prev_outputs.append(tf_inputs[0])
                            elif l.name == 'input_gt_cls':
                                prev_outputs.append(tf_inputs[1])
                            elif l.name == 'input_gt_bbox':
                                prev_outputs.append(tf_inputs[2])
                            else:
                                keras_layer = _explored_layers[l.name][1]
                                _tmp_output = keras_layer.get_output_at(node.node_indices[idx])
                                prev_outputs.append(_tmp_output)
                        assert prev_outputs, "Expected non-input layer to have inputs."
                        if len(prev_outputs) == 1:
                            prev_outputs = prev_outputs[0]
                        outputs.append(new_layer(prev_outputs))
                    if len(outputs) == 1:
                        outputs = outputs[0]
                    weights = layer.get_weights()
                    if weights is not None:
                        new_layer.set_weights(weights)
                outbound_nodes = layer._outbound_nodes
                # RPN outputs will be excluded since it has outbound nodes.
                if not outbound_nodes:
                    model_outputs[layer.output.name] = outputs
                # Patch for Faster-RCNN RPN output.
                # It's an output layer, but still has outbound_nodes
                for idx, node in enumerate(new_layer._inbound_nodes):
                    _output = layer.get_output_at(idx)
                    new_output = new_layer.get_output_at(idx)
                    if (_output in model.outputs) and (_output.name not in model_outputs):
                        model_outputs[_output.name] = new_output
                layers_to_explore.extend([node.outbound_layer for node in outbound_nodes])
                # Mark current layer as visited and assign output nodes to the layer.
                _explored_layers[layer.name][0] = True
                _explored_layers[layer.name][1] = new_layer
            else:
                continue
        # Create new keras model object from pruned specifications.
        # only use input_image as Model Input.
        output_tensors = [model_outputs[l.name] for l in model.outputs if l.name in model_outputs]
        new_model = keras.models.Model(inputs=tf_inputs, outputs=output_tensors, name=model.name)
        if freeze_bn:
            # Unpatch Keras before return.
            keras.layers.normalization.BatchNormalization.call = prev_batchnorm_call
            keras.layers.wrappers.TimeDistributed.call = prev_td_call
        return new_model

    def load_pruned_model(self, pruned_model_path, logger):
        '''loading pruned model for retrain.'''
        assert os.path.isfile(pruned_model_path), '''
        Pruned model file not found: {}'''.format(pruned_model_path)
        pruned_model = iva_utils.decode_to_keras(str(pruned_model_path),
                                                 str.encode(self.enc_key),
                                                 input_model=None,
                                                 compile_model=False,
                                                 by_name=None)
        logger.info('Pruned model loaded!')
        return pruned_model

    def override_regularizers(self, model, reg_config):
        """Update regularizers according the spec(config)."""
        regularizer_dict = {
            RegularizerConfig.L1: l1,
            RegularizerConfig.L2: l2
        }
        model_weights = model.get_weights()
        mconfig = model.get_config()
        assert 0.0 < reg_config.weight < 1.0, \
            "Weight decay should be no less than 0 and less than 1"
        # Obtain type and scope of the regularizer
        reg_type = reg_config.type
        for layer, layer_config in zip(model.layers, mconfig['layers']):
            # Regularizer settings
            if reg_type:
                if hasattr(layer, 'kernel_regularizer'):
                    if reg_type in regularizer_dict.keys():
                        regularizer = regularizer_dict[reg_type](reg_config.weight)
                        layer_config['config']['kernel_regularizer'] = \
                            {'class_name': regularizer.__class__.__name__,
                             'config': regularizer.get_config()}
                    else:
                        layer_config['config']['kernel_regularizer'] = None
        with CustomObjectScope({'CropAndResize': CropAndResize,
                                'Proposal': Proposal,
                                'ProposalTarget': ProposalTarget}):
            new_model = keras.models.Model.from_config(mconfig)
        new_model.set_weights(model_weights)
        return new_model

    def override_custom_layers(self, model):
        """Update regularizers according the spec(config)."""
        model_weights = model.get_weights()
        mconfig = model.get_config()
        for layer, layer_config in zip(model.layers, mconfig['layers']):
            # Regularizer settings
            if self.prop_config is not None and type(layer) == Proposal:
                layer_config['config'].update(self.prop_config)
            elif self.propt_config is not None and type(layer) == ProposalTarget:
                layer_config['config'].update(self.propt_config)
            elif self.cr_config is not None and type(layer) == CropAndResize:
                layer_config['config'].update(self.cr_config)
        with CustomObjectScope({'CropAndResize': CropAndResize,
                                'Proposal': Proposal,
                                'ProposalTarget': ProposalTarget}):
            new_model = keras.models.Model.from_config(mconfig)
        new_model.set_weights(model_weights)
        return new_model

    def build_model_from_pruned(self, pruned_model_path, input_images,
                                input_gt_class, inut_gt_bbox, logger,
                                reg_config):
        '''build keras model from pruned model.'''
        logger.info('Loading pretrained model: {} for retrain.'.format(pruned_model_path))
        pruned_model = self.load_pruned_model(pruned_model_path, logger)
        model_qat = check_for_quantized_layers(pruned_model)
        if self.qat ^ model_qat:
            qat_strings = {True: "enabled", False: "disabled"}
            logger.error(
                "Pruned model architecture does not align with "
                f"`enable_qat` flag in spec file. Model QAT is {qat_strings[model_qat]} "
                f"while spec file has QAT {qat_strings[self.qat]}"
            )
            sys.exit(1)
        pruned_model = self.override_regularizers(pruned_model, reg_config)
        pruned_model = self.override_custom_layers(pruned_model)
        logger.info('Regularizers updated for the loaded model.')
        inputs = [input_images, input_gt_class, inut_gt_bbox]
        self.keras_model = self.apply_model_to_new_inputs(pruned_model,
                                                          inputs,
                                                          freeze_bn=self.freeze_bn)
        self.inputs = self.keras_model.inputs
        self.outputs = self.keras_model.outputs
        return self.keras_model

    def get_initial_epoch(self, model_path):
        '''Get the epoch number from the pattern of the saved model path.'''
        epoch = int(model_path.split('epoch_')[1].split(".")[0])
        return epoch

    def resume_model(self, spec, tf_inputs, hvd, logger=None):
        '''resume model from checkpoints and continue to train.'''
        initial_epoch = self.get_initial_epoch(spec.resume_from_model)
        if logger is not None:
            logger.info('Resuming training from {}'.format(spec.resume_from_model))
        # build the loss functions for later use
        self.build_losses()
        custom_objs = {'rpn_loss_cls' : self.rpn_class_loss,
                       'rpn_loss_regr' : self.rpn_bbox_loss,
                       'rcnn_loss_cls' : self.rcnn_class_loss,
                       'rcnn_loss_regr' : self.rcnn_bbox_loss}
        resumed_model = iva_utils.decode_to_keras(spec.resume_from_model,
                                                  str.encode(spec.enc_key),
                                                  input_model=None,
                                                  compile_model=True,
                                                  by_name=None,
                                                  custom_objects=custom_objs)
        optimizer = resumed_model.optimizer
        new_model = self.apply_model_to_new_inputs(resumed_model,
                                                   tf_inputs,
                                                   freeze_bn=self.freeze_bn)
        self.keras_model = new_model
        self.inputs = self.keras_model.inputs
        self.outputs = self.keras_model.outputs
        self.set_optimizer(optimizer, hvd)
        return initial_epoch

    def parallelize(self, parallelism):
        """parallelize the model on multiple GPUs."""
        self.keras_model = self.model_parallelism(
            self.keras_model,
            parallelism,
            freeze_bn=self.freeze_bn
        )
        self.inputs = self.keras_model.inputs
        self.outputs = self.keras_model.outputs

    def model_parallelism(
        self,
        model,
        parallelism,
        freeze_bn=False
    ):
        """Split the model into several parts on multiple GPUs for model parallelism."""
        # set training=False for BN layers if freeze_bn=True
        # otherwise the freeze_bn flag in model builder will be ineffective
        def compose_call(prev_call_method):
            def call(self, inputs, training=False):
                return prev_call_method(self, inputs, training)

            return call

        prev_batchnorm_call = keras.layers.normalization.BatchNormalization.call
        if freeze_bn:
            keras.layers.normalization.BatchNormalization.call = compose_call(
                prev_batchnorm_call
            )
        world_size = len(parallelism)
        # in case that model parallelism is not enabled at all...
        if world_size == 0:
            world_size = 1
            parallelism = (1.0,)
        p_arr = np.array((0.0,) + parallelism, dtype=np.float32)
        cum_p_arr = np.cumsum(p_arr)
        # splitting points for each segment of the model
        splits = cum_p_arr / cum_p_arr[-1]
        layer_splits = np.round(splits * len(model.layers))
        layer_idx = 0
        _explored_layers = dict()
        for l in model.layers:
            _explored_layers[l.name] = [False, None]
        input_layer = [l for l in model.layers if (type(l) == keras.layers.InputLayer)]
        layers_to_explore = input_layer
        model_outputs = {}
        # black list of layer types that cannot run on GPU.
        black_list = [Proposal, ProposalTarget, CropAndResize]
        # Loop until we reach the last layer.
        while layers_to_explore:
            layer = layers_to_explore.pop(0)
            # Skip layers that may be revisited in the graph to prevent duplicates.
            if not _explored_layers[layer.name][0]:
                # Check if all inbound layers explored for given layer.
                if not all([
                        _explored_layers[l.name][0]
                        for n in layer._inbound_nodes
                        for l in n.inbound_layers
                        ]):
                    continue
                outputs = None
                # Visit input layer.
                if type(layer) == keras.layers.InputLayer:
                    # Re-use the existing InputLayer.
                    outputs = layer.output
                    new_layer = layer
                elif type(layer) in black_list:
                    gpu_idx = find_segment_idx(layer_idx, layer_splits)
                    layer_idx += 1
                    # Create new layer.
                    layer_config = layer.get_config()
                    new_layer = type(layer).from_config(layer_config)
                    # Add to model.
                    outputs = []
                    for node in layer._inbound_nodes:
                        prev_outputs = []
                        for idx, l in enumerate(node.inbound_layers):
                            keras_layer = _explored_layers[l.name][1]
                            _tmp_outputs = keras_layer.get_output_at(node.node_indices[idx])
                            if type(l) == ProposalTarget:
                                _tmp_outputs = _tmp_outputs[0]
                            prev_outputs.append(_tmp_outputs)
                        assert prev_outputs, "Expected non-input layer to have inputs."
                        if len(prev_outputs) == 1:
                            prev_outputs = prev_outputs[0]
                        outputs.append(new_layer(prev_outputs))
                    if len(outputs) == 1:
                        outputs = outputs[0]
                    weights = layer.get_weights()
                    if weights is not None:
                        new_layer.set_weights(weights)
                else:
                    gpu_idx = find_segment_idx(layer_idx, layer_splits)
                    layer_idx += 1
                    # pin this layer on a certain GPU
                    with tf.device("/gpu:{}".format(gpu_idx)):
                        # Create new layer.
                        layer_config = layer.get_config()
                        new_layer = type(layer).from_config(layer_config)
                        # Add to model.
                        outputs = []
                        for node in layer._inbound_nodes:
                            prev_outputs = []
                            for idx, l in enumerate(node.inbound_layers):
                                keras_layer = _explored_layers[l.name][1]
                                _tmp_outputs = keras_layer.get_output_at(node.node_indices[idx])
                                if type(l) == ProposalTarget:
                                    _tmp_outputs = _tmp_outputs[0]
                                prev_outputs.append(_tmp_outputs)
                            assert prev_outputs, "Expected non-input layer to have inputs."
                            if len(prev_outputs) == 1:
                                prev_outputs = prev_outputs[0]
                            outputs.append(new_layer(prev_outputs))
                        if len(outputs) == 1:
                            outputs = outputs[0]
                        weights = layer.get_weights()
                        if weights is not None:
                            new_layer.set_weights(weights)
                outbound_nodes = layer._outbound_nodes
                if not outbound_nodes:
                    model_outputs[layer.output.name] = outputs
                # Patch for Faster-RCNN RPN output.
                # It's an output layer, but still has outbound_nodes
                for idx, node in enumerate(new_layer._inbound_nodes):
                    _output = layer.get_output_at(idx)
                    new_output = new_layer.get_output_at(idx)
                    if (_output in model.outputs) and (_output.name not in model_outputs):
                        model_outputs[_output.name] = new_output
                layers_to_explore.extend([node.outbound_layer for node in outbound_nodes])
                # Mark current layer as visited and assign output nodes to the layer.
                _explored_layers[layer.name][0] = True
                _explored_layers[layer.name][1] = new_layer
            else:
                continue
        output_tensors = [model_outputs[l.name] for l in model.outputs if l.name in model_outputs]
        new_model = keras.models.Model(
            inputs=model.inputs,
            outputs=output_tensors,
            name=model.name
        )
        # restore the BN call method before return
        if freeze_bn:
            keras.layers.normalization.BatchNormalization.call = prev_batchnorm_call
        return new_model
