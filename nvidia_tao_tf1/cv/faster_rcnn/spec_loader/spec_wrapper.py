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
'''Wrapper for experiment_spec to make it easier for validation and change.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import re

from keras.regularizers import l1, l2
import numpy as np

from nvidia_tao_tf1.cv.detectnet_v2.proto.regularizer_config_pb2 import RegularizerConfig


class ExperimentSpec(object):
    '''A class to wrap the experiment spec and do some validations.

    When using the experiment spec, usually we did not do extensive validations for the
    parameters in it. If a user provides an invalid parameter that is valid in terms of protobuf
    laguage syntax, it will not be caught by some exception handling mechanism except we have
    implemented the checks before using it. The result is usually the program aborts with some
    Python internal log printed on the screen. This kind of error log is not quite helpful from
    the user's perspective. Sometimes, there is even no errors during the execution of the program.
    From issues raised by users, many of them are due to provding invalid parameters in spec. So we
    need to do some validations here and print useful message to users to help them correct their
    spec files.
    '''

    def __init__(self, spec_proto):
        '''Initialize the ExperimentSpec.

        Args:
            spec_proto(experiment_spec proto): the proto object for experiment_spec.
        '''
        self.spec_proto = spec_proto
        self.validate_spec()

    @property
    def random_seed(self):
        '''random seed.'''
        return int(self.spec_proto.random_seed)

    @property
    def enc_key(self):
        '''encoding key.'''
        return str(self.spec_proto.enc_key)

    @property
    def verbose(self):
        '''verbose or not.'''
        return bool(self.spec_proto.verbose)

    @property
    def model_config(self):
        '''model configurations.'''
        return self.spec_proto.model_config

    @property
    def image_type(self):
        '''Image type.'''
        return int(self.model_config.input_image_config.image_type)

    @property
    def image_c(self):
        '''Image channel number.'''
        return 3 if self.image_type == 0 else 1

    @property
    def image_h(self):
        '''Image height.'''
        if self.model_config.input_image_config.WhichOneof("image_size_config") == "size_min":
            return 0
        return int(self.model_config.input_image_config.size_height_width.height)

    @property
    def image_w(self):
        '''Image width.'''
        if self.model_config.input_image_config.WhichOneof("image_size_config") == "size_min":
            return 0
        return int(self.model_config.input_image_config.size_height_width.width)

    @property
    def image_min(self):
        """Image smaller size of height and width."""
        if self.model_config.input_image_config.WhichOneof("image_size_config") == "size_min":
            return int(self.model_config.input_image_config.size_min.min)
        return 0

    @property
    def input_dims(self):
        '''Input dimensions in (C, H, W) format.'''
        # replace 0 with None to be compatible with Keras Input layer
        _image_h = self.image_h or None
        _image_w = self.image_w or None
        return [self.image_c, _image_h, _image_w]

    @property
    def image_channel_order(self):
        '''Image channel order.'''
        return str(self.model_config.input_image_config.image_channel_order)

    @property
    def image_mean_values(self):
        '''Image mean values per channel.'''
        means = self.model_config.input_image_config.image_channel_mean
        if self.image_c == 3:
            assert ('r' in means) and ('g' in means) and ('b' in means), (
                "'r', 'g', 'b' should all be present in image_channel_mean "
                "for images with 3 channels."
            )
            means_list = [means['r'], means['g'], means['b']]
        else:
            assert 'l' in means, (
                "'l' should be present in image_channel_mean for images "
                "with 1 channel."
            )
            means_list = [means['l']]
        return [float(m) for m in means_list]

    @property
    def image_scaling_factor(self):
        '''Image scaling factor.'''
        return float(self.model_config.input_image_config.image_scaling_factor)

    @property
    def max_objs_per_img(self):
        '''Maximum number of objects in an image in the dataset.'''
        return self.model_config.input_image_config.max_objects_num_per_image

    @property
    def _backbone(self):
        '''backbone type with number of layers.'''
        return str(self.model_config.arch)

    @property
    def backbone(self):
        '''backbone type, without number of layers.'''
        return self._backbone.split(':')[0]

    @property
    def nlayers(self):
        '''number of layers in backbone or subsets like EfficientNet B0, etc.'''
        if ':' in self._backbone:
            # normal case like resnet:18
            if self._backbone.split(':')[1].isdigit():
                return int(self._backbone.split(':')[1])
            # case like efficientnet:b0
            return self._backbone.split(':')[1]
        return None

    @property
    def anchor_config(self):
        '''anchor box configurations.'''
        anc_config = self.model_config.anchor_box_config
        return anc_config

    @property
    def anchor_sizes(self):
        '''anchor box sizes configurations.'''
        anc_scales = list(self.anchor_config.scale)
        return [float(a) for a in anc_scales]

    @property
    def anchor_ratios(self):
        '''anchor box ratios configurations.'''
        anc_ratios = list(self.anchor_config.ratio)
        return [float(a) for a in anc_ratios]

    @property
    def freeze_bn(self):
        '''freeze BN layers or not.'''
        return bool(self.model_config.freeze_bn)

    @property
    def freeze_blocks(self):
        '''List of blocks to freeze.'''
        blocks = list(self.model_config.freeze_blocks)
        return [int(b) for b in blocks]

    @property
    def dropout_rate(self):
        """Dropout rate."""
        if self.model_config.dropout_rate < 0.0:
            raise ValueError("Dropout rate cannot be negative. Got {}.".format(
                self.model_config.dropout_rate
            ))
        return float(self.model_config.dropout_rate)

    @property
    def drop_connect_rate(self):
        """Drop-connect rate."""
        if self.model_config.drop_connect_rate < 0.0:
            raise ValueError("Drop connect rate cannot be negative. Got {}.".format(
                self.model_config.drop_connect_rate
            ))
        return float(self.model_config.drop_connect_rate)

    @property
    def rcnn_train_bs(self):
        '''RCNN train batch size.'''
        return int(self.model_config.roi_mini_batch)

    @property
    def rpn_stride(self):
        '''RPN stride with respect to input image.'''
        assert 16.0 == float(self.model_config.rpn_stride), '''
        RPN stride can only be 16, got {}'''.format(self.model_config.rpn_stride)
        return 16.0

    @property
    def conv_bn_share_bias(self):
        '''Conv and BN layers share bias or not.'''
        return not bool(self.model_config.use_bias)

    @property
    def roi_pool_size(self):
        '''CropAndResize output spatial size.'''
        return int(self.model_config.roi_pooling_config.pool_size)

    @property
    def roi_pool_2x(self):
        '''Whether or not to double the roi_pool_size and then apply a pooling/strided conv.'''
        return bool(self.model_config.roi_pooling_config.pool_size_2x)

    @property
    def all_projections(self):
        '''Whether or not to use all_projections for shorcut connections.'''
        return bool(self.model_config.all_projections)

    @property
    def use_pooling(self):
        '''use pooling or use strided conv instead.'''
        return bool(self.model_config.use_pooling)

    @property
    def enable_qat(self):
        '''Enable QAT or not.'''
        return bool(self.training_config.enable_qat)

    @property
    def activation_type(self):
        """activation function type."""
        return str(self.model_config.activation.activation_type)

    @property
    def training_config(self):
        '''Training config.'''
        return self.spec_proto.training_config

    @property
    def training_dataset(self):
        '''Training dataset.'''
        # data_sources can be repeated(multiple)
        for ds in self.spec_proto.dataset_config.data_sources:
            image_path = str(ds.image_directory_path)
            assert os.path.isdir(image_path), (
                "Training image path not found: {}".format(image_path)
            )
            tfrecords_path = str(ds.tfrecords_path)
            tfrecords = glob.glob(tfrecords_path)
            assert tfrecords, (
                "No TFRecord file found with the pattern : {}".format(tfrecords_path)
            )
        val_type = self.spec_proto.dataset_config.WhichOneof('dataset_split_type')
        if val_type == 'validation_fold':
            val_fold = self.spec_proto.dataset_config.validation_fold
            assert val_fold >= 0, (
                "Validation fold should be non-negative, got {}".format(val_fold)
            )
        elif val_type == 'validation_data_source':
            val_set = self.spec_proto.dataset_config.validation_data_source
            assert os.path.isdir(str(val_set.image_directory_path)), (
                "Validation image directory not found: {}".format(str(val_set.image_directory_path))
            )
            val_tfrecords = str(val_set.tfrecords_path)
            assert glob.glob(val_tfrecords), (
                "Validation TFRecords with the pattern: {} not found.".format(val_tfrecords)
            )
        image_ext = str(self.spec_proto.dataset_config.image_extension)
        assert image_ext.lower() in ['jpg', 'jpeg', 'png'], (
            "Only image format jpg/jpeg/png are supported, "
            "got extension {}".format(image_ext)
        )
        return self.spec_proto.dataset_config

    @property
    def class_mapping(self):
        '''class mapping.'''
        cm = dict(self.spec_proto.dataset_config.target_class_mapping)
        assert len(cm), 'Class mapping is empty.'
        # class_mapping should not contains a background class because we
        # will append it implicitly
        assert 'background' not in cm, (
            "Class mapping should not "
            "contain a background class."
        )
        return cm

    @property
    def class_to_id(self):
        '''dict to map class names to class IDs(including background).'''
        class_names = sorted(set(self.class_mapping.values())) + ['background']
        return dict(zip(class_names, range(len(class_names))))

    @property
    def id_to_class(self):
        '''dict to map class IDs to class names(including background).'''
        class_names = self.class_to_id.keys()
        class_ids = [self.class_to_id[c] for c in class_names]
        return dict(zip(class_ids, class_names))

    @property
    def num_classes(self):
        '''number of classes(including background).'''
        return len(self.class_to_id.keys())

    @property
    def data_augmentation(self):
        '''data augmentation config.'''
        return self.spec_proto.augmentation_config

    @property
    def enable_augmentation(self):
        '''Enable data augmentation or not.'''
        return bool(self.training_config.enable_augmentation)

    @property
    def epochs(self):
        '''Number of epochs for training.'''
        return int(self.training_config.num_epochs)

    @property
    def batch_size_per_gpu(self):
        '''Image batch size per GPU.'''
        return int(self.training_config.batch_size_per_gpu)

    @property
    def pretrained_weights(self):
        '''path of the pretrained weights.'''
        pw = str(self.training_config.pretrained_weights)
        if pw:
            assert os.path.isfile(pw), (
                "Pretrained weights not found: {}".format(pw)
            )
        return pw

    @property
    def pretrained_model(self):
        '''path of the pretrained(pruned) model.'''
        pm = str(self.training_config.retrain_pruned_model)
        if pm:
            assert os.path.isfile(pm), (
                "Pruned model for retrain not found: {}".format(pm)
            )
        return pm

    @property
    def resume_from_model(self):
        '''resume training from checkpoint model.'''
        rm = str(self.training_config.resume_from_model)
        if rm:
            assert os.path.isfile(rm), (
                "Model to be resumed is not found: {}".format(rm)
            )
            assert re.match(r'.*\.epoch_[0-9]+\.[tlt|hdf5]', rm), (
                "`resume_from_model` path not conforming to the saved model pattern: "
                r"`*\.epoch_[0-9]+\.[tlt|hdf5]`"
                ", got {}".format(rm)
            )
        return rm

    @property
    def checkpoint_interval(self):
        '''Saving checkpoint every k epochs.'''
        # defaults to k=1(unset)
        if self.training_config.checkpoint_interval == 0:
            return 1
        return int(self.training_config.checkpoint_interval)

    @property
    def rpn_min_overlap(self):
        '''RPN min overlap below which we regard anchors as negative anchors.'''
        return float(self.training_config.rpn_min_overlap)

    @property
    def rpn_max_overlap(self):
        '''RPN max overlap above which we regard anchors as positive anchors.'''
        return float(self.training_config.rpn_max_overlap)

    @property
    def rcnn_min_overlap(self):
        '''RCNN min overlap below which we regard RoIs as negative.'''
        return float(self.training_config.classifier_min_overlap)

    @property
    def rcnn_max_overlap(self):
        '''RCNN max overlap above which we regard RoIs as positive.'''
        return float(self.training_config.classifier_max_overlap)

    @property
    def gt_as_roi(self):
        '''Whether or not to use groundtruth boxes as RoIs to train RCNN.'''
        return bool(self.training_config.gt_as_roi)

    @property
    def std_scaling(self):
        '''scaling factor applied to RPN deltas output.'''
        return float(self.training_config.std_scaling)

    @property
    def rcnn_regr_std(self):
        '''scaling factors applied to RCNN deltas output.'''
        stds = dict(self.training_config.classifier_regr_std)
        return [float(stds['x']), float(stds['y']),
                float(stds['w']), float(stds['h'])]

    @property
    def rpn_train_bs(self):
        '''training batch size for RPN for each image.'''
        return int(self.training_config.rpn_mini_batch)

    @property
    def rpn_pre_nms_top_N(self):
        '''RPN pre NMS top N.'''
        return int(self.training_config.rpn_pre_nms_top_N)

    @property
    def rpn_post_nms_top_N(self):
        '''RPN post NMS top N.'''
        return int(self.training_config.rpn_nms_max_boxes)

    @property
    def rpn_nms_iou_thres(self):
        '''IoU threshold for RPN NMS.'''
        return float(self.training_config.rpn_nms_overlap_threshold)

    @property
    def regularization_config(self):
        '''regularization config.'''
        return self.training_config.regularizer

    @property
    def reg_type(self):
        '''regularization type in enum.'''
        return self.regularization_config.type

    @property
    def regularizer(self):
        '''regularizer in keras object.'''
        if self.type == RegularizerConfig.L1:
            return l1
        if self.type == RegularizerConfig.L2:
            return l2
        return None

    @property
    def weight_decay(self):
        '''weight decay factor.'''
        return float(self.regularization_config.weight)

    @property
    def optimizer(self):
        '''Optimizer.'''
        return self.training_config.optimizer

    @property
    def lr_scheduler(self):
        '''Learning rate scheduler.'''
        return self.training_config.learning_rate

    @property
    def lambda_rpn_regr(self):
        '''scaling factor for RPN regressor loss.'''
        return float(self.training_config.lambda_rpn_regr)

    @property
    def lambda_rpn_class(self):
        '''scaling factor for RPN classifier loss.'''
        return float(self.training_config.lambda_rpn_class)

    @property
    def lambda_cls_regr(self):
        '''scaling factor for RCNN classifier loss.'''
        return float(self.training_config.lambda_cls_regr)

    @property
    def lambda_cls_class(self):
        '''scaling factor for RCNN regressor loss.'''
        return float(self.training_config.lambda_cls_class)

    @property
    def inference_config(self):
        '''inference config.'''
        return self.spec_proto.inference_config

    @property
    def inference_images_dir(self):
        '''the path to the image directory for doing inference.'''
        infer_image_dir = str(self.inference_config.images_dir)
        assert infer_image_dir and os.path.isdir(infer_image_dir), (
            "Inference images directory not found: {}".format(infer_image_dir)
        )
        image_ext = str(self.spec_proto.dataset_config.image_extension)
        images = glob.glob(os.path.join(infer_image_dir, '*.'+image_ext))
        assert images, (
            "Inference images not found in the directory: {}".format(infer_image_dir)
        )
        return str(self.inference_config.images_dir)

    @property
    def inference_model(self):
        '''The model path for doing inference.'''
        # Inference model may not exist at the start time,
        # so we check it when it is called here instead of in constructor.
        assert os.path.isfile(str(self.inference_config.model)), '''
        Inference model not found: {}'''.format(str(self.inference_config.model))
        return str(self.inference_config.model)

    @property
    def inference_trt_config(self):
        '''TensorRT inference config.'''
        if self.inference_config.HasField('trt_inference'):
            return self.inference_config.trt_inference
        return None

    @property
    def inference_trt_engine(self):
        '''The TensorRT engine file from tlt-converter for inference.'''
        if (self.inference_trt_config is not None):
            _engine_file = str(self.inference_trt_config.trt_engine)
            assert os.path.isfile(_engine_file), \
                'TensorRT Engine for inference not found: {}'.format(_engine_file)
            return _engine_file
        return None

    @property
    def inference_output_images_dir(self):
        '''The output image directory during inference.'''
        return str(self.inference_config.detection_image_output_dir)

    @property
    def inference_output_labels_dir(self):
        '''The output labels directory during inference.'''
        return str(self.inference_config.labels_dump_dir)

    @property
    def infer_rpn_pre_nms_top_N(self):
        '''RPN pre NMS top N during inference.'''
        return int(self.inference_config.rpn_pre_nms_top_N)

    @property
    def infer_rpn_post_nms_top_N(self):
        '''RPN post NMS top N during inference.'''
        return int(self.inference_config.rpn_nms_max_boxes)

    @property
    def infer_rpn_nms_iou_thres(self):
        '''RPN NMS IoU threshold during inference.'''
        return float(self.inference_config.rpn_nms_overlap_threshold)

    @property
    def vis_conf(self):
        '''bbox visualize confidence threshold for inference.'''
        return float(self.inference_config.bbox_visualize_threshold)

    @property
    def infer_confidence_thres(self):
        '''bbox confidence threshold for inference for NMS export.'''
        return float(self.inference_config.object_confidence_thres)

    @property
    def infer_rcnn_post_nms_top_N(self):
        '''RCNN post NMS top N during inference.'''
        return int(self.inference_config.classifier_nms_max_boxes)

    @property
    def infer_rcnn_nms_iou_thres(self):
        '''RCNN NMS IoU threshold during inference.'''
        return float(self.inference_config.classifier_nms_overlap_threshold)

    @property
    def infer_batch_size(self):
        """Batch size for inference."""
        # defaults to 1 if 0(unset)
        return int(self.inference_config.batch_size) or 1

    @property
    def infer_nms_score_bits(self):
        """NMS score bits for TensorRT inference."""
        return int(self.inference_config.nms_score_bits)

    @property
    def eval_config(self):
        '''Evaluation config.'''
        return self.spec_proto.evaluation_config

    @property
    def eval_trt_config(self):
        """TensorRT based evaluation config."""
        if self.eval_config.HasField("trt_evaluation"):
            return self.eval_config.trt_evaluation
        return None

    @property
    def eval_trt_engine(self):
        '''The TensorRT engine file from tlt-converter for evaluation.'''
        if (self.eval_trt_config is not None):
            _engine_file = str(self.eval_trt_config.trt_engine)
            assert os.path.isfile(_engine_file), \
                'TensorRT Engine for evaluation not found: {}'.format(_engine_file)
            return _engine_file
        return None

    @property
    def eval_model(self):
        '''Model path for evaluation.'''
        _model = str(self.eval_config.model)
        assert os.path.isfile(_model), (
            "Evaluation model not found: {}".format(_model)
        )
        return _model

    @property
    def eval_rpn_pre_nms_top_N(self):
        '''RPN pre nms top N during evaluation.'''
        return int(self.eval_config.rpn_pre_nms_top_N)

    @property
    def eval_rpn_post_nms_top_N(self):
        '''RPN post NMS top N during evaluation.'''
        return int(self.eval_config.rpn_nms_max_boxes)

    @property
    def eval_rpn_nms_iou_thres(self):
        '''RPN NMS IoU threshold for evaluation.'''
        return float(self.eval_config.rpn_nms_overlap_threshold)

    @property
    def eval_rcnn_post_nms_top_N(self):
        '''RCNN post NMS top N during evaluation.'''
        return int(self.eval_config.classifier_nms_max_boxes)

    @property
    def eval_rcnn_nms_iou_thres(self):
        '''RCNN NMS IoU threshold for evaluation.'''
        return float(self.eval_config.classifier_nms_overlap_threshold)

    @property
    def eval_confidence_thres(self):
        '''Confidence threshold for evaluation.'''
        return float(self.eval_config.object_confidence_thres)

    @property
    def validation_period(self):
        """Validation period during training for online validation."""
        val_period = int(self.eval_config.validation_period_during_training)
        # defaults to 1 if not set
        if val_period == 0:
            val_period = 1
        return val_period

    @property
    def eval_batch_size(self):
        """Batch size for evaluation and online validation."""
        if int(self.eval_config.batch_size):
            return int(self.eval_config.batch_size)
        # if 0(unset), use 1 as default
        return 1

    @property
    def use_voc07_metric(self):
        '''Whether or not to use PASCAL VOC 07 metric for AP calculation.'''
        return bool(self.eval_config.use_voc07_11point_metric)

    @property
    def eval_gt_matching_iou_thres(self):
        """Evaluation IoU threshold between detected box and groundtruth box."""
        if self.eval_config.WhichOneof("iou_threshold_config") == "gt_matching_iou_threshold":
            assert 0.0 < self.eval_config.gt_matching_iou_threshold < 1.0, (
                "IoU threshold should be in the range (0, 1), got {}".format(
                    self.eval_config.gt_matching_iou_threshold
                )
            )
            return self.eval_config.gt_matching_iou_threshold
        return None

    @property
    def eval_gt_matching_iou_thres_range(self):
        """Evaluation IoU threshold range between detected box and groundtruth box."""
        if self.eval_config.WhichOneof("iou_threshold_config") == "gt_matching_iou_threshold_range":
            thres_range = self.eval_config.gt_matching_iou_threshold_range
            assert 0.0 < thres_range.start < thres_range.end <= 1.0, (
                "IoU threshold should be in the range (0, 1), got start: {}, end: {}".format(
                    thres_range.start,
                    thres_range.end
                )
            )
            assert 0.0 < thres_range.step < 1.0, (
                "IoU threshold range step size should be in (0, 1), got {}".format(
                    thres_range.step,
                )
            )
            return self.eval_config.gt_matching_iou_threshold_range
        return None

    @property
    def eval_gt_matching_iou_list(self):
        """The list of IoUs for matching detected boxes and groundtruth boxes."""
        if self.eval_gt_matching_iou_thres_range is not None:
            return np.arange(
                self.eval_gt_matching_iou_thres_range.start,
                self.eval_gt_matching_iou_thres_range.end,
                self.eval_gt_matching_iou_thres_range.step
            ).tolist()
        if self.eval_gt_matching_iou_thres is not None:
            return [self.eval_gt_matching_iou_thres]
        raise ValueError(
            "Either specify a gt_matching_iou_threshold_range "
            "or a gt_matching_iou_threshold in the evaluation_config. "
            "Neither is found."
        )

    @property
    def early_stopping(self):
        """Early stopping config."""
        if self.training_config.HasField("early_stopping"):
            es = self.training_config.early_stopping
            if es.monitor not in ["loss"]:
                raise ValueError(
                    "Only `loss` is supported monitor"
                    f", got {es.monitor}"
                )
            if es.min_delta < 0.:
                raise ValueError(
                    f"`min_delta` should be non-negative, got {es.min_delta}"
                )
            if es.patience == 0:
                raise ValueError(
                    f"`patience` should be positive, got {es.patience}"
                )
            return es
        return None

    def validate_spec(self):
        '''Validate parameters in spec file.'''
        self.validate_model_config()
        self.validate_training_config()
        self.validate_evaluation_config()
        self.validate_inference_config()

    def validate_model_config(self):
        '''Check for model config.'''
        # Check image type
        assert self.image_type in [0, 1], '''
        Input image type can only be RGB(0) or grayscale(1),
         got {}'''.format(self.image_type)

        # Check image channel order
        assert self.image_channel_order in ['bgr', 'rgb', 'l'], '''
        Image channel order can only be bgr, rgb or l,
         got {}'''.format(self.image_channel_order)

        # Check image height and width
        assert (self.image_h == 0 or self.image_h >= 160), '''Image height should be at least 160,
         got {}'''.format(self.image_h)
        assert (self.image_w == 0 or self.image_w >= 160), '''Image width should be at least 160,
         got {}'''.format(self.image_w)
        assert (self.image_min == 0 or self.image_min >= 160), (
            "Image min side should be at least 160, got {}".format(self.image_min)
        )
        # Check image mean values
        assert len(self.image_mean_values) == self.image_c, '''
        Length of image mean values: {} does not match
         image channel number: {}'''.format(len(self.image_mean_values), self.image_c)
        for idx, m in enumerate(self.image_mean_values):
            assert 0.0 < m < 255.0, '''image_mean_values[{}]
             should be between 0.0 and 255.0, got {}'''.format(idx, m)

        # Check image scaling factor
        assert self.image_scaling_factor > 0.0, '''
        Image scaling factor should be positive,
         got {}'''.format(self.image_scaling_factor)

        # check max_objs_per_img
        assert self.max_objs_per_img > 0, ('Maximum number of objects in an image should be ' +
                                           'positive, got {}'.format(self.max_objs_per_img))
        # Check backbone
        _valid_backbones = ['resnet:10',
                            'resnet:18',
                            'resnet:34',
                            'resnet:50',
                            'resnet:101',
                            'vgg16',
                            'vgg:16',
                            'vgg:19',
                            'googlenet',
                            'mobilenet_v1',
                            'mobilenet_v2',
                            'darknet:19',
                            'darknet:53',
                            'resnet101',
                            'efficientnet:b0',
                            'efficientnet:b1',
                            'efficientnet:b2',
                            'efficientnet:b3',
                            'efficientnet:b4',
                            'efficientnet:b5',
                            'efficientnet:b6',
                            'efficientnet:b7']
        assert self._backbone in _valid_backbones, '''
        Backbone {} is not implemented, please
         choose from {}.'''.format(self._backbone, _valid_backbones)

        # Check Anchors
        assert len(self.anchor_sizes) > 0, '''
         Anchor sizes should not be empty.'''
        assert len(self.anchor_ratios) > 0, '''
         Anchor ratios should not be empty.'''
        for _as in self.anchor_sizes:
            assert _as > 0.0, '''Anchor size should be positive,
             got {}'''.format(_as)
        for _ar in self.anchor_ratios:
            assert _ar > 0.0, '''Anchor ratios should be positive
            , got {}'''.format(_ar)

        # Check freeze_blocks
        if self._backbone.startswith('resnet'):
            assert set(self.freeze_blocks) <= set([0, 1, 2, 3]), '''
            ResNet freeze_blocks should be a subset of {}
             got {}'''.format([0, 1, 2, 3], self.freeze_blocks)
        elif self._backbone.startswith('vgg'):
            assert set(self.freeze_blocks) <= set([1, 2, 3, 4, 5]), '''
            VGG freeze_blocks should be a subset of {}
             got {}'''.format([1, 2, 3, 4, 5], self.freeze_blocks)
        elif self._backbone.startswith('googlenet'):
            assert set(self.freeze_blocks) <= set([0, 1, 2, 3, 4, 5, 6, 7]), '''
            GoogLeNet freeze_blocks should be a subset of {}
             got {}'''.format([0, 1, 2, 3, 4, 5, 6, 7], self.freeze_blocks)
        elif self._backbone.startswith('mobilenet_v1'):
            assert set(self.freeze_blocks) <= set(range(12)), '''
            MobileNet V1 freeze_blocks should be a subset of {}
             got {}'''.format(list(range(12)), self.freeze_blocks)
        elif self._backbone.startswith('mobilenet_v2'):
            assert set(self.freeze_blocks) <= set(range(14)), '''
            MobileNet V2 freeze_blocks should be a subset of {}
             got {}'''.format(list(range(14)), self.freeze_blocks)
        elif self._backbone.startswith('darknet'):
            assert set(self.freeze_blocks) <= set(range(6)), '''
            DarkNet freeze_blocks should be a subset of {}
             got {}'''.format(list(range(6)), self.freeze_blocks)

        assert self.rcnn_train_bs > 0, '''
        RCNN train batch size should be a positive integer
         got {}'''.format(self.rcnn_train_bs)

    def validate_training_config(self):
        '''Check for training config.'''
        self.validate_augmentation()
        assert self.batch_size_per_gpu >= 1, '''
        Batch size per GPU should be positive, got {}'''.format(self.batch_size_per_gpu)
        assert self.epochs > 0, '''
        Number of epochs should be positive, got {}'''.format(self.epochs)
        assert 0.0 <= self.rcnn_min_overlap < self.rcnn_max_overlap <= 1.0, '''
        RCNN min overlap should be non-negative and less than
         RCNN max overlap, got {}(min),
         and {}(max)'''.format(self.rcnn_min_overlap, self.rcnn_max_overlap)
        assert 0.0 < self.rpn_min_overlap < self.rpn_max_overlap <= 1.0, '''
        RPN min overlap should be positive and less than RPN max overlap
         got {}(min) and {}(max)'''.format(self.rpn_min_overlap, self.rpn_max_overlap)
        assert self.std_scaling > 0.0, '''std_scaling should be positive
         got {}'''.format(self.std_scaling)
        for idx, s in enumerate(self.rcnn_regr_std):
            assert s > 0.0, '''RCNN regressor std[{}] should be positive
             got {}'''.format(idx, s)

        assert self.rpn_train_bs > 0, '''RPN train batch size should be positive
         got {}'''.format(self.rpn_train_bs)
        assert self.rpn_post_nms_top_N > 0, '''RPN post NMS top N should be positive
         got {}'''.format(self.rpn_post_nms_top_N)
        assert self.rpn_post_nms_top_N < self.rpn_pre_nms_top_N, '''
        RPN post NMS topN should be less than RPN pre NMS top N
         got {}'''.format(self.rpn_post_nms_top_N)
        assert 1.0 > self.rpn_nms_iou_thres > 0.0, '''
        RPN NMS IoU threshold should in (0, 1),
         got {}'''.format(self.rpn_nms_iou_thres)
        if self.reg_type in [RegularizerConfig.L1, RegularizerConfig.L2]:
            assert 0.0 < self.weight_decay < 1.0, '''
            Weight decay should be positive and less than 1.0, got {}'''.format(self.weight_decay)
        assert self.lambda_rpn_regr > 0.0, '''
        lambda_rpn_regr should be positive, got {}'''.format(self.lambda_rpn_regr)
        assert self.lambda_rpn_class > 0.0, '''
        lambda_rpn_class should be positive, got {}'''.format(self.lambda_rpn_class)
        assert self.lambda_cls_regr > 0.0, '''
        lambda_cls_regr should be positive, got {}'''.format(self.lambda_cls_regr)
        assert self.lambda_cls_class > 0.0, '''
        lambda_cls_class should be positive, got {}'''.format(self.lambda_cls_class)

    def validate_inference_config(self):
        '''Check for inference config.'''
        assert 0 < self.infer_rpn_post_nms_top_N < self.infer_rpn_pre_nms_top_N, '''
        Inference RPN post NMS should be positive and less than Inference RPN pre NMS top N
         got {}(pre) and {}(post)'''.format(self.infer_rpn_pre_nms_top_N,
                                            self.infer_rpn_post_nms_top_N)
        assert 0.0 < self.infer_rpn_nms_iou_thres < 1.0, '''
        Inference RPN NMS IoU threshold should be in (0, 1), got
         {}'''.format(self.infer_rpn_nms_iou_thres)
        assert 0.0 < self.vis_conf < 1.0, '''
        Bbox visualization threshold for inference should be in (0, 1),
         got {}'''.format(self.vis_conf)
        assert 0.0 < self.infer_confidence_thres < 1.0, (
            "object_confidence_thres for inference should be in (0, 1), "
            "got {}").format(self.infer_confidence_thres)
        assert 0.0 < self.infer_rcnn_nms_iou_thres < 1.0, '''
        Inference RCNN NMS IoU threshold should be in (0, 1),
         got {}'''.format(self.infer_rcnn_nms_iou_thres)
        assert self.infer_rcnn_post_nms_top_N > 0, '''
        Inference RCNN post NMS top N should be positive,
         got {}'''.format(self.infer_rcnn_post_nms_top_N)

    def validate_evaluation_config(self):
        '''Check for evaluation config.'''
        assert 0 < self.eval_rpn_post_nms_top_N < self.eval_rpn_pre_nms_top_N, '''
        Evaluation RPN post NMS top N should be positive and less than
         pre NMS top N, got {}(pre), and {}(post)'''.format(self.eval_rpn_pre_nms_top_N,
                                                            self.eval_rpn_post_nms_top_N)
        assert 0.0 < self.eval_rpn_nms_iou_thres < 1.0, '''
        Evaluation RPN NMS IoU threshold should be in (0, 1),
         got {}'''.format(self.eval_rpn_nms_iou_thres)
        assert 0.0 < self.eval_rcnn_nms_iou_thres < 1.0, '''
        Evaluation RCNN NMS IoU threshold should be in (0, 1),
        got {}'''.format(self.eval_rcnn_nms_iou_thres)
        assert self.eval_rcnn_post_nms_top_N > 0, '''
        Evaluation RCNN post NMS top N should be positive,
        got {}'''.format(self.eval_rcnn_post_nms_top_N)
        assert 0.0 < self.eval_confidence_thres < 1.0, '''
        Evaluation object confidence threshold should be positive,
         got {}'''.format(self.eval_confidence_thres)

    def validate_augmentation(self):
        '''Check for data augmentation config.'''
        aug = self.data_augmentation
        if (self.image_w > 0 and aug.preprocessing.output_image_width > 0):
            assert aug.preprocessing.output_image_width == self.image_w, '''
            Augmentation ouput image width not match model input width
            {} vs {}'''.format(aug.preprocessing.output_image_width,
                               self.image_w)
        if (self.image_h > 0 and aug.preprocessing.output_image_height > 0):
            assert aug.preprocessing.output_image_height == self.image_h, '''
            Augmentation output image height not match model input height
            {} vs {}'''.format(aug.preprocessing.output_image_height,
                               self.image_h)
        assert aug.preprocessing.output_image_channel == self.image_c, '''
        Augmentation output image channel number not match model input
         channel number, {} vs {}'''.format(aug.preprocessing.output_image_channel,
                                            self.image_c)
