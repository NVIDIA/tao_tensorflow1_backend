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
"""EfficientDet model definition.

[1] Mingxing Tan, Ruoming Pang, Quoc Le.
        EfficientDet: Scalable and Efficient Object Detection.
        CVPR 2020, https://arxiv.org/abs/1911.09070
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import re

import tensorflow.compat.v1 as tf

from nvidia_tao_tf1.cv.efficientdet.backbone import backbone_factory
from nvidia_tao_tf1.cv.efficientdet.layers.image_resize_layer import ImageResizeLayer
from nvidia_tao_tf1.cv.efficientdet.layers.weighted_fusion_layer import WeightedFusion
from nvidia_tao_tf1.cv.efficientdet.models.utils_keras import BoxNet, ClassNet
from nvidia_tao_tf1.cv.efficientdet.utils import hparams_config
from nvidia_tao_tf1.cv.efficientdet.utils import utils
from nvidia_tao_tf1.cv.efficientdet.utils.distributed_utils import MPI_is_distributed
from nvidia_tao_tf1.cv.efficientdet.utils.distributed_utils import MPI_rank
from nvidia_tao_tf1.cv.efficientdet.utils.model_loader import dump_json, get_model_with_input


################################################################################
def freeze_vars(variables, pattern):
    """Removes backbone+fpn variables from the input.

    Args:
        variables: all the variables in training
        pattern: a reg experession such as ".*(efficientnet|fpn_cells).*".

    Returns:
        var_list: a list containing variables for training
    """
    if pattern:
        variables = [v for v in variables if not re.match(pattern, v.name)]
    return variables


def resample_feature_map(feat,
                         name,
                         target_height,
                         target_width,
                         target_num_channels,
                         apply_bn=False,
                         is_training=None,
                         conv_after_downsample=False,
                         use_native_resize_op=False,
                         pooling_type=None,
                         data_format='channels_last'):
    """Resample input feature map to have target number of channels and size."""
    if data_format == 'channels_first':
        _, num_channels, height, width = feat.get_shape().as_list()
    else:
        _, height, width, num_channels = feat.get_shape().as_list()

    if height is None or width is None or num_channels is None:
        raise ValueError(
            'shape[1] or shape[2] or shape[3] of feat is None (shape:{}).'.format(
                feat.shape))
    if apply_bn and is_training is None:
        raise ValueError('If BN is applied, need to provide is_training')

    def _maybe_apply_1x1(feat, name='conv_1x1'):
        """Apply 1x1 conv to change layer width if necessary."""
        if num_channels != target_num_channels:
            feat = tf.keras.layers.Conv2D(
                filters=target_num_channels,
                kernel_size=(1, 1),
                padding='same',
                name=name,
                data_format=data_format)(feat)
            if apply_bn:
                feat = utils.batch_norm_act(
                    feat,
                    is_training_bn=is_training,
                    act_type=None,
                    data_format=data_format,
                    name=name+'_bn')
        return feat

    # If conv_after_downsample is True, when downsampling, apply 1x1 after
    # downsampling for efficiency.
    if height > target_height and width > target_width:
        if not conv_after_downsample:
            feat = _maybe_apply_1x1(feat, name=name)
        height_stride_size = int((height - 1) // target_height + 1)
        width_stride_size = int((width - 1) // target_width + 1)
        if pooling_type == 'max' or pooling_type is None:
            # Use max pooling in default.
            feat = tf.keras.layers.MaxPool2D(
                pool_size=[height_stride_size + 1, width_stride_size + 1],
                strides=[height_stride_size, width_stride_size],
                padding='SAME',
                data_format=data_format)(feat)
        elif pooling_type == 'avg':
            feat = tf.keras.layers.AveragePooling2D(
                pool_size=[height_stride_size + 1, width_stride_size + 1],
                strides=[height_stride_size, width_stride_size],
                padding='SAME',
                data_format=data_format)(feat)
        else:
            raise ValueError('Unknown pooling type: {}'.format(pooling_type))
        if conv_after_downsample:
            feat = _maybe_apply_1x1(feat, name=name+'_conv_after_downsample')
    elif height <= target_height and width <= target_width:
        feat = _maybe_apply_1x1(feat, name=name)
        if height < target_height or width < target_width:
            height_scale = target_height // height
            width_scale = target_width // width
            if (use_native_resize_op or target_height % height != 0 or
                    target_width % width != 0):
                if data_format == 'channels_first':
                    # feat = tf.transpose(feat, [0, 2, 3, 1])
                    feat = tf.keras.layers.Permute((2, 3, 1))(feat)
                tf_resize_layer = ImageResizeLayer(target_height, target_width)
                feat = tf_resize_layer(feat)
                if data_format == 'channels_first':
                    # feat = tf.transpose(feat, [0, 3, 1, 2])
                    feat = tf.keras.layers.Permute((3, 1, 2))(feat)
            else:
                feat = tf.keras.layers.UpSampling2D(
                    size=(height_scale, width_scale),
                    data_format=data_format)(feat)
    else:
        raise ValueError(
            'Incompatible target feature map size: target_height: {},'
            'target_width: {}'.format(target_height, target_width))

    return feat


def build_class_and_box_outputs(feats, config):
    """Builds box net and class net.

    Args:
     feats: input tensor.
     config: a dict-like config, including all parameters.

    Returns:
     A tuple (class_outputs, box_outputs) for class/box predictions.
    """

    class_outputs = {}
    box_outputs = {}
    num_anchors = len(config.aspect_ratios) * config.num_scales
    num_filters = config.fpn_num_filters
    class_outputs = ClassNet(num_classes=config.num_classes,
                             num_anchors=num_anchors,
                             num_filters=num_filters,
                             min_level=config.min_level,
                             max_level=config.max_level,
                             is_training=config.is_training_bn,
                             act_type=config.act_type,
                             repeats=config.box_class_repeats,
                             separable_conv=config.separable_conv,
                             survival_prob=config.survival_prob,
                             data_format=config.data_format)(feats)

    box_outputs = BoxNet(num_anchors=num_anchors,
                         num_filters=num_filters,
                         min_level=config.min_level,
                         max_level=config.max_level,
                         is_training=config.is_training_bn,
                         act_type=config.act_type,
                         repeats=config.box_class_repeats,
                         separable_conv=config.separable_conv,
                         survival_prob=config.survival_prob,
                         data_format=config.data_format)(feats)

    return class_outputs, box_outputs


def build_backbone(features, config):
    """Builds backbone model.

    Args:
     features: input tensor.
     config: config for backbone, such as is_training_bn and backbone name.

    Returns:
        A dict from levels to the feature maps from the output of the backbone model
        with strides of 8, 16 and 32.

    Raises:
        ValueError: if backbone_name is not supported.
    """
    backbone_name = config.name
    model_builder = backbone_factory.get_model_builder(backbone_name)
    # build tf efficientnet backbone from IVA templates
    u1, u2, u3, u4, u5 = model_builder.build_model_base(
        features, backbone_name,
        freeze_blocks=config.freeze_blocks,
        freeze_bn=config.freeze_bn)

    return {0: features, 1: u1, 2: u2, 3: u3, 4: u4, 5: u5}


def build_feature_network(features, config):
    """Build FPN input features.

    Args:
     features: input tensor.
     config: a dict-like config, including all parameters.

    Returns:
        A dict from levels to the feature maps processed after feature network.
    """
    feat_sizes = utils.get_feat_sizes(config.image_size, config.max_level)
    feats = []
    if config.min_level not in features.keys():
        raise ValueError('features.keys ({}) should include min_level ({})'.format(
            features.keys(), config.min_level))

    # Build additional input features that are not from backbone.
    for level in range(config.min_level, config.max_level + 1):
        if level in features.keys():
            feats.append(features[level])
        else:
            h_id, w_id = (2, 3) if config.data_format == 'channels_first' else (1, 2)
            # Adds a coarser level by downsampling the last feature map.
            feats.append(
                resample_feature_map(
                    feats[-1],
                    name='p%d' % level,
                    target_height=(feats[-1].shape[h_id] - 1) // 2 + 1,
                    target_width=(feats[-1].shape[w_id] - 1) // 2 + 1,
                    target_num_channels=config.fpn_num_filters,
                    apply_bn=config.apply_bn_for_resampling,
                    is_training=config.is_training_bn,
                    conv_after_downsample=config.conv_after_downsample,
                    use_native_resize_op=config.use_native_resize_op,
                    pooling_type=config.pooling_type,
                    data_format=config.data_format
                ))

    utils.verify_feats_size(
        feats,
        feat_sizes=feat_sizes,
        min_level=config.min_level,
        max_level=config.max_level,
        data_format=config.data_format)

    for rep in range(config.fpn_cell_repeats):
        new_feats = build_bifpn_layer(feats, feat_sizes, config, rep=str(rep))

        feats = [
            new_feats[level]
            for level in range(
                config.min_level, config.max_level + 1)
        ]

        utils.verify_feats_size(
            feats,
            feat_sizes=feat_sizes,
            min_level=config.min_level,
            max_level=config.max_level,
            data_format=config.data_format)

    return new_feats


def bifpn_sum_config():
    """BiFPN config with sum."""
    p = hparams_config.Config()
    p.nodes = [
        {'feat_level': 6, 'inputs_offsets': [3, 4]},
        {'feat_level': 5, 'inputs_offsets': [2, 5]},
        {'feat_level': 4, 'inputs_offsets': [1, 6]},
        {'feat_level': 3, 'inputs_offsets': [0, 7]},
        {'feat_level': 4, 'inputs_offsets': [1, 7, 8]},
        {'feat_level': 5, 'inputs_offsets': [2, 6, 9]},
        {'feat_level': 6, 'inputs_offsets': [3, 5, 10]},
        {'feat_level': 7, 'inputs_offsets': [4, 11]},
    ]
    p.weight_method = 'sum'
    return p


def bifpn_fa_config():
    """BiFPN config with fast weighted sum."""
    p = bifpn_sum_config()
    p.weight_method = 'fastattn'
    return p


def bifpn_dynamic_config(min_level, max_level, weight_method):
    """A dynamic bifpn config that can adapt to different min/max levels."""
    p = hparams_config.Config()
    p.weight_method = weight_method or 'fastattn'

    # Node id starts from the input features and monotonically increase whenever
    # a new node is added. Here is an example for level P3 - P7:
    #     P7 (4)              P7" (12)
    #     P6 (3)    P6' (5)   P6" (11)
    #     P5 (2)    P5' (6)   P5" (10)
    #     P4 (1)    P4' (7)   P4" (9)
    #     P3 (0)              P3" (8)
    # So output would be like:
    # [
    #   {'feat_level': 6, 'inputs_offsets': [3, 4]},  # for P6'
    #   {'feat_level': 5, 'inputs_offsets': [2, 5]},  # for P5'
    #   {'feat_level': 4, 'inputs_offsets': [1, 6]},  # for P4'
    #   {'feat_level': 3, 'inputs_offsets': [0, 7]},  # for P3"
    #   {'feat_level': 4, 'inputs_offsets': [1, 7, 8]},  # for P4"
    #   {'feat_level': 5, 'inputs_offsets': [2, 6, 9]},  # for P5"
    #   {'feat_level': 6, 'inputs_offsets': [3, 5, 10]},  # for P6"
    #   {'feat_level': 7, 'inputs_offsets': [4, 11]},  # for P7"
    # ]
    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}

    # level_last_id = lambda level: node_ids[level][-1]
    # level_all_ids = lambda level: node_ids[level]
    def level_last_id(level):
        return node_ids[level][-1]

    def level_all_ids(level):
        return node_ids[level]

    id_cnt = itertools.count(num_levels)

    p.nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        # top-down path.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': [level_last_id(i), level_last_id(i + 1)]
        })
        node_ids[i].append(next(id_cnt))

    for i in range(min_level + 1, max_level + 1):
        # bottom-up path.
        p.nodes.append({
            'feat_level': i,
            'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)]
        })
        node_ids[i].append(next(id_cnt))

    return p


def get_fpn_config(fpn_name, min_level, max_level, weight_method):
    """Get fpn related configuration."""
    if not fpn_name:
        fpn_name = 'bifpn_fa'
    name_to_config = {
        'bifpn_sum': bifpn_sum_config(),
        'bifpn_fa': bifpn_fa_config(),
        'bifpn_dyn': bifpn_dynamic_config(min_level, max_level, weight_method)
    }
    return name_to_config[fpn_name]


def build_bifpn_layer(feats, feat_sizes, config, rep='0'):
    """Builds a feature pyramid given previous feature pyramid and config."""
    p = config  # use p to denote the network config.
    if p.fpn_config:
        fpn_config = p.fpn_config
    else:
        fpn_config = get_fpn_config(
            p.fpn_name, p.min_level, p.max_level, p.fpn_weight_method)

    num_output_connections = [0 for _ in feats]
    for i, fnode in enumerate(fpn_config.nodes):
        # with tf.variable_scope('fnode{}'.format(i)):
        # logging.info('fnode %d : %s', i, fnode)
        new_node_height = feat_sizes[fnode['feat_level']]['height']
        new_node_width = feat_sizes[fnode['feat_level']]['width']
        nodes = []
        for idx, input_offset in enumerate(fnode['inputs_offsets']):
            input_node = feats[input_offset]
            num_output_connections[input_offset] += 1
            input_node = resample_feature_map(
                input_node, 'bifpn{}_{}_{}_{}'.format(idx, input_offset, len(feats), rep),
                new_node_height, new_node_width, p.fpn_num_filters,
                p.apply_bn_for_resampling, p.is_training_bn,
                p.conv_after_downsample,
                p.use_native_resize_op,
                p.pooling_type,
                data_format=config.data_format)
            nodes.append(input_node)

        # new_node = fuse_features(nodes, fpn_config.weight_method)
        new_node = WeightedFusion(name='weighted_fusion_{}_{}'.format(i, rep))(nodes)

        # with tf.variable_scope('op_after_combine{}'.format(len(feats))):
        if not p.conv_bn_act_pattern:
            new_node = utils.activation_fn(new_node, p.act_type)

        if p.separable_conv:
            conv_layer = tf.keras.layers.SeparableConv2D(
                depth_multiplier=1,
                filters=p.fpn_num_filters,
                kernel_size=(3, 3),
                padding='same',
                data_format=config.data_format,
                name='after_combine_dw_conv_{}_{}'.format(i, rep))
        else:
            conv_layer = tf.keras.layers.Conv2D(
                filters=p.fpn_num_filters,
                kernel_size=(3, 3),
                padding='same',
                data_format=config.data_format,
                name='after_combine_conv_{}_{}'.format(i, rep))

        new_node = conv_layer(new_node)

        new_node = utils.batch_norm_act(
            new_node,
            is_training_bn=p.is_training_bn,
            act_type=None if not p.conv_bn_act_pattern else p.act_type,
            data_format=config.data_format,
            name='bifpn_bn_{}_{}'.format(i, rep))

        feats.append(new_node)
        num_output_connections.append(0)

    output_feats = {}
    for l in range(p.min_level, p.max_level + 1):
        for i, fnode in enumerate(reversed(fpn_config.nodes)):
            if fnode['feat_level'] == l:
                output_feats[l] = feats[-1 - i]
                break
    return output_feats


def efficientdet(inputs, model_name=None, config=None, **kwargs):
    """Build EfficientDet model."""
    if not config and not model_name:
        raise ValueError('please specify either model name or config')

    if not config:
        config = hparams_config.get_efficientdet_config(model_name)
    elif isinstance(config, dict):
        config = hparams_config.Config(config)  # wrap dict in Config object

    if kwargs:
        config.override(kwargs)

    if config.pruned_model_path:
        input_layer = tf.keras.layers.InputLayer(input_tensor=inputs, name="Input")
        if config.mode == 'export':
            pruned_model = get_model_with_input(
                os.path.join(os.path.dirname(config.pruned_model_path), 'pruned_eval.json'),
                input_layer)
        else:
            pruned_model = get_model_with_input(
                os.path.join(os.path.dirname(config.pruned_model_path), 'pruned_train.json'),
                input_layer)
        if (not MPI_is_distributed() or MPI_rank() == 0) and config.mode == 'train':
            pruned_model.summary()
            print("Pruned graph is loaded succesfully.")
        model_outputs = pruned_model.outputs

        lvl_list = list(range(config.min_level, config.max_level+1))
        class_outputs = dict(zip(lvl_list, model_outputs[0:len(lvl_list)]))
        box_outputs = dict(zip(lvl_list, model_outputs[len(lvl_list):]))
    else:
        # build backbone features.
        features = build_backbone(inputs, config)
        # build feature network.
        fpn_feats = build_feature_network(features, config)
        # build class and box predictions.
        class_outputs, box_outputs = build_class_and_box_outputs(fpn_feats, config)

        m = tf.keras.models.Model(
            inputs=inputs,
            outputs=list(class_outputs.values())+list(box_outputs.values()))
        if (not MPI_is_distributed() or MPI_rank() == 0) and config.mode == 'train':
            dump_json(m, os.path.join(config.model_dir, "graph.json"))

    return class_outputs, box_outputs
