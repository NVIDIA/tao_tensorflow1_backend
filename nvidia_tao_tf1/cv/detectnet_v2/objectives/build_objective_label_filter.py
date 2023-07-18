# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""ObjectiveLabelFilter class builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.cv.detectnet_v2.label_filter.build_label_filter import build_label_filter
from nvidia_tao_tf1.cv.detectnet_v2.objectives.objective_label_filter import ObjectiveLabelFilter
from nvidia_tao_tf1.cv.detectnet_v2.objectives.objective_label_filter_config import (
    ObjectiveLabelFilterConfig
)


def build_objective_label_filter_config(objective_label_filter_config_proto):
    """Build a ObjectiveLabelFilterConfig from proto.

    Args:
        objective_label_filter_config_proto:
            proto.objective_label_filter.ObjectiveLabelFilter.ObjectiveLabelFilterConfig message.

    Returns:
        objective_label_filter_config (ObjectiveLabelFilterConfig).
    """
    label_filter = build_label_filter(
        objective_label_filter_config_proto.label_filter)

    if not objective_label_filter_config_proto.target_class_names:
        target_class_names = None
    else:
        target_class_names = objective_label_filter_config_proto.target_class_names

    if not objective_label_filter_config_proto.objective_names:
        objective_names = None
    else:
        objective_names = objective_label_filter_config_proto.objective_names

    return ObjectiveLabelFilterConfig(
        label_filter=label_filter,
        objective_names=objective_names,
        target_class_names=target_class_names
    )


def build_objective_label_filter(objective_label_filter_proto,
                                 target_class_to_source_classes_mapping,
                                 learnable_objective_names):
    """Build a ObjectiveLabelFilter.

    Args:
        objective_label_filter_proto: proto.objective_label_filter.ObjectiveLabelFilter message.
        target_class_to_source_classes_mapping (dict): maps from target class name to a list of
            source class names.
        learnable_objective_names (list of str): List of learnable objective names. These are
            the objective names a LabelFilter will be applied to if the ObjectiveLabelFilterConfig
            containing it has objective_names set to <NoneType>.

    Returns:
        objective_label_filter (ObjectiveLabelFilter).
    """
    objective_label_filter_configs = \
        [build_objective_label_filter_config(
            con_temp) for con_temp in objective_label_filter_proto.objective_label_filter_configs]

    mask_multiplier = objective_label_filter_proto.mask_multiplier
    preserve_ground_truth = objective_label_filter_proto.preserve_ground_truth

    return ObjectiveLabelFilter(
        objective_label_filter_configs=objective_label_filter_configs,
        target_class_to_source_classes_mapping=target_class_to_source_classes_mapping,
        learnable_objective_names=learnable_objective_names,
        mask_multiplier=mask_multiplier,
        preserve_ground_truth=preserve_ground_truth)
