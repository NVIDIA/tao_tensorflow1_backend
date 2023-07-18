# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Build for the BboxRasterizerConfig."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from nvidia_tao_tf1.cv.detectnet_v2.rasterizers.bbox_rasterizer_config import BboxRasterizerConfig


def build_bbox_rasterizer_config(bbox_rasterizer_proto):
    """Build BboxRasterizerConfig from a proto.

    Args:
        bbox_rasterizer_proto: proto.bbox_rasterizer_config.BboxRasterizerConfig message.

    Returns:
        bbox_rasterizer_config: BboxRasterizerConfig instance.
    """
    bbox_rasterizer_config = BboxRasterizerConfig(bbox_rasterizer_proto.deadzone_radius)

    for target_class_name, target_class_config in \
            six.iteritems(bbox_rasterizer_proto.target_class_config):
        bbox_rasterizer_config[target_class_name] = \
            BboxRasterizerConfig.TargetClassConfig(target_class_config.cov_center_x,
                                                   target_class_config.cov_center_y,
                                                   target_class_config.cov_radius_x,
                                                   target_class_config.cov_radius_y,
                                                   target_class_config.bbox_min_radius)

    return bbox_rasterizer_config
