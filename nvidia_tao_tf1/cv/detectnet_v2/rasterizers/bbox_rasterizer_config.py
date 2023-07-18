# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Bbox rasterizer config class that holds parameters for BboxRasterizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class BboxRasterizerConfig(dict):
    """Hold the parameters for BboxRasterizer."""

    class TargetClassConfig(object):
        """Hold target class specific parameters."""

        __slots__ = ["cov_center_x", "cov_center_y", "cov_radius_x", "cov_radius_y",
                     "bbox_min_radius"]

        def __init__(self, cov_center_x, cov_center_y, cov_radius_x, cov_radius_y, bbox_min_radius):
            """Constructor.

            Args:
                cov_center_x/y (float): The x / y coordinate of the center of the coverage region
                    relative to the bbox. E.g. If we want the center of the coverage region to be
                    that of the bbox, the value would be 0.5.
                cov_radius_x/y (float): The radius of the coverage region along the x / y axis,
                    relative to the full extent of the bbox. E.g. If we want the coverage region
                    to span the entire length of a bbox along a given axis, the value would be 1.0.
                bbox_min_radius (float): Minimum radius of the coverage region in output space (not
                    input pixel space).

            Raises:
                ValueError: If the input args are not in the accepted ranges.
            """
            if cov_center_x < 0.0 or cov_center_x > 1.0:
                raise ValueError("BboxRasterizerConfig.TargetClassConfig.cov_center_x must be in "
                                 "[0.0, 1.0]")
            if cov_center_y < 0.0 or cov_center_y > 1.0:
                raise ValueError("BboxRasterizerConfig.TargetClassConfig.cov_center_y must be in "
                                 "[0.0, 1.0]")
            if cov_radius_x <= 0.0:
                raise ValueError("BboxRasterizerConfig.TargetClassConfig.cov_radius_x must be > 0")
            if cov_radius_y <= 0.0:
                raise ValueError("BboxRasterizerConfig.TargetClassConfig.cov_radius_y must be > 0")
            if bbox_min_radius <= 0.0:
                raise ValueError("BboxRasterizerConfig.TargetClassConfig.bbox_min_radius "
                                 "must be > 0")
            self.cov_center_x = cov_center_x
            self.cov_center_y = cov_center_y
            self.cov_radius_x = cov_radius_x
            self.cov_radius_y = cov_radius_y
            self.bbox_min_radius = bbox_min_radius

    def __init__(self, deadzone_radius):
        """Constructor.

        Args:
            deadzone_radius (float): Radius of the deadzone to be drawn in between overlapping
                coverage regions.

        Raises:
            ValueError: If the input arg is not within the accepted range.
        """
        if deadzone_radius < 0.0 or deadzone_radius > 1.0:
            raise ValueError("BboxRasterizerConfig.deadzone_radius must be in [0.0, 1.0]")
        self.deadzone_radius = deadzone_radius
