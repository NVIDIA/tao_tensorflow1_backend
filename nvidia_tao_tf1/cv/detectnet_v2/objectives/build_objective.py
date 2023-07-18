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

"""Objective builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.cv.detectnet_v2.objectives.bbox_objective import BboxObjective
from nvidia_tao_tf1.cv.detectnet_v2.objectives.cov_norm_objective import CovNormObjective
from nvidia_tao_tf1.cv.detectnet_v2.objectives.cov_objective import CovObjective


def build_objective(name, output_height, output_width, input_height, input_width, objective_config):
    """Construct objective of desired type.

    Args:
        name (str): objective name
        output_* (float): output tensor shape
        input_* (float): input tensor shape
        objective_config: Objective configuration proto
    """
    if objective_config:
        input_layer_name = objective_config.input
    else:
        input_layer_name = None

    if name == 'bbox':
        scale = objective_config.scale
        offset = objective_config.offset
        objective = BboxObjective(input_layer_name, output_height, output_width,
                                  input_height, input_width, scale, offset, loss_ratios=None)
    elif name == 'cov':
        objective = CovObjective(input_layer_name, output_height, output_width)
    elif name == 'cov_norm':
        objective = CovNormObjective(
            input_layer_name, output_height, output_width)
    else:
        raise ValueError("Unknown objective: %s" % name)

    return objective
