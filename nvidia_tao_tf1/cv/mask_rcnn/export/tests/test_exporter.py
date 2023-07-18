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
"""Test exporter."""
import os
from google.protobuf.json_format import MessageToDict
import pytest

from nvidia_tao_tf1.cv.mask_rcnn.utils.spec_loader import load_experiment_spec


@pytest.fixture
def generate_params():
    """Get MRCNN default params."""
    file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_spec_path = os.path.join(file_path, '../experiment_specs/default.txt')
    experiment_spec = load_experiment_spec(default_spec_path, merge_from_default=False)
    default_params = MessageToDict(experiment_spec,
                                   preserving_proto_field_name=True,
                                   including_default_value_fields=True)
    try:
        data_config = default_params['data_config']
        maskrcnn_config = default_params['maskrcnn_config']
    except ValueError:
        print("Make sure data_config and maskrcnn_config are configured properly.")
    finally:
        del default_params['data_config']
        del default_params['maskrcnn_config']
    default_params.update(data_config)
    default_params.update(maskrcnn_config)
    default_params['aspect_ratios'] = eval(default_params['aspect_ratios'])
    default_params['freeze_blocks'] = eval(default_params['freeze_blocks'])
    default_params['bbox_reg_weights'] = eval(default_params['bbox_reg_weights'])
    # default_params['image_size'] = (hh, ww)
    default_params['use_batched_nms'] = True
    # default_params['nlayers'] = nlayers
    return default_params


def test_exporter():
    """Test exporter."""
    pass
