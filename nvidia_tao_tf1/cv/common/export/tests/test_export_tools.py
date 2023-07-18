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

"""Simple script to test export tools(tensorrt, uff, graphsurgeon)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import graphsurgeon as gs
import tensorrt as trt
import uff


# Check gs has the create_plugin_node method
def test_gs_create_plugin_node():
    n = gs.create_plugin_node(name='roi_pooling_conv_1/CropAndResize_new',
                              op="CropAndResize",
                              inputs=['activation_13/Relu', 'proposal'],
                              crop_height=7,
                              crop_width=7)
    assert n


# Check the TRT version
def test_trt_version():
    assert trt.__version__ == '8.5.1.7'


# Check the UFF version
def test_uff_version():
    assert uff.__version__ == '0.6.7'


# Check the gs version
def test_gs_version():
    assert gs.__version__ == '0.4.1'
