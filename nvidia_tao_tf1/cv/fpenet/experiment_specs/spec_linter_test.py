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

"""Tests for FpeNet experiment specs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
from mock import patch
import yaml

import nvidia_tao_tf1.core as tao_core


def test_spec_deserialization():
    """Test whether all experiment specs can deserialize correctly into MagLev objects."""
    spec_paths = glob.glob("nvidia_tao_tf1/cv/fpenet/experiment_specs/*.yaml")
    for spec_path in spec_paths:
        with open(spec_path, 'r') as f:
            spec = yaml.load(f.read())
            print(spec)
            trainer = tao_core.coreobject.deserialize_tao_object(spec)
            # Type of trainer object should be included in the '__class_name__' field of the spec.
            assert type(trainer).__name__ in spec['__class_name__']
