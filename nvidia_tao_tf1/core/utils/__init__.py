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
"""TAO Core utils APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.core.utils.utils import find_latest_keras_model_in_directory
from nvidia_tao_tf1.core.utils.utils import get_all_simple_values_from_event_file
from nvidia_tao_tf1.core.utils.utils import get_uid
from nvidia_tao_tf1.core.utils.utils import get_uid_name
from nvidia_tao_tf1.core.utils.utils import mkdir_p
from nvidia_tao_tf1.core.utils.utils import recursive_map_dict
from nvidia_tao_tf1.core.utils.utils import set_random_seed
from nvidia_tao_tf1.core.utils.utils import summary_from_value
from nvidia_tao_tf1.core.utils.utils import test_session
from nvidia_tao_tf1.core.utils.utils import to_camel_case
from nvidia_tao_tf1.core.utils.utils import to_snake_case

__all__ = (
    "find_latest_keras_model_in_directory",
    "get_all_simple_values_from_event_file",
    "get_uid",
    "get_uid_name",
    "mkdir_p",
    "recursive_map_dict",
    "set_random_seed",
    "summary_from_value",
    "test_session",
    "to_camel_case",
    "to_snake_case",
)
