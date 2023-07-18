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
"""Pybind Test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import pytest
import sys


def test_import_and_use():
    import third_party.pybind11.py_bind_test_lib as py_bind_test_lib

    assert 3 == py_bind_test_lib.add(5, -2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
