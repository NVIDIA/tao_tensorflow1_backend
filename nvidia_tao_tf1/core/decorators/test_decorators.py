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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

from nvidia_tao_tf1.core.decorators import override, subclass


class Foo(object):
    """Dummy class to test @override decorator."""

    def bar(self):
        pass


class TestOverride(object):
    """Test that @override decorator works as expected."""

    def test_invalid_override(self):
        # pylint: disable=unused-variable
        with pytest.raises(AssertionError):

            @subclass
            class FooChild(Foo):
                @override
                def barz(self):
                    pass

    def test_valid_override(self):
        # pylint: disable=unused-variable
        @subclass
        class FooChild(Foo):
            @override
            def bar(self):
                pass
