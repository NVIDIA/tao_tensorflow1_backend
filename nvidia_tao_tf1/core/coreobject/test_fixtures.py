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


from nvidia_tao_tf1.core.coreobject import (
    register_tao_function,
    save_args,
    TAOObject
)

def get_duplicated_tao_object_child_class():
    # This class is a copy of the one in test_modulusobject.py. It is here so that we can test
    # the handling of name collisions at deserialization time.
    class TAOObjectChild(TAOObject):
        @save_args
        def __init__(
            self, arg1, arg2="default_arg2_val", *args, **kwargs
        ):  # pylint: disable=W1113
            super(TAOObjectChild, self).__init__(*args, **kwargs)
            self.arg1 = arg1
            self.arg2 = arg2

    return TAOObjectChild


MY_RETURN_VALUE = "takes skill to be real"


@register_tao_function
def my_test_function():
    # Used for testing serialization of functions.
    return MY_RETURN_VALUE
