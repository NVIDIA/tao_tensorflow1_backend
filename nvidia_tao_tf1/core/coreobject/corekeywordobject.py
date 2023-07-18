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
"""Object that takes in a variable amount of arguments and sets its attributes accordingly."""

from nvidia_tao_tf1.core.coreobject.coreobject import TAOObject, save_args


class TAOKeywordObject(TAOObject):
    """Class that takes kwargs into its __init__ method and maps them to its attributes."""

    @save_args
    def __init__(self, **kwargs):
        """Init the object with a variable amount of arguments which are mapped to attrs."""
        super(TAOKeywordObject, self).__init__()
        self.__dict__.update(kwargs)
