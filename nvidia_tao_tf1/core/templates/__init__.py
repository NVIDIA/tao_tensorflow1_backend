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

"""TAO Toolkit model templates."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.utils import get_custom_objects

from nvidia_tao_tf1.core.templates import alexnet
from nvidia_tao_tf1.core.templates import googlenet
from nvidia_tao_tf1.core.templates import mobilenet
from nvidia_tao_tf1.core.templates import resnet
from nvidia_tao_tf1.core.templates import squeezenet
from nvidia_tao_tf1.core.templates import utils
from nvidia_tao_tf1.core.templates import vgg
from nvidia_tao_tf1.core.templates.utils import swish

__all__ = ('alexnet', 'googlenet', 'mobilenet', 'resnet', 'squeezenet', 'utils', 'vgg')

get_custom_objects()["swish"] = swish
