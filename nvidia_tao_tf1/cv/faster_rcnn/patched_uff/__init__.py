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
"""Patch UFF only for FasterRCNN.

The background of this patch deserves some explanation. In FasterRCNN models, when exported to
UFF model, the Softmax layer has some issue in UFF converter and cannot be converted successfully.
In the official UFF converter functions, Softmax layer defaults to use axis=0 and
data format = NHWC, i.e, 'N+C', in UFF's notation. While this default settings doesn't work
for FasterRCNN due to the 5D tensor in it. If we dig it deeper, we will found this is due to
some transpose is applied during parsing the Softmax layer and that transpose operation doesn't
support 5D yet in UFF. So to walk around this, we have comed up with a solution. That is, force
the default data format to be NCHW for FasterRCNN model. However, this hack will break the unit
test 'test_3d_softmax' in 'nvdia_tao_tf1/core/export/test_export.py' and the reason is unclear.
Again, to avoid breaking this unit test, we would like to apply this patch not globally,
but rather for FasterRCNN only. For other parts of this repo, they will see the unpatched
UFF package and hence everything goes like before.

The converter_functions.py is copied from the UFF package with some changes of the code in it.
the code style in it is not conforming to the TLT standard. But to avoid confusion, we would
not change the code format in it, and hence we prefer to drop static tests for it in the BUILD file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uff as patched_uff

from nvidia_tao_tf1.cv.faster_rcnn.patched_uff import converter_functions


patched_uff.converters.tensorflow.converter_functions = converter_functions
