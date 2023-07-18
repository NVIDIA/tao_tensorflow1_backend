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
"""test lprnet model builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.cv.lprnet.utils.ctc_decoder import decode_ctc_conf


def test_ctc_decoder():
    classes = ["a", "b", "c"]
    blank_id = 3

    pred_id = [[0, 0, 0, 3, 1, 2]]
    pred_conf = [[0.99, 0.99, 0.99, 0.99, 0.99, 0.99]]
    expected_lp = "abc"
    decoded_lp, _ = decode_ctc_conf((pred_id, pred_conf), classes, blank_id)

    assert expected_lp == decoded_lp[0]
