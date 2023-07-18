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
"""EfficientDet hparam config tests."""

from nvidia_tao_tf1.cv.efficientdet.utils import hparams_config


def test_hparams_config():
    c = hparams_config.Config({'a': 1, 'b': 2})
    assert c.as_dict() == {'a': 1, 'b': 2}

    c.update({'a': 10})
    assert c.as_dict() == {'a': 10, 'b': 2}

    c.b = 20
    assert c.as_dict() == {'a': 10, 'b': 20}

    c.override('a=true,b=ss')
    assert c.as_dict() == {'a': True, 'b': 'ss'}

    c.override('a=100,,,b=2.3,')  # extra ',' is fine.
    assert c.as_dict() == {'a': 100, 'b': 2.3}

    c.override('a=2x3,b=50')  # a is a special format for image size.
    assert c.as_dict() == {'a': '2x3', 'b': 50}
