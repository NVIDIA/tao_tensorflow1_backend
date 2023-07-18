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

"""Tests for jsonnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from third_party import jsonnet
from third_party.jsonnet import fake_pb2


def test_jsonnet():
    result = jsonnet.evaluate_file("third_party/jsonnet/testdata/template.jsonnet")

    assert json.loads(result)["foo"] == "bar"


def test_relative_import():
    result = jsonnet.evaluate_file(
        "third_party/jsonnet/testdata/relimport_template.jsonnet"
    )

    assert json.loads(result)["foo"] == "bar"


def test_jsonnet_with_proto():
    result = jsonnet.evaluate_file(
        "third_party/jsonnet/testdata/template.jsonnet",
        proto_constructor=fake_pb2.FakeMessage,
    )

    assert result.foo == "bar"


def test_external_variable():
    result = jsonnet.evaluate_file(
        "third_party/jsonnet/testdata/ext_var.jsonnet",
        ext_vars={"external_value": "hello world"},
    )

    assert json.loads(result)["the_val"] == "hello world"
