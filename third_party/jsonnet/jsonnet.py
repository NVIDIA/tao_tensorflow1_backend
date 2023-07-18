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

"""Some helpers for using jsonnet within ai-infra."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import _jsonnet
from google.protobuf import json_format
from google.protobuf import text_format

_PROTO_EXTENSIONS = [".txt", ".prototxt"]


def _import_callback(proto_constructor=None):
    def callback(owning_directory, path):
        """Reads a file from disk.

        Args:
            unused (?): Not used, but required to pass the callback to jsonnet.
            path (str): The path that is being imported.

        Returns:
            (str, str): The full path and contents of the file.
        """
        # This enables both relative and absolute pathing. First, check to see if the absolute
        # path exists, at least relative to the working directory. If it doesn't, then prepend
        # the path of the directory of the currently executing file
        if not os.path.exists(path):
            path = os.path.join(owning_directory, path)
        with open(path, "r") as infile:
            data = infile.read()

            if proto_constructor and any(
                path.endswith(ext) for ext in _PROTO_EXTENSIONS
            ):
                proto = proto_constructor()
                text_format.Merge(data, proto)
                data = json_format.MessageToJson(proto)

            return path, data

    return callback


def evaluate_file(path, ext_vars=None, proto_constructor=None):
    """Evaluates a jsonnet file.

    If a proto_constructor is given, then any `import` statements in the jsonnet file can import
    text protos; and the returned value will be an object of that proto type.

    Args:
        path (str): Path to the jsonnet file, relative to ai-infra root.
        ext_vars (dict): Variables to pass to the jsonnet script.
        proto_constructor (callable): Callable used for converting to protos.

    Returns:
        (str|protobuf): If a proto constructor is provided, the jsonnet file is evaluated as a text
            proto; otherwise it is evaluated as a JSON string.
    """
    data = _jsonnet.evaluate_file(
        path, ext_vars=ext_vars, import_callback=_import_callback(proto_constructor)
    )

    if proto_constructor:
        proto = proto_constructor()
        json_format.Parse(data, proto)
        return proto

    return data
