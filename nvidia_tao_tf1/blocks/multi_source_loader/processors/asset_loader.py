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
"""Processor that loads assets referenced in Examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nvidia_tao_tf1.core import processors


class AssetLoader(processors.Processor):
    """Processor that loads assets referenced in Examples."""

    def __init__(self, output_dtype=tf.float32, **kwargs):
        """Construct a processor that loads referenced assets.

        Args:
            output_dtype (tf.dtypes.DType): Output image dtype. Defaults to tf.float32.
        """
        super(AssetLoader, self).__init__(**kwargs)
        self._output_dtype = output_dtype

    def call(self, example):
        """Load all referenced assets recursively.

        Args:
            example (SequenceExample): Example composing of geometric primitives. All primitives
                with a `load` method will be replaced by the return value of the load method.

        Returns:
            (SequenceExample): Example with all references to assets replaced with the actual
                assets.
        """
        return self._load_features(example)

    def _load_features(self, example):
        # TODO(vkallioniemi): This functionality is mostly the same with the recursion
        # code in TransformedExample - extract into a separate method.
        def _load_recursive(value):
            def _is_namedtuple(value):
                """Return true if value is a namedtuple."""
                return isinstance(value, tuple) and hasattr(value, "_fields")

            # Call load only if implemented on a namedtuple
            load_op = getattr(value, "load", None)
            if _is_namedtuple(value) and (load_op is not None and callable(load_op)):
                return load_op(self._output_dtype)
            if isinstance(value, (list, set)):
                return [_load_recursive(v) for v in value]
            if isinstance(value, dict):
                return {k: _load_recursive(v) for (k, v) in value.items()}
            if _is_namedtuple(value):
                return value._make(
                    [_load_recursive(field) for field in value._asdict().values()]
                )
            # Stop recursion - unknown non-collection types are treated as leaf nodes.
            return value

        return _load_recursive(example)
