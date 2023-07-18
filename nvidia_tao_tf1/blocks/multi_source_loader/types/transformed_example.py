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
"""Example whose transformation has been delayed."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple


class TransformedExample(
    namedtuple("TransformedExample", ["transformation", "example"])
):
    """Container for an example and the transformation that can be applied to it.

    Args:
        transformation (Transformation): Transformation which will be applied by __call__.
        example (SequenceExample): Example that the transformation applies to.
    """

    def __call__(self, **kwargs):
        """Return original type after applying transformations."""
        return self._apply_recursive(self.example, self.transformation, **kwargs)

    def _apply_recursive(self, value, transformation, **kwargs):
        """
        Apply transformations to tf.data.Dataset compatible value.

        * Transformation will be applied recursively to members of container types
          (dict, list, namedtuple).
        * Transformations are applied to types that have an apply method.
        """

        def _is_namedtuple(value):
            """Return true if value is a namedtuple."""
            return isinstance(value, tuple) and hasattr(value, "_fields")

        # Call apply only if implemented on a namedtuple
        apply_op = getattr(value, "apply", None)
        if _is_namedtuple(value) and (apply_op is not None and callable(apply_op)):
            return apply_op(transformation, **kwargs)
        if isinstance(value, (list, set)):
            return [self._apply_recursive(v, transformation, **kwargs) for v in value]
        if isinstance(value, dict):
            return {
                k: self._apply_recursive(v, transformation, **kwargs)
                for (k, v) in value.items()
            }
        if _is_namedtuple(value):
            return value._make(
                [
                    self._apply_recursive(field, transformation, **kwargs)
                    for field in value._asdict().values()
                ]
            )
        # Stop recursion - unknown non-collection types are treated as leaf nodes.
        return value
