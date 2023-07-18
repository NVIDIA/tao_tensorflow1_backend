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

"""Lookup Table Processor."""

import tensorflow as tf
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.processors import Processor


class LookupTable(Processor):
    """Create a lookup table (LUT) to relate keys to values, or uses a default value.

    Args:
        keys (list): list of keys, with the same length as ``values``.
        values (list):  list of values, with the same length as ``keys``.
        default_value: the default value to be used when a key is not present in the ``keys`` list.
        kwargs (dict): keyword arguments passed to parent class.

    Raises:
        ValueError: if ``keys`` or ``values`` are not of type ``list``, or if length of ``keys`` do
            not match length of ``values``.
    """

    @save_args
    def __init__(self, keys, values, default_value, **kwargs):
        """__init__ method."""
        self.keys = keys
        self.values = values
        self.default_value = default_value

        if type(keys) != list:
            raise TypeError('"keys" is not of type "list"')

        if type(values) != list:
            raise TypeError('"values" is not of type "list"')

        nkeys, nvalues = len(keys), len(values)
        if nkeys != nvalues:
            raise ValueError(
                'keys/values list discrepancy: received %d "keys" and %d "values".'
                % (nkeys, nvalues)
            )

        # Only possible LUT key types are string and int64, problem occurs when list
        # of 'ints' is passed in, as that one is by default converted to list of tf.int32.
        # Need to do the type specification explicitly!
        self.key_dtype = None
        if type(keys[0]) is str:
            self.key_dtype = tf.string
        elif type(keys[0]) is int:
            self.key_dtype = tf.int64

        super(LookupTable, self).__init__(**kwargs)

    def _build(self, *args, **kwargs):
        """Build and initialize the LUT."""
        self._table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                self.keys, self.values, key_dtype=self.key_dtype
            ),
            self.default_value,
        )

    def call(self, key):
        """call method.

        Args:
            key (tensor): input key to be related to a value in ``values`` through the LUT.
        Returns:
            tensor: mapped tensor as `x` relates to a `value` in the LUT, or uses the
                ``default_value``.
        """
        return self._table.lookup(key)
