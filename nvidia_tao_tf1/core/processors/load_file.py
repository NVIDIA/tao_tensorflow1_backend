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

"""Load File Processor."""

import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.processors import Processor


class LoadFile(Processor):
    """Load and read a file from an input string.

    Args:
        prefix (str): Optional prefix to be added to the input filename.
        suffix (str): Optional suffix to be added to the input filename.
        kwargs (dict): keyword arguments passed to parent class.
    """

    @save_args
    def __init__(self, prefix=None, suffix=None, **kwargs):
        """__init__ method."""
        self.prefix = prefix
        self.suffix = suffix
        super(LoadFile, self).__init__(**kwargs)

    def call(self, filename):
        """call method.

        Args:
            filename (tensor): Tensor (string) containing the filename to be loaded. The filename
                can be joined with an optional `prefix` and `suffix` as supplied with this layers
                creation.
        Returns:
            tensor: File contents as loaded from `filename`.
        """
        if self.prefix and self.suffix:
            filename = tf.strings.join([self.prefix, filename, self.suffix])
        elif self.prefix:
            filename = tf.strings.join([self.prefix, filename])
        elif self.suffix:
            filename = tf.strings.join([filename, self.suffix])
        return tf.io.read_file(filename)
