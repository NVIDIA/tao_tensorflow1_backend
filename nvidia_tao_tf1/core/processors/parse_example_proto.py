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

"""Parse Example Proto Processor."""

import tensorflow as tf
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.processors import Processor


class ParseExampleProto(Processor):
    """Parse and deserialize an example proto to a dictionary with tensor values.

    Args:
        features (dict): a dictionary with strings as keys and feature-tensors as values.
            The keys should relate to those in the proto, and the values should be of type
            `tf.VarLenFeature` or `tf.FixedLenFeature`.
        single (bool): indicates whether we're parsing a single example, or a batch of examples.
        kwargs (dict): keyword arguments passed to parent class.
    """

    @save_args
    def __init__(self, features, single, **kwargs):
        """"__init__ method."""
        self.features = features
        self.single = single
        super(ParseExampleProto, self).__init__(**kwargs)

    def call(self, serialized):
        """call method.

        Note: only `values` will be extracted from `tf.VarLenFeature` outputs. Therefore this
        method might not be ideally compatible with purely sparse tensors.

        Args:
            serialized (tensor): a serialized example proto.
        Returns:
            dict: a dict of tensors with the same keys as the `features` dict, and dense tensor
                values as extracted from the example proto's relating key value.
        """
        if self.single:
            example = tf.io.parse_single_example(
                serialized=serialized, features=self.features
            )
        else:
            example = tf.io.parse_example(serialized=serialized, features=self.features)
        for key, value in example.items():
            if isinstance(value, tf.SparseTensor):
                default_value = "" if value.dtype == tf.string else 0
                example[key] = tf.sparse.to_dense(value, default_value=default_value)
                if not self.single:
                    # If known, retain the shape of the batch.
                    shape = example[key].get_shape().as_list()
                    shape[0] = serialized.get_shape()[0]
                    example[key].set_shape(shape)
        return example
