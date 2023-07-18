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
"""Processor that forms temporal batches."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from nvidia_tao_tf1.core import processors
from nvidia_tao_tf1.blocks.multi_source_loader import types


# This is mostly copy-pasted code from SlidingWindowSequence in Modulus. The biggest change is
# that it uses the types.Session structure for session information.
class TemporalBatcher(processors.Processor):
    """Sliding window dataset with frames in the same sequence.

    Takes in a dataset, slides a window over it and returns the result if all elements in the
    current window come from the same sequence.

    NOTE: If the input dataset is not ordered by (1) session_uuid (2) frame_number
    The sliding window sequence will not work correctly, and might return an empty dataset as
    it could be unable to find a sequence where the session_uuids match.
    Furthermore, it does not assert for frame_number order or striding at all, so the sequence of
    frames (when the session_uuid and camera_name are unique) will be the frame-order that was
    presented to it.
    """

    def __init__(self, size=1, **kwargs):
        """
        Construct a temporal batcher.

        Args:
            size (int): Length of the sequence expressed in number of timesteps.
            shift (int): Shift between multiple windows.
            stride (int): Stride between timesteps within a window.
        """
        if size < 1:
            raise ValueError(
                "TemporalBatcher.size must be a positive number, not: {}".format(size)
            )
        self.size = size
        super(TemporalBatcher, self).__init__(**kwargs)

    def __repr__(self):
        """Return string representation of this processor."""
        return "TemporalBatcher(size={})".format(self.size)

    @staticmethod
    def predicate(example):
        """Predicate function that determines if the current input should be considered a sequence.

        Args:
            example (SequenceExample): The ``Example`` namedtuple containing the tensors.

        Returns:
            A tf.bool dependent on whether the current input samples are in sequence.
                It returns True only of the ``session_uuid`` inside the input sequence
                are identical.
        """

        def _all_elements_identical(tensor):
            unique, _ = tf.unique(tensor)
            n_unique = tf.shape(input=unique)[0]
            return tf.equal(n_unique, 1)

        if types.FEATURE_SESSION not in example.instances:
            raise ValueError(
                "FEATURE_SESSION is required for temporal batching but is not present "
                "in example.instances."
            )

        # TODO(vkallioniemi): Add an assertion and/or make this more robust. The current
        # implementation relies on the sequence_extender script to ensure that datasets are
        # aligned at sequence boundaries.
        return _all_elements_identical(example.instances[types.FEATURE_SESSION].uuid)

    def call(self, dataset):
        """Process dataset by grouping consecutive frames into sequences.

        Args:
            dataset (tf.data.Dataset<SequenceExample>): Input dataset on which to perform temporal
                batching on.

        Returns:
            (tf.data.Dataset<SequenceExample>): Examples from the input dataset batched temporally.
                All tensors included in the example will gain an additional timestep
                dimension (e.g. a CHW image will become a TCHW, where T matches ``size``.)
        """
        dataset = dataset.batch(self.size, drop_remainder=True)

        # We only need to filter if there was actually temporal batching.
        if self.size > 1:
            dataset = dataset.filter(predicate=self.predicate)

        return dataset
