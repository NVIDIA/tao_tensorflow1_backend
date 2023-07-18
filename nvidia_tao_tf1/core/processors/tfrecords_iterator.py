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

"""TFRecords Iterator Processor."""

import tensorflow as tf
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.processors import Processor


class TFRecordsIterator(Processor):
    """Processor that sets up a TFRecordsDataset and yields values from that input.

    This uses TF Dataset API with initializable iterator. Note that when used for evaluation,
    ``repeat`` option should be set to ``False``.

    Args:
        file_list (string): File paths to tf records files, possibly containing wildcards.
            All matching files will be iterated over each epoch. If this is `None`, you need to
            pass in a ``tf.dataset`` object to the ``build`` method.
        batch_size (int): How many records to return at a time.
        shuffle_buffer_size (int): The maximum number of records the buffer will contain.
            If more than 0, ``shuffle`` needs to be ``True``.
        shuffle (bool): Toggle shuffling. If ``True``, ``shuffle_buffer_size`` needs to be
            more than 0.
        repeat (bool): Toggle repeating the tfrecords. If this is False, it will only output
            tensors for one full cycle through all the data in the tfrecords files. If ``True``,
            this can result in the last batch size of the epoch not being identical to
            ``batch_size``.
        batch_as_list (bool): Whether a batch should be returned as a list (i.e. split into single
            elements, rather than as a single large tensor with first dimension = ``batch_size``).
            False by default.
        sequence_length (int): Length of the sequence for sequence batching. A value of 0 means
            disabled and is the default value. The output of the iterator is flattened to a
            batch size of ``sequence_length * batch_size``. The sequence is obtained before
            shuffling.
        prefetch_buffer_size (int): How many batches should be prefetched (buffered). If this value
            is 0, no buffering or prefetching will occur.
        cache (bool): If you want to cache the entire dataset in memory.

    Raises:
        ValueError: if ``batch_as_list`` is set while ``repeat`` is False,
            or if ``shuffle_buffer_size`` is greater than zero when ``shuffle`` is False,
            or if ``shuffle`` is set with ``shuffle_buffer_size`` less than 1.
    """

    ITERATOR_INIT_OP_NAME = "iterator_init"

    @save_args
    def __init__(
        self,
        file_list,
        batch_size,
        shuffle_buffer_size=0,
        shuffle=False,
        repeat=False,
        batch_as_list=False,
        sequence_length=0,
        prefetch_buffer_size=0,
        cache=False,
        **kwargs
    ):
        """__init__ method."""
        self.file_list = file_list
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.batch_as_list = batch_as_list
        self.sequence_length = sequence_length
        self.prefetch_buffer_size = (
            prefetch_buffer_size
        )  # TODO(xiangbok): set default to >0
        self.cache = cache

        if self.repeat is False and self.batch_as_list:
            raise ValueError(
                "`batch_as_list` cannot be True if `repeat` is False because the "
                "split dimension (batch size) is not fixed before run-time. Because "
                "when repeat is False, the last batch can have a truncated size."
            )

        if self.shuffle is False and self.shuffle_buffer_size > 0:
            raise ValueError(
                "'shuffle' is False while 'shuffle_buffer_size' is %d."
                % shuffle_buffer_size
            )

        if self.shuffle is True and self.shuffle_buffer_size < 1:
            raise ValueError(
                "'shuffle' is True while 'shuffle_buffer_size' is %d."
                % self.shuffle_buffer_size
            )

        super(TFRecordsIterator, self).__init__(**kwargs)

    def _build(self, dataset=None, *args, **kwargs):  # pylint: disable=W1113
        """Build the record input.

        Args:
            dataset (TFRecordDataset): Optionally pass in a dataset object that's already been
                prepared.

        Raises:
            ValueError: if no ``file_list`` was specified in ``init`` and ``dataset`` is also None.
        """
        if dataset is None:
            if self.file_list is None:
                raise ValueError(
                    "If no `file_list` has been provided, a `dataset` needs to be "
                    "provided to the `build` method."
                )
            dataset = tf.data.TFRecordDataset(self.file_list)

        if self.cache:
            dataset = dataset.cache()

        if self.prefetch_buffer_size:
            dataset = dataset.prefetch(self.prefetch_buffer_size)

        if self.sequence_length:
            dataset = dataset.batch(self.sequence_length)

        if self.shuffle:
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer_size, reshuffle_each_iteration=True
            )
        if self.repeat:
            dataset = dataset.repeat()

        dataset = dataset.batch(self.batch_size)

        self.iterator = tf.compat.v1.data.Iterator.from_structure(
            dataset.output_types, dataset.output_shapes
        )
        self.iterator_init_op = self.iterator.make_initializer(dataset)

        # Add the iterator to our custom collection to easily retrieve later.
        tf.compat.v1.add_to_collection(
            self.ITERATOR_INIT_OP_NAME, self.iterator_init_op
        )

    def initialize(self, sess):
        """Initialize the iterator."""
        sess.run(self.iterator_init_op)

    def reset(self, sess):
        """Reset the iterator, as if no data has been pulled.

        Note that resetting is the same operation as initialization.
        """
        sess.run(self.iterator_init_op)

    def process_records(self, records):
        """Process records helper function."""
        if self.repeat:
            # Only if repeat is True, our batch size is fixed and we can perform reshaping.
            if self.sequence_length:
                records = tf.reshape(records, [self.batch_size * self.sequence_length])
            else:
                records = tf.reshape(records, [self.batch_size])
            if self.batch_as_list:
                records = tf.split(records, int(records.get_shape()[0]), 0)
                records = [tf.reshape(record, []) for record in records]

        return records

    def call(self):
        """call method.

        Returns:
            records: a list or dense tensor (depending on the value of ``batch_as_list``) containing
                the next batch as yielded from the `TFRecordsDataset`. Each new call will pull a
                fresh batch of samples. The set of input records cannot be depleted, as the records
                will wrap around to the next epoch as required.
                If the iterator reaches end of dataset, reinitialize the iterator
        """
        records = self.iterator.get_next()

        return self.process_records(records)
