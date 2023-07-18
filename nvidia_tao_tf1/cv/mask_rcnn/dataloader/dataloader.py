# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Data loader and processing.

Defines input_fn of Mask-RCNN for TF Estimator. The input_fn includes training
data for category classification, bounding box regression, and number of
positive examples to normalize the loss during training.

"""
import functools
import math

import tensorflow as tf

from nvidia_tao_tf1.cv.mask_rcnn.dataloader.dataloader_utils import dataset_parser
from nvidia_tao_tf1.cv.mask_rcnn.utils.distributed_utils import MPI_is_distributed
from nvidia_tao_tf1.cv.mask_rcnn.utils.distributed_utils import MPI_rank
from nvidia_tao_tf1.cv.mask_rcnn.utils.distributed_utils import MPI_rank_and_size
from nvidia_tao_tf1.cv.mask_rcnn.utils.distributed_utils import MPI_size
from nvidia_tao_tf1.cv.mask_rcnn.utils.logging_formatter import logging

# common functions


class InputReader(object):
    """Input reader for dataset."""

    def __init__(
        self,
        file_pattern,
        mode=tf.estimator.ModeKeys.TRAIN,
        num_examples=0,
        use_fake_data=False,
        use_instance_mask=False,
        seed=None
    ):
        """Init."""
        self._mode = mode
        self._file_pattern = file_pattern
        self._num_examples = num_examples
        self._use_fake_data = use_fake_data
        self._use_instance_mask = use_instance_mask
        self._seed = seed

    def _create_dataset_parser_fn(self, params):
        """Create parser for parsing input data (dictionary)."""

        return functools.partial(
            dataset_parser,
            mode=self._mode,
            params=params,
            use_instance_mask=self._use_instance_mask,
            seed=self._seed
        )

    def __call__(self, params, input_context=None):
        """Call."""
        batch_size = params['batch_size'] if 'batch_size' in params else 1
        try:
            seed = params['seed'] if not MPI_is_distributed() else params['seed'] * MPI_rank()
        except (KeyError, TypeError):
            seed = None

        if MPI_is_distributed():
            n_gpus = MPI_size()

        elif input_context is not None:
            n_gpus = input_context.num_input_pipelines

        else:
            n_gpus = 1
        logging.debug("Number of GPUs: ".format(n_gpus))
        ##################################################

        dataset = tf.data.Dataset.list_files(
            self._file_pattern,
            shuffle=False
        )

        if self._mode == tf.estimator.ModeKeys.TRAIN:

            if input_context is not None:
                logging.info("Using Dataset Sharding with TF Distributed")
                _num_shards = input_context.num_input_pipelines
                _shard_idx = input_context.input_pipeline_id

            elif MPI_is_distributed():
                logging.info("Using Dataset Sharding with Horovod")
                _shard_idx, _num_shards = MPI_rank_and_size()

            try:
                dataset = dataset.shard(
                    num_shards=_num_shards,
                    index=_shard_idx
                )
                dataset = dataset.shuffle(math.ceil(256 / _num_shards))

            except NameError:  # Not a distributed training setup
                pass

        def _prefetch_dataset(filename):
            return tf.data.TFRecordDataset(filename).prefetch(1)

        dataset = dataset.interleave(
            map_func=_prefetch_dataset,
            cycle_length=32,
            block_length=64,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        if self._num_examples is not None and self._num_examples > 0:
            logging.info("[*] Limiting the amount of sample to: %d" % self._num_examples)
            dataset = dataset.take(self._num_examples)

        dataset = dataset.cache()

        if self._mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(
                buffer_size=params['shuffle_buffer_size'] or 4096,
                reshuffle_each_iteration=True,
                seed=seed
            )

            dataset = dataset.repeat()

        # Parse the fetched records to input tensors for model function.
        dataset = dataset.map(
            map_func=self._create_dataset_parser_fn(params),
            num_parallel_calls=params['n_workers'] or 16,
        )

        dataset = dataset.batch(
            batch_size=batch_size,
            drop_remainder=True
        )

        if self._use_fake_data:
            # Turn this dataset into a semi-fake dataset which always loop at the
            # first batch. This reduces variance in performance and is useful in
            # testing.
            logging.info("Using Fake Dataset Loop...")
            dataset = dataset.take(1).cache().repeat()

            if self._mode != tf.estimator.ModeKeys.TRAIN:
                dataset = dataset.take(int(5000 / batch_size))

        dataset = dataset.prefetch(
            buffer_size=params['prefetch_buffer_size'] or tf.data.experimental.AUTOTUNE,
        )

        if not tf.distribute.has_strategy():
            dataset = dataset.apply(
                tf.data.experimental.prefetch_to_device(
                    '/gpu:0',  # With Horovod the local GPU is always 0
                    buffer_size=1,
                )
            )

        data_options = tf.data.Options()

        data_options.experimental_deterministic = seed is not None
        data_options.experimental_distribute.auto_shard = False
        data_options.experimental_slack = True

        data_options.experimental_threading.max_intra_op_parallelism = 1

        # ================= experimental_optimization ================= #

        data_options.experimental_optimization.apply_default_optimizations = False

        # data_options.experimental_optimization.autotune = True
        data_options.experimental_optimization.filter_fusion = True
        data_options.experimental_optimization.map_and_batch_fusion = True
        data_options.experimental_optimization.map_and_filter_fusion = True
        data_options.experimental_optimization.map_fusion = True
        data_options.experimental_optimization.map_parallelization = True

        map_vectorization_options = tf.data.experimental.MapVectorizationOptions()
        map_vectorization_options.enabled = True
        map_vectorization_options.use_choose_fastest = True

        data_options.experimental_optimization.map_vectorization = map_vectorization_options

        data_options.experimental_optimization.noop_elimination = True
        data_options.experimental_optimization.parallel_batch = True
        data_options.experimental_optimization.shuffle_and_repeat_fusion = True

        # ========== Stats on TF Data =============
        # aggregator = tf.data.experimental.StatsAggregator()
        # data_options.experimental_stats.aggregator = aggregator
        # data_options.experimental_stats.latency_all_edges = True

        dataset = dataset.with_options(data_options)

        return dataset
