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
"""Data loader for ingesting training data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import multiprocessing
import os

import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader import processors
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    FEATURE_CAMERA,
    TransformedExample,
)
from nvidia_tao_tf1.blocks.trainer.data_loader_interface import (
    DataLoaderInterface,
)
from nvidia_tao_tf1.core.coreobject import save_args


logger = logging.getLogger(__name__)
MAX_SHUFFLE_BUFFER = 10000


def _normalize_images(example, dtype=tf.float32):
    """Cast uint8 jpeg/png images to dtype and normalize it into the range [0 , 1].

    Args:
        example (Example or TransformedExample): The example that contains images.
        dtype (tf.dtypes.DType): The dtype that the images are cast to.
    """
    camera = example.instances[FEATURE_CAMERA]
    images = 1.0 / 255 * tf.cast(camera.images, dtype)
    example.instances[FEATURE_CAMERA] = camera._replace(images=images)


def _pick_largest_image_dtype(data_sources):
    """Pick a image_dtype for a list of data_source.

    The policy is that when a list of data_source with mixed dtypes are given,
    the dtype with highest precision is picked.

    Args:
        data_sources (list<data_source>): A list of data_sources.

    Return:
        (tf.dtypes.Dtype) The picked datatype.
    """
    sorted_data_sources = sorted(
        data_sources, key=lambda data_source: data_source.image_dtype.size
    )

    if (
        sorted_data_sources[0].image_dtype.size
        != sorted_data_sources[-1].image_dtype.size
    ):
        logger.warning(
            "Warning: Data sources are not with the same dtype, might result in reduced perf."
            "For example: dtype {} will be casted to  dtype {}".format(
                sorted_data_sources[0].image_dtype.name,
                sorted_data_sources[-1].image_dtype.name,
            )
        )
    return sorted_data_sources[-1].image_dtype


class DataLoader(DataLoaderInterface):
    """Functor for feeding data into Estimators."""

    @save_args
    def __init__(
        self,
        data_sources,
        augmentation_pipeline,
        batch_size=None,
        batch_size_per_gpu=None,
        shuffle=True,
        preprocessing=None,
        pipeline_dtype=None,
        pseudo_sharding=False,
        repeat=True,
        sampling="user_defined",
        serial_augmentation=False,
        label_names=None,
    ):
        """
        Construct an input pipeline.

        Args:
            data_sources (list): Each element is a ``DataSource`` to read examples from.
            augmentation_pipeline (Pipeline or list[Processor]): Transformations that get applied
                to examples from all data sources.
            batch_size (int): Number of examples to batch together.
                              If not set, batch_size_per_gpu should be set.
            batch_size_per_gpu (int): Number of examples per gpu to batch together.
                                         If not set, batch_size_per_gpu should be set.
            shuffle (bool): If True, data will be shuffled.
            preprocessing (list<Processor>): Processors for preprocessing all sources. If no
                temporal batching processor is included, one is automatically added to the list
                to ensure DataLoader output always includes a time dimension.
                NOTE: Defaults to None for backwards compatibility with DataSources that implement
                temporal batching (i.e. produce 4D images.)
            pipeline_dtype (str): Feature tensors (eg. images) are converted to this
                dtype for processing. Defaults 'float16'.
            pseudo_sharding (bool): If True, then data is not actually sharded, but different
                shuffle seeds are used to differentiate shard batches.
            repeat (bool): Whether or not this DataLoader iterates over its contents ad vitam
                aeternam.
            sampling (str): A sampling strategy for how to sample the individual data sources.
                Accepted values are:
                    'user_defined': Use the sample_ratio field of each data source. This is the
                        default behavior.
                    'uniform': The equivalent of every data source's sampling ratio being 1., i.e.
                        each data source is equally likely to be the one producing the next sample.
                    'proportional': The sample ratio of each data source is proportional to the
                        number of samples within each data source.
                    `None`: No sampling is applied. Instead, the data sources are all concatenated
                        to form a single dataset. This should be the default when iterating on
                        a validation / test dataset, as we would like to see each sample exactly
                        once.
                        !!! NOTE !!!: Do not use this in conjunction with `repeat=True`.
            serial_augmentation(bool): Whether to apply augmentation in serial to aid in
                reproducibility. Default is False which means augmentations would be applied in
                parallel.
            label_names (set<str>): Set of label names produced by this data loader.

        Raises:
            ValueError: When no ``data_sources`` are provided.
        """
        super(DataLoader, self).__init__()
        if not data_sources:
            raise ValueError("DataLoader excepts at least one element in data_sources.")
        if sampling not in ["user_defined", "uniform", "proportional", None]:
            raise ValueError("Unsupported sampling %s" % sampling)
        self._data_sources = data_sources
        self._augmentation_pipeline = augmentation_pipeline
        self._pipeline_dtype = pipeline_dtype
        # Note, currently we are supporting both batch size for all workers (batch_size)
        # and batch size per gpu (batch_size_per_gpu). Only one of them can be set.
        # TODO (weich) remove batch_size support and fully switch to batch_size_per_gpu.
        if not ((batch_size is None) ^ (batch_size_per_gpu is None)):
            raise ValueError(
                "Exactly one of batch_size and batch_size_per_gpu must be set."
            )
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size_per_gpu is not None and batch_size_per_gpu <= 0:
            raise ValueError("batch_size_per_gpu must be positive.")
        self._batch_size = batch_size
        self._batch_size_per_gpu = batch_size_per_gpu
        self._shuffle = shuffle
        self._shard_count = 1
        self._shard_index = 0
        self._sharding_configured = False
        self._pseudo_sharding = pseudo_sharding
        self._serial_augmentation = serial_augmentation
        self._label_names = label_names or set()
        logger.info("Serial augmentation enabled = {}".format(serial_augmentation))
        self._sampling = sampling
        logger.info("Pseudo sharding enabled = {}".format(pseudo_sharding))
        self._temporal_size = None
        # TODO(vkallioniemi): Default to always adding a temporal batcher if it is missing. Not
        # adding it when preprocessing is None is a temporary solution that ensures that code
        # that is not using geometric primitives keeps on working.
        if preprocessing is None:
            preprocessing = []
        else:
            has_temporal_batcher = False
            for processor in preprocessing:
                if isinstance(processor, processors.TemporalBatcher):
                    self._temporal_size = processor.size
                    has_temporal_batcher = True
            if not has_temporal_batcher:
                # Ensure outputs always include a temporal dimension because preprocessing,
                # augmentation and models expect that.
                self._temporal_size = 1
                preprocessing = [processors.TemporalBatcher()] + preprocessing

        self._preprocessing = preprocessing
        self._repeat = repeat

    @property
    def steps(self):
        """Return the number of steps."""
        if not self._sharding_configured:
            raise ValueError(
                "After constructing the DataLoader `set_shard()` method needs to "
                "be called to set the sharding arguments and configure the sources."
            )
        size = len(self)
        # Assign each worker a fraction of total steps,
        # because training data is evenly sharded among them.
        sharded_size = int(math.ceil(size / self._shard_count))
        return int(math.ceil(sharded_size / self._batch_size_per_gpu))

    @property
    def batch_size_per_gpu(self):
        """Return the number of examples each batch contains."""
        return self._batch_size_per_gpu

    @property
    def label_names(self):
        """Gets the label names produced by this dataloader."""
        return self._label_names

    def __len__(self):
        """Return the total number of examples that will be produced."""
        if not self._sharding_configured:
            raise ValueError(
                "After constructing the DataLoader `set_shard()` method needs to "
                "be called to set the sharding arguments and configure the sources."
            )
        num_examples = 0
        for source in self._data_sources:
            if source.supports_temporal_batching():
                num_examples += len(source)
            else:
                num_examples += len(source) // (self._temporal_size or 1)
        return num_examples

    def _configure_sources(self):
        all_max_dims = None
        for source in self._data_sources:
            if source.supports_sharding():
                # source handles its own sharding
                source.set_shard(
                    self._shard_count, self._shard_index, self._pseudo_sharding
                )
            if source.supports_shuffling() and self._shuffle:
                source.set_shuffle(MAX_SHUFFLE_BUFFER)
            source.set_sequence_length(self._temporal_size)
            source.initialize()

            # Verify that all of the input sources are temporal-compatible given the settings
            if (
                self._temporal_size
                and len(source) % self._temporal_size != 0
                and not source.supports_temporal_batching()
            ):
                raise ValueError(
                    "All datasources must have a number of samples divisible by "
                    "the temporal sequence length. Sequence Length: {}. "
                    "Invalid source: {}".format(self._temporal_size, source)
                )

            max_dims = source.get_image_properties()
            if not all_max_dims:
                all_max_dims = max_dims
            else:
                all_max_dims = [max(a, b) for a, b in zip(all_max_dims, max_dims)]

        logger.info("Max Image Dimensions (all sources): {}".format(all_max_dims))

        for source in self._data_sources:
            source.set_image_properties(*all_max_dims)

        # Handle sampling ratios here. Note that the default of 'user_defined' need not be
        # addressed since it'll resolve to using what each data_source was configured with
        # in terms of sample_ratio.
        if self._sampling == "uniform":
            for source in self._data_sources:
                source.sample_ratio = 1.0
        elif self._sampling == "proportional":
            for source in self._data_sources:
                source.sample_ratio = float(len(source))

    def set_shard(self, shard_count=1, shard_index=0):
        """
        Configure the sharding for the current job.

        Args:
            shard_count (int): Number of shards that each dataset will be split into.
            shard_index (int): Index of shard to use [0, shard_count-1].
        """
        if shard_count < 1:
            raise ValueError("at least 1 shard is needed")
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError("shard_index must be between 0 and shard_count-1")
        self._shard_count = shard_count
        self._shard_index = shard_index

        if self._batch_size_per_gpu is None:
            # Compute batch_size_per_gpu of training for each process when using multi-GPU
            batch_size_per_gpu, remainder = divmod(self._batch_size, shard_count)
            if remainder != 0:
                raise ValueError(
                    "Cannot evenly distribute a batch size of {} over {} "
                    "processors".format(self._batch_size, shard_count)
                )
            self._batch_size_per_gpu = batch_size_per_gpu
        self._configure_sources()
        self._sharding_configured = True

    def summary(self, print_fn=None):
        """
        Print a summary of the contents of this data loader.

        Args:
            print_fn (function): Optional function that each line of the summary will be passed to.
                Prints to stdout if not specified.
        """
        if print_fn is None:
            print_fn = print

        print_fn("  - examples: {}".format(len(self)))
        print_fn("  - steps: {}".format(self.steps))
        print_fn("  - batch size per gpu: {}".format(self.batch_size_per_gpu))
        print_fn("  - shuffle: {}".format(self._shuffle))
        print_fn("  - shard count: {}".format(self._shard_count))
        print_fn("  - shard index: {}".format(self._shard_index))
        print_fn("  - pseudo-sharding: {}".format(self._pseudo_sharding))
        print_fn("  - serial augmentation: {}".format(self._serial_augmentation))
        print_fn("  - sources:")

        def indented_print_fn(string):
            print_fn("    " + string)

        for i, source in enumerate(self._data_sources):
            indented_print_fn("Source {}: {}".format(i, type(source).__name__))
            source.summary(print_fn=indented_print_fn)

        if self._preprocessing:
            print_fn("  - preprocessing:")
            for processor in self._preprocessing:
                indented_print_fn("  - {}".format(str(processor)))

        if self._augmentation_pipeline:
            print_fn("  - augmentations:")
            for augmentation in self._augmentation_pipeline:
                indented_print_fn("  - {}".format(str(augmentation)))

    def call(self):
        """Produce examples with input features (such as images) and labels.

        Returns:
            examples (Example / SequenceExample): Example structure containing tf.Tensor that have
                had augmentation applied to them.
        """
        if self._pipeline_dtype:
            image_loading_dtype = tf.dtypes.as_dtype(self._pipeline_dtype)
        else:
            image_loading_dtype = _pick_largest_image_dtype(self._data_sources).name
        # TODO (vkallioniemi): refactor to move batching to sources.

        # See [performance guideline](https://www.tensorflow.org/performance/datasets_performance)
        # before making changes here.
        core_count = multiprocessing.cpu_count()
        # Roughly tuned on pathnet, 8 GPUS, against 469735 images roughly half of which were png
        # and half were fp16.
        # Previously this field was configurable, we discovered that in no situation was using 1
        # IO thread better, and it is strictly much much worse on AVDC.
        # TODO(@williamz): When multi-node is used, base shard count and cpu count on the local
        # node.
        io_threads = max(2 * (core_count // self._shard_count), 1)
        if self._serial_augmentation:
            compute_threads = 1
        else:
            compute_threads = core_count // self._shard_count

        # For a typical batch of 32 images sized 604x960x3, the memory requirements are:
        # 1 buffered batch: 32 * 3MB ~= 100MB.
        buffered_batches = 4
        logger.info(
            "number of cpus: %d, io threads: %d, compute threads: %d, buffered batches: %d",
            core_count,
            io_threads,
            compute_threads,
            buffered_batches,
        )
        logger.info(
            "total dataset size %d, number of sources: %d, batch size per gpu: %d, steps: %d",
            len(self),
            len(self._data_sources),
            self.batch_size_per_gpu,
            self.steps,
        )

        # TODO(@williamz): Break this method up into smaller functional pieces.
        datasets = []
        weights = []
        for source in self._data_sources:
            dataset = source()

            if not source:
                logger.warning("skipping empty datasource")
                continue

            # TODO(vkallioniemi): Remove this if statement once all sources have been changed
            # to parse examples by default.
            if source.parse_example is not None:
                dataset = dataset.apply(source.parse_example)
            logger.info(
                "shuffle: %s - shard %d of %d",
                self._shuffle,
                self._shard_index,
                self._shard_count,
            )

            # Apply temporal batching and other global preprocessing.
            # NOTE: This needs to be performed before sharding because operations such as
            # temporal batching require examples to be chunked together.
            for processor in self._preprocessing:
                # If source itself handles the temporal batching (like sqlite, we do not apply
                # temporal batcher here).
                if source.supports_temporal_batching():
                    if isinstance(processor, processors.TemporalBatcher):
                        continue
                dataset = dataset.apply(processor)

            # Evenly distribute records from each dataset to each GPU.
            if self._shard_count != 1:
                if not source.supports_sharding() and not self._pseudo_sharding:
                    dataset = dataset.shard(self._shard_count, self._shard_index)

            if self._shuffle:
                if not source.supports_shuffling():
                    dataset = dataset.apply(
                        tf.data.experimental.shuffle_and_repeat(
                            buffer_size=min(len(source), MAX_SHUFFLE_BUFFER), count=None
                        )
                    )
                elif self._repeat:
                    # NOTE (@williamz): this line seems to imply the tf.data.Dataset object
                    # produced by the source handles the repeat() call in a way that does not
                    # result in the order being the same at every iteration over the entire dataset.
                    # This needs to be investigated properly for the sqlite data source.
                    dataset = dataset.repeat()
            elif self._repeat:
                dataset = dataset.repeat()

            # Combine pipelines so that we do not lose information when affine
            # transforms get applied at the end of a pipeline.
            # The processors that are not transform processors need to be applied later.
            # They require the images to be loaded.
            delayed_processors = []
            transform_processors = []
            if isinstance(source.preprocessing, list):
                # Source processors to apply specified as a list.
                pipeline = processors.Pipeline(source.preprocessing)
            else:
                pipeline = source.preprocessing
            if isinstance(self._augmentation_pipeline, list):
                # Processors to apply specified as a list.
                for processor in self._augmentation_pipeline:
                    if not isinstance(processor, processors.TransformProcessor):
                        delayed_processors.append(processor)
                    else:
                        transform_processors.append(processor)
                augmentation_pipeline = processors.Pipeline(transform_processors)
                pipeline += augmentation_pipeline
            else:
                pipeline += self._augmentation_pipeline
            if pipeline:
                dataset = dataset.map(pipeline, num_parallel_calls=compute_threads)

            datasets.append(dataset)
            weights.append(source.sample_ratio)

        if self._sampling is not None:
            total = sum(weights)
            weights = [weight / total for weight in weights]
            logger.info("sampling %d datasets with weights:", len(datasets))
            for index, weight in enumerate(weights):
                logger.info("source: %d weight: %f", index, weight)
            combined = tf.data.experimental.sample_from_datasets(
                datasets, weights=weights
            )
        else:
            combined = datasets[0]
            for dataset in datasets[1:]:
                combined = combined.concatenate(dataset)

        # Note: increasing parallelism will increase memory usage.
        combined = combined.map(
            processors.AssetLoader(output_dtype=image_loading_dtype),
            num_parallel_calls=io_threads,
        )

        if delayed_processors:
            delayed_pipeline = processors.Pipeline(delayed_processors)
        else:
            delayed_pipeline = processors.Pipeline([])

        # Drop remainder since some downstream users rely on known batch size, specifically
        # weighted_binary_crossentropy.
        # TODO(ehall): Fix loss function to be able to handle variable sized batches.
        # TODO(vkallioniemi): consider batching per dataset instead.
        combined = combined.batch(self._batch_size_per_gpu, drop_remainder=True)

        # Buffer more data here to pipeline CPU-GPU processing.
        combined = combined.prefetch(buffered_batches)
        if tf.executing_eagerly():
            batch = combined
        else:

            # Take one item from dataloader and use it repeatedly to serve the rest pipeline.
            # This feature is used to benchmark the training pipeline with all data in memory.
            # !!! Do not enable it in production.
            if os.environ.get("MODULUS_DATALOADER_BYPASS_LOADING"):
                combined = combined.take(1)
                combined = combined.cache()
                combined = combined.repeat()

            iterator = tf.compat.v1.data.make_initializable_iterator(combined)
            # So it can be easily initialized downstream.
            tf.compat.v1.add_to_collection("iterator_init", iterator.initializer)
            batch = iterator.get_next(name="data_loader_out")
            # Cast / normalize images if the images are uint8 png / jpeg.
            # TODO(weich, mlehr) batch is not an Example instance in eager mode.
            # Disable this for eager mode until this issue has been fixed.
            if isinstance(batch, TransformedExample):
                example = batch.example
            else:
                example = batch

            if example.instances[FEATURE_CAMERA].images.dtype == tf.uint8:
                _normalize_images(example)

        # Apply transform matrices stored with a TransformedExample. Transformations
        # are delayed up until after tf.data processing so that they can be performed on
        # the GPU.
        examples = batch() if isinstance(batch, TransformedExample) else batch
        # (mlehr): Delayed pipeline needs to be applied after the transform processors.
        # If the delayed pipeline is applied before, the output of it will be SequenceExample, so
        # the transform processors would not be applied.
        processed_examples = delayed_pipeline(examples)

        return processed_examples


class DataLoaderYOLOv3(DataLoader):
    """Customized DataLoader for YOLO v3."""

    def call(self):
        """Produce examples with input features (such as images) and labels.

        Returns:
            examples (Example / SequenceExample): Example structure containing tf.Tensor that have
                had augmentation applied to them.
        """
        if self._pipeline_dtype:
            image_loading_dtype = tf.dtypes.as_dtype(self._pipeline_dtype)
        else:
            image_loading_dtype = _pick_largest_image_dtype(self._data_sources).name
        # TODO (vkallioniemi): refactor to move batching to sources.

        # See [performance guideline](https://www.tensorflow.org/performance/datasets_performance)
        # before making changes here.
        core_count = multiprocessing.cpu_count()
        # Roughly tuned on pathnet, 8 GPUS, against 469735 images roughly half of which were png
        # and half were fp16.
        # Previously this field was configurable, we discovered that in no situation was using 1
        # IO thread better, and it is strictly much much worse on AVDC.
        # TODO(@williamz): When multi-node is used, base shard count and cpu count on the local
        # node.
        io_threads = max(2 * (core_count // self._shard_count), 1)
        if self._serial_augmentation:
            compute_threads = 1
        else:
            compute_threads = core_count // self._shard_count

        # For a typical batch of 32 images sized 604x960x3, the memory requirements are:
        # 1 buffered batch: 32 * 3MB ~= 100MB.
        buffered_batches = tf.data.experimental.AUTOTUNE
        logger.info(
            "number of cpus: %d, io threads: %d, compute threads: %d, buffered batches: %d",
            core_count,
            io_threads,
            compute_threads,
            buffered_batches,
        )
        logger.info(
            "total dataset size %d, number of sources: %d, batch size per gpu: %d, steps: %d",
            len(self),
            len(self._data_sources),
            self.batch_size_per_gpu,
            self.steps,
        )

        # TODO(@williamz): Break this method up into smaller functional pieces.
        datasets = []
        weights = []
        for source in self._data_sources:
            dataset = source()

            if not source:
                logger.warning("skipping empty datasource")
                continue

            # TODO(vkallioniemi): Remove this if statement once all sources have been changed
            # to parse examples by default.
            if source.parse_example is not None:
                dataset = dataset.apply(source.parse_example)
            logger.info(
                "shuffle: %s - shard %d of %d",
                self._shuffle,
                self._shard_index,
                self._shard_count,
            )

            # Apply temporal batching and other global preprocessing.
            # NOTE: This needs to be performed before sharding because operations such as
            # temporal batching require examples to be chunked together.
            for processor in self._preprocessing:
                # If source itself handles the temporal batching (like sqlite, we do not apply
                # temporal batcher here).
                if source.supports_temporal_batching():
                    if isinstance(processor, processors.TemporalBatcher):
                        continue
                dataset = dataset.apply(processor)

            # Evenly distribute records from each dataset to each GPU.
            if self._shard_count != 1:
                if not source.supports_sharding() and not self._pseudo_sharding:
                    dataset = dataset.shard(self._shard_count, self._shard_index)

            if self._shuffle:
                if not source.supports_shuffling():
                    dataset = dataset.apply(
                        tf.data.experimental.shuffle_and_repeat(
                            buffer_size=min(len(source), MAX_SHUFFLE_BUFFER), count=None
                        )
                    )
                elif self._repeat:
                    # NOTE (@williamz): this line seems to imply the tf.data.Dataset object
                    # produced by the source handles the repeat() call in a way that does not
                    # result in the order being the same at every iteration over the entire dataset.
                    # This needs to be investigated properly for the sqlite data source.
                    dataset = dataset.repeat()
            elif self._repeat:
                dataset = dataset.repeat()

            # Combine pipelines so that we do not lose information when affine
            # transforms get applied at the end of a pipeline.
            # The processors that are not transform processors need to be applied later.
            # They require the images to be loaded.
            delayed_processors = []
            transform_processors = []
            if isinstance(source.preprocessing, list):
                # Source processors to apply specified as a list.
                pipeline = processors.Pipeline(source.preprocessing)
            else:
                pipeline = source.preprocessing
            if isinstance(self._augmentation_pipeline, list):
                # Processors to apply specified as a list.
                for processor in self._augmentation_pipeline:
                    if not isinstance(processor, processors.TransformProcessor):
                        delayed_processors.append(processor)
                    else:
                        transform_processors.append(processor)
                augmentation_pipeline = processors.Pipeline(transform_processors)
                pipeline += augmentation_pipeline
            else:
                pipeline += self._augmentation_pipeline
            if pipeline:
                dataset = dataset.map(pipeline, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            datasets.append(dataset)
            weights.append(source.sample_ratio)

        if self._sampling is not None:
            total = sum(weights)
            weights = [weight / total for weight in weights]
            logger.info("sampling %d datasets with weights:", len(datasets))
            for index, weight in enumerate(weights):
                logger.info("source: %d weight: %f", index, weight)
            combined = tf.data.experimental.sample_from_datasets(
                datasets, weights=weights
            )
        else:
            combined = datasets[0]
            for dataset in datasets[1:]:
                combined = combined.concatenate(dataset)

        # Note: increasing parallelism will increase memory usage.
        combined = combined.map(
            processors.AssetLoader(output_dtype=image_loading_dtype),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        if delayed_processors:
            delayed_pipeline = processors.Pipeline(delayed_processors)
        else:
            delayed_pipeline = processors.Pipeline([])

        # Drop remainder since some downstream users rely on known batch size, specifically
        # weighted_binary_crossentropy.
        # TODO(ehall): Fix loss function to be able to handle variable sized batches.
        # TODO(vkallioniemi): consider batching per dataset instead.
        combined = combined.batch(self._batch_size_per_gpu, drop_remainder=True)

        # Buffer more data here to pipeline CPU-GPU processing.
        combined = combined.prefetch(buffered_batches)
        if tf.executing_eagerly():
            batch = combined
        else:

            # Take one item from dataloader and use it repeatedly to serve the rest pipeline.
            # This feature is used to benchmark the training pipeline with all data in memory.
            # !!! Do not enable it in production.
            if os.environ.get("MODULUS_DATALOADER_BYPASS_LOADING"):
                combined = combined.take(1)
                combined = combined.cache()
                combined = combined.repeat()

            iterator = tf.compat.v1.data.make_initializable_iterator(combined)
            # So it can be easily initialized downstream.
            tf.compat.v1.add_to_collection("iterator_init", iterator.initializer)
            batch = iterator.get_next(name="data_loader_out")
        # Apply transform matrices stored with a TransformedExample. Transformations
        # are delayed up until after tf.data processing so that they can be performed on
        # the GPU.
        examples = batch() if isinstance(batch, TransformedExample) else batch
        # (mlehr): Delayed pipeline needs to be applied after the transform processors.
        # If the delayed pipeline is applied before, the output of it will be SequenceExample, so
        # the transform processors would not be applied.
        processed_examples = delayed_pipeline(examples)

        return processed_examples
