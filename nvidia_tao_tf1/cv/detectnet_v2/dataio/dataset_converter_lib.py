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

"""Converts an object detection dataset to TFRecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
from collections import Counter
import json
import logging
import os
import random
import six
import tensorflow as tf

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import _shard
from nvidia_tao_tf1.cv.detectnet_v2.common.dataio.converter_lib import _shuffle
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.utilities import get_data_root


logger = logging.getLogger(__name__)


class DatasetConverter(six.with_metaclass(ABCMeta, object)):
    """Converts an object detection dataset to TFRecords.

    This class needs to be subclassed, and the convert() and
    create_example_proto() methods overridden to do the dataset
    conversion. Splitting of partitions to shards, shuffling and
    writing TFRecords are implemented here, as well as counting
    of written targets.
    """

    @abstractmethod
    def __init__(self, root_directory_path, num_partitions, num_shards,
                 output_filename, sample_modifier):
        """Initialize the converter.

        Args:
            root_directory_path (string): Dataset directory path relative to data root.
            num_partitions (int): Number of partitions (folds).
            num_shards (int): Number of shards.
            output_filename (str): Path for the output file.
            sample_modifier(SampleModifier): An instance of sample modifier
                that does e.g. duplication and filtering of samples.
        """
        self.root_dir = os.path.join(get_data_root(), root_directory_path)
        self.root_dir = os.path.abspath(self.root_dir)
        self.output_partitions = num_partitions
        self.output_shards = num_shards
        self.output_filename = output_filename
        output_dir = os.path.dirname(self.output_filename)
        # Make the output directory to write the shard.
        if not os.path.exists(output_dir):
            logger.info("Creating output directory {}".format(output_dir))
            os.makedirs(output_dir)
        self.sample_modifier = sample_modifier
        self.class_map = {}
        self.log_warning = {}
        # Set a fixed seed to get a reproducible sequence.
        random.seed(42)

    def convert(self):
        """Do the dataset conversion."""
        # Divide dataset into partitions and shuffle them.
        partitions = self._partition()
        _shuffle(partitions)

        # Shard and write the partitions to tfrecords.
        object_count = self._write_partitions(partitions)

        # Log how many objects per class got written in total.
        logger.info("Cumulative object statistics")
        cumulative_count_dict = {
            target_class.decode("ascii"): object_count.get(target_class)
            for target_class in object_count.keys()
        }
        s_logger = status_logging.get_status_logger()
        s_logger.categorical = {"num_objects": cumulative_count_dict}
        s_logger.write(
            message="Cumulative object statistics"
        )
        self._log_object_count(object_count)

        # Print out the class map
        log_str = "Class map. \nLabel in GT: Label in tfrecords file "
        for key, value in six.iteritems(self.class_map):
            log_str += "\n{}: {}".format(key, value)
        logger.info(log_str)
        s_logger.write(message=log_str)
        note_string = (
            "For the dataset_config in the experiment_spec, "
            "please use labels in the tfrecords file, while writing the classmap.\n"
        )
        print(note_string)
        s_logger.write(message=note_string)

        logger.info("Tfrecords generation complete.")
        s_logger.write(
            status_level=status_logging.Status.SUCCESS,
            message="TFRecords generation complete."
            )

        # Save labels with error to a JSON file
        self._save_log_warnings()

    def _write_partitions(self, partitions):
        """Shard and write partitions into tfrecords.

        Args:
            partitions (list): A list of list of frame IDs.

        Returns:
            object_count (Counter): The total number of objects per target class.
        """
        # Divide partitions into shards.
        sharded_partitions = _shard(partitions, self.output_shards)

        # Write .tfrecords to disk for each partition and shard.
        # Also count the target objects per partition and over the whole dataset.
        object_count = Counter()
        for p, partition in enumerate(sharded_partitions):
            partition_object_count = Counter()
            for s, shard in enumerate(partition):
                shard_object_count = self._write_shard(shard, p, s)
                partition_object_count += shard_object_count

            # Log the count in this partition and increase total
            # object count.
            self._log_object_count(partition_object_count)
            object_count += partition_object_count

        return object_count

    def _write_shard(self, shard, partition_number, shard_number):
        """Write a single shard into the tfrecords file.

        Note that the dataset-specific part is captured in function
        create_example_proto() which needs to be overridden for each
        specific dataset.

        Args:
            shard (list): A list of frame IDs for this shard.
            partition_number (int): Current partition (fold) index.
            shard_number (int): Current shard index.

        Returns:
            object_count (Counter): The number of written objects per target class.
        """
        logger.info('Writing partition {}, shard {}'.format(partition_number, shard_number))
        status_logging.get_status_logger().write(
            message='Writing partition {}, shard {}'.format(partition_number, shard_number)
        )
        output = self.output_filename

        if self.output_partitions != 0:
            output = '{}-fold-{:03d}-of-{:03d}'.format(output, partition_number,
                                                       self.output_partitions)
        if self.output_shards != 0:
            output = '{}-shard-{:05d}-of-{:05d}'.format(output, shard_number, self.output_shards)

        object_count = Counter()

        # Store all the data for the shard.
        writer = tf.python_io.TFRecordWriter(output)
        for frame_id in shard:

            # Create the Example with all labels for this frame_id.
            example = self._create_example_proto(frame_id)

            # The example might be skipped e.g. due to missing labels.
            if example is not None:
                # Apply modifications to the current sample such as filtering and duplication.
                # Only samples in the training set are modified.
                modified_examples = self.sample_modifier.modify_sample(example, partition_number)

                # Write the list of (possibly) modified samples.
                frame_object_count = Counter()
                for modified_example in modified_examples:
                    # Serialize the example.
                    writer.write(modified_example.SerializeToString())

                    # Count objects that got written per target class.
                    frame_object_count += self._count_targets(modified_example)

                object_count += frame_object_count

        writer.close()

        return object_count

    @abstractmethod
    def _partition(self):
        """Return dataset partitions."""
        pass

    @abstractmethod
    def _create_example_proto(self, frame_id):
        """Generate the example for this frame."""
        pass

    def _save_log_warnings(self):
        """Store out of bound bounding boxes to a json file."""
        if self.log_warning:
            logger.info("Writing the log_warning.json")
            with open(f"{self.output_filename}_warning.json", "w") as f:
                json.dump(self.log_warning, f, indent=2)
            logger.info("There were errors in the labels. Details are logged at"
                        " %s_waring.json", self.output_filename)

    def _count_targets(self, example):
        """Count the target objects in the given example protobuf.

        Args:
            example (tf.train.Example): Example protobuf containing the labels for a frame.

        Returns:
            object_count (Counter): Number of objects per target class.
        """
        target_classes = example.features.feature['target/object_class'].bytes_list.value
        object_count = Counter(target_classes)
        return object_count

    def _log_object_count(self, object_counts):
        """Log object counts per target class.

        Args:
            objects_counts (Counter or dict): Number of objects per target class.
        """
        log_str = '\nWrote the following numbers of objects:'
        for target_class, object_count in six.iteritems(object_counts):
            log_str += "\n{}: {}".format(target_class, object_count)
        log_str += "\n"
        logger.info(log_str)
