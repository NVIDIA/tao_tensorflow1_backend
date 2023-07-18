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
"""Data loader benchmark suite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import logging
import os
import sqlite3
import tempfile
import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader import processors
from nvidia_tao_tf1.blocks.multi_source_loader.data_loader import DataLoader
from nvidia_tao_tf1.blocks.multi_source_loader.sources.sqlite_data_source import (
    SqliteDataSource,
)
from nvidia_tao_tf1.blocks.multi_source_loader.sources.synthetic_data_source import (
    SyntheticDataSource,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import test_fixtures


def _generate_fp16_image(
    export_path, sequence_id, camera_name, frame_number, height, width
):
    """Create files filled with zeros that look like exported fp16 images."""
    FP16_BYTES_PER_CHANNEL = 2
    size = height * width * 3 * FP16_BYTES_PER_CHANNEL
    image = np.zeros(size, dtype=np.int8)
    dir_path = os.path.join(export_path, sequence_id, camera_name)
    try:
        os.makedirs(dir_path)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(dir_path):
            pass
        else:
            raise

    path = os.path.join(dir_path, "{}.fp16".format(frame_number))
    image.tofile(path)


def _fetch_frames_and_sequence(sqlite_path):
    connection = sqlite3.connect(sqlite_path)
    cursor = connection.cursor()
    sequence_frames = cursor.execute(
        "SELECT frame_number, session_uuid "
        "FROM frames fr JOIN sequences seq ON fr.sequence_id=seq.id;"
    )
    return sequence_frames.fetchall()


class DataLoaderBenchmark(tf.test.Benchmark):
    """Data loader benchmark suite."""

    SQLITE_PATH = (
        "./moduluspy/modulus/dataloader/testdata/lane-assignment-RR-KPI_mini.sqlite"
    )
    ITERATIONS = 100
    TRACE = False
    MEMORY_USAGE = False

    def _create_synthetic_dataset(self, image_width, image_height, batch_size):
        """
        Build a synthetic datasource without any preprocessing/augmentation.

        This synthetic data source is an in-memory dataset without any further processing,
        the actual I/O on dataset is bypassed.

        Args:
            image_width (int): Image width to generate image.
            image_height (int): Image height to generate image.
            batch_size (int): Batch size.

        Return:
            (Example): A 3D example fetched from a dataset.
        """
        # Create a synthetic datasource with 1 example and use dataset.repeat()
        # to make it repeat forever.
        data_source = SyntheticDataSource(
            preprocessing=[processors.Crop(left=0, top=0, right=960, bottom=504)],
            example_count=1,
            template=test_fixtures.make_example_3d(image_height, image_width),
        )
        data_loader = DataLoader(
            data_sources=[data_source],
            augmentation_pipeline=[],
            batch_size=batch_size,
            preprocessing=[],
        )
        dataset = data_loader()
        dataset = dataset.repeat()
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        return iterator.get_next()

    def _create_sqlite_dataset(self, batch_size):
        """
        Build a sqlite datasource without any preprocessing/augmentation.

        Args:
            batch_size (int): Batch size.

        Return:
            (Example): A 3D example fetched from a dataset.
        """
        self.export_path = tempfile.mkdtemp()
        frames_and_sequences = _fetch_frames_and_sequence(self.SQLITE_PATH)
        for frame_number, sequence_uuid in frames_and_sequences:
            _generate_fp16_image(
                export_path=self.export_path,
                sequence_id=sequence_uuid,
                camera_name="video_B0_FC_60/rgb_half_dwsoftisp_v0.52b",
                frame_number=str(frame_number),
                height=604,
                width=960,
            )

        source = SqliteDataSource(
            sqlite_path=self.SQLITE_PATH,
            image_dir=self.export_path,
            export_format=SqliteDataSource.FORMAT_RGB_HALF_DWSOFTISP,
            preprocessing=[processors.Crop(left=0, top=0, right=960, bottom=504)],
        )

        data_loader = DataLoader(
            data_sources=[source],
            augmentation_pipeline=[],
            batch_size=batch_size,
            preprocessing=[],
        )
        dataset = data_loader()
        dataset = dataset.repeat()
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        return iterator.get_next()

    def benchmark_synthetic_dataset_24batch(self):
        """Benchmark 604x960 image with batch_size 24 for 100 iterations."""
        with tf.compat.v1.Session() as sess:
            run_tensor = self._create_synthetic_dataset(960, 604, 24)
            self.run_op_benchmark(
                sess=sess,
                op_or_tensor=run_tensor,
                min_iters=self.ITERATIONS,
                store_trace=self.TRACE,
                store_memory_usage=self.MEMORY_USAGE,
            )

    def benchmark_sqlite_dataset_24batch(self):
        """Benchmark 604x960 image with batch_size 24 for 100 iterations."""
        with tf.compat.v1.Session() as sess:
            run_tensor = self._create_sqlite_dataset(24)
            self.run_op_benchmark(
                sess=sess,
                op_or_tensor=run_tensor,
                min_iters=self.ITERATIONS,
                store_trace=self.TRACE,
                store_memory_usage=self.MEMORY_USAGE,
            )

    def benchmark_sqlite_dataset_32batch(self):
        """Benchmark 604x960 image with batch_size 32 for 100 iterations."""
        with tf.compat.v1.Session() as sess:
            run_tensor = self._create_sqlite_dataset(32)
            self.run_op_benchmark(
                sess=sess,
                op_or_tensor=run_tensor,
                min_iters=self.ITERATIONS,
                store_trace=self.TRACE,
                store_memory_usage=self.MEMORY_USAGE,
            )

    def benchmark_synthetic_dataset_none_32batch(self):
        """Benchmark 604x960 image with batch_size 32 for 100 iterations."""
        sess = tf.compat.v1.Session()
        run_tensor = self._create_synthetic_dataset(960, 604, 32)
        self.run_op_benchmark(
            sess=sess,
            op_or_tensor=run_tensor,
            min_iters=self.ITERATIONS,
            store_trace=self.TRACE,
            store_memory_usage=self.MEMORY_USAGE,
        )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.test.main()
