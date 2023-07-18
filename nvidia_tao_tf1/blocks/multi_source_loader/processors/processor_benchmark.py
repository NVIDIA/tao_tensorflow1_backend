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
"""Processors benchmark suite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors.crop import Crop
from nvidia_tao_tf1.blocks.multi_source_loader.processors.pipeline import Pipeline
from nvidia_tao_tf1.blocks.multi_source_loader.processors.random_contrast import (
    RandomContrast,
)
from nvidia_tao_tf1.blocks.multi_source_loader.processors.scale import Scale
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    empty_polygon_label,
    Example,
)


class ProcessorBenchmark(tf.test.Benchmark):
    """Processors benchmark suite."""

    ITERATIONS = 1000

    def create_example(self, image_width, image_height):
        """Create an example.

        Args:
            image_width (int): Image width to generate image.
            image_height (int): Image height to generate image.

        Return:
            (Example): A 4D example tensor.
        """
        frames = tf.constant(1.0, shape=[3, image_height, image_width])
        return Example(
            frames=tf.expand_dims(frames, axis=0),
            labels=empty_polygon_label(),
            ids=tf.constant(42),
        )

    def _benchmark_single_processor(self, sess):
        """Benchmark single processor.

        Args:
            sess (tf.Session()): Session to run the benchmark.
        """
        example = self.create_example(960, 504)
        crop = Crop(left=181, top=315, right=779, bottom=440)
        pipeline = Pipeline([crop])
        run_tensor = pipeline(example)
        self.run_op_benchmark(
            sess=sess,
            op_or_tensor=run_tensor,
            min_iters=self.ITERATIONS,
            store_trace=True,
            store_memory_usage=True,
        )

    def _benchmark_multi_processors(self, sess):
        """Benchmark multiple processors.

        Args:
            sess (tf.Session()): Session to run the benchmark.
        """
        example = self.create_example(960, 504)
        crop = Crop(left=181, top=315, right=779, bottom=440)
        scale = Scale(height=200, width=960)
        random_contrast = RandomContrast(scale_max=0.5, center=0.5)
        pipeline = Pipeline([crop, scale, random_contrast])
        run_tensor = pipeline(example)
        self.run_op_benchmark(
            sess=sess,
            op_or_tensor=run_tensor,
            min_iters=self.ITERATIONS,
            store_trace=True,
            store_memory_usage=True,
        )

    def benchmark_single_processor_with_gpu(self):
        """Benchmark a single processor with gpu."""
        with tf.device("/gpu:0"), tf.compat.v1.Session() as sess:
            self._benchmark_single_processor(sess)

    def benchmark_single_processor_with_cpu(self):
        """Benchmark a single processor with cpu."""
        with tf.device("/cpu:0"), tf.compat.v1.Session() as sess:
            self._benchmark_single_processor(sess)

    def benchmark_multi_processors_with_gpu(self):
        """Benchmark multiple processors with gpu."""
        with tf.device("/gpu:0"), tf.compat.v1.Session() as sess:
            self._benchmark_multi_processors(sess)

    def benchmark_multi_processors_with_cpu(self):
        """Benchmark multiple processors with cpu."""
        with tf.device("/cpu:0"), tf.compat.v1.Session() as sess:
            self._benchmark_multi_processors(sess)


if __name__ == "__main__":
    tf.test.main()
