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
"""PolygonRasterizer benchmark suite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from parameterized import parameterized
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors.polygon_rasterizer import (
    PolygonRasterizer,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.test_fixtures import (
    make_polygon2d_label,
)
from modulus.utils import test_session


class PolygonRasterizerBenchmark(tf.test.Benchmark):
    """PolygonRasterizer benchmark suite."""

    ITERATIONS = 100

    def _benchmark_polygon_rasterizer(
        self,
        sess,
        example_count,
        frame_count,
        image_width,
        image_height,
        polygon_count,
        vertex_count,
        rasterizer_width,
        rasterizer_height,
    ):
        """
        Benchmark polygon_rasterizer with an example as input.

        Args:
            sess (tf.Session()): Session to run the benchmark.
            example_count (int): Number of examples in each batch.
            frame_count (int): Number of frames in each example.
            image_width (int): Image width to generate image.
            image_height (int): Image height to generate image.
            polygon_count (int): Number of polygons in each image.
            vertex_count (int): Number of vertices in each polygon.
            rasterizer_width (int): Width of output map.
            rasterizer_height (int): Height of output map.
        """
        shapes_per_frame = []
        for _ in range(example_count):
            shapes_per_frame.append([polygon_count for _ in range(frame_count)])

        labels2d = make_polygon2d_label(
            shapes_per_frame=shapes_per_frame,
            shape_classes=[4],
            shape_attributes=[4],
            height=image_height,
            width=image_width,
            coordinates_per_polygon=vertex_count,
        )

        processor = PolygonRasterizer(
            height=rasterizer_height,
            width=rasterizer_width,
            one_hot=True,
            binarize=True,
            nclasses=5,
        )
        rasterized = processor.process(labels2d)
        self.run_op_benchmark(
            sess=sess,
            op_or_tensor=rasterized.op,
            min_iters=self.ITERATIONS,
            store_trace=True,
            store_memory_usage=True,
        )

    @parameterized.expand(
        [
            [1, "/cpu:0"],
            [10, "/cpu:0"],
            [100, "/cpu:0"],
            [1000, "/cpu:0"],
            [1, "/gpu:0"],
            [10, "/gpu:0"],
            [100, "/gpu:0"],
            [1000, "/gpu:0"],
        ]
    )
    def benchmark_polygon_rasterizer_triangle_count(
        self, polygon_count, device_placement
    ):
        """Benchmark different numbers of triangles in each frame."""
        print("triangles_count {} device {}.".format(polygon_count, device_placement))
        with tf.device(device_placement), test_session(
            allow_soft_placement=True
        ) as sess:
            self._benchmark_polygon_rasterizer(
                sess=sess,
                example_count=32,
                frame_count=1,
                image_width=960,
                image_height=504,
                polygon_count=polygon_count,
                vertex_count=3,
                rasterizer_width=240,
                rasterizer_height=80,
            )

    @parameterized.expand(
        [
            [1, "/cpu:0"],
            [10, "/cpu:0"],
            [50, "/cpu:0"],
            [100, "/cpu:0"],
            [1, "/gpu:0"],
            [10, "/gpu:0"],
            [50, "/gpu:0"],
            [100, "/gpu:0"],
        ]
    )
    def benchmark_polygon_rasterizer_vertice_count(
        self, vertex_count, device_placement
    ):
        """Benchmark different numbers of vertices in each polygon."""
        print("vertex_count {} device {}.".format(vertex_count, device_placement))
        with tf.device(device_placement), test_session(
            allow_soft_placement=True
        ) as sess:
            self._benchmark_polygon_rasterizer(
                sess=sess,
                example_count=32,
                frame_count=1,
                image_width=960,
                image_height=504,
                polygon_count=5,
                vertex_count=vertex_count,
                rasterizer_width=240,
                rasterizer_height=80,
            )

    @parameterized.expand(
        [
            [1, "/cpu:0"],
            [4, "/cpu:0"],
            [8, "/cpu:0"],
            [16, "/cpu:0"],
            [32, "/cpu:0"],
            [64, "/cpu:0"],
            [1, "/gpu:0"],
            [4, "/gpu:0"],
            [8, "/gpu:0"],
            [16, "/gpu:0"],
            [32, "/gpu:0"],
            [64, "/gpu:0"],
        ]
    )
    def benchmark_polygon_rasterizer_example_count(
        self, example_count, device_placement
    ):
        """Benchmark different numbers of examples in each batch."""
        print("example_count {} device {}.".format(example_count, device_placement))
        with tf.device(device_placement), test_session(
            allow_soft_placement=True
        ) as sess:
            self._benchmark_polygon_rasterizer(
                sess=sess,
                example_count=example_count,
                frame_count=1,
                image_width=960,
                image_height=504,
                polygon_count=5,
                vertex_count=3,
                rasterizer_width=240,
                rasterizer_height=80,
            )

    @parameterized.expand(
        [
            [1, "/cpu:0"],
            [2, "/cpu:0"],
            [4, "/cpu:0"],
            [8, "/cpu:0"],
            [1, "/gpu:0"],
            [2, "/gpu:0"],
            [4, "/gpu:0"],
            [8, "/gpu:0"],
        ]
    )
    def benchmark_polygon_rasterizer_frame_count(self, frame_count, device_placement):
        """Benchmark different numbers of frames in each example."""
        print("frame_count {} device {}.".format(frame_count, device_placement))
        with tf.device(device_placement), test_session(
            allow_soft_placement=True
        ) as sess:
            self._benchmark_polygon_rasterizer(
                sess=sess,
                example_count=32,
                frame_count=frame_count,
                image_width=960,
                image_height=504,
                polygon_count=5,
                vertex_count=3,
                rasterizer_width=240,
                rasterizer_height=80,
            )

    @parameterized.expand(
        [
            [24, 8, "/cpu:0"],
            [240, 80, "/cpu:0"],
            [480, 160, "/cpu:0"],
            [960, 504, "/cpu:0"],
            [24, 8, "/gpu:0"],
            [240, 80, "/gpu:0"],
            [480, 160, "/gpu:0"],
            [960, 504, "/gpu:0"],
        ]
    )
    def benchmark_polygon_rasterizer_size(
        self, rasterizer_width, rasterizer_height, device_placement
    ):
        """Benchmark different rasterizer size."""
        print(
            "rasterizer size {} x {} device {}.".format(
                rasterizer_width, rasterizer_height, device_placement
            )
        )
        with tf.device(device_placement), test_session(
            allow_soft_placement=True
        ) as sess:
            self._benchmark_polygon_rasterizer(
                sess=sess,
                example_count=32,
                frame_count=1,
                image_width=960,
                image_height=504,
                polygon_count=5,
                vertex_count=3,
                rasterizer_width=rasterizer_width,
                rasterizer_height=rasterizer_height,
            )


if __name__ == "__main__":
    tf.test.main()
