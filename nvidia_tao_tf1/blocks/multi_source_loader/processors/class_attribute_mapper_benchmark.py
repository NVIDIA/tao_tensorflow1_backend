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
"""ClassAttributeMapper benchmark suite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from parameterized import parameterized
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors.class_attribute_mapper import (
    ClassAttributeMapper,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.test_fixtures import (
    make_polygon2d_label,
)
from modulus.utils import test_session


class ClassAttributeMapperBenchmark(tf.test.Benchmark):
    """ClassAttributeMapper benchmark suite."""

    ITERATIONS = 100

    def _benchmark_class_attribute_mapper(
        self,
        sess,
        class_mapping_count,
        match_class_count,
        match_attribute_count,
        example_count,
        frame_count,
        polygon_count,
        attribute_count,
    ):
        """
        Build class_attribute_mapper and polygon_2d_label.

        Args:
            sess (tf.Session()): Session to run the benchmark.
            class_mapping_count (int): Number of classes in the class_mapping table.
            match_class_count (int): Number of class names in each match_any_class row.
            match_attribute_count (int): Number of attriute names in match_all/any_attribute row.
            example_count (int): Number of examples in each batch.
            frame_count (int): Number of frames in each example.
            polygon_count (int): Number of polygons in each image.
            attribute_count (int): Number of attributes in each polygon (class).
        """
        # Build class mapping table.
        class_mapping = []
        for index in range(class_mapping_count):
            _class = {}
            _class["match_any_class"] = ["path" for _ in range(match_class_count)]
            _class["match_any_attribute"] = [
                "edge" for _ in range(match_attribute_count)
            ]
            _class["match_all_attributes"] = [
                "EDGE" for _ in range(match_attribute_count)
            ]
            _class["class_name"] = "class_{}".format(index)
            _class["class_id"] = index
            class_mapping.append(_class)

        # Attribute mapping is implemend as lookup table, shoud be fast.
        attribute_mapping = [{"name": "attr1", "id": 1}, {"name": "attr2", "id": 2}]

        # Build polygon2d_lable.
        shapes_per_frame = []
        for _ in range(example_count):
            shapes_per_frame.append([polygon_count for _ in range(frame_count)])

        attributes = ["edge" for _ in range(attribute_count)]
        labels2d = make_polygon2d_label(
            shapes_per_frame=shapes_per_frame,
            shape_classes=["path"],
            shape_attributes=attributes,
            height=940,
            width=504,
            coordinates_per_polygon=3,
        )

        mapper = ClassAttributeMapper(
            class_mapping, "Default", -1, attribute_mapping, -1
        )
        mappered_labels2d = mapper(labels2d)
        run_op = tf.group(
            mappered_labels2d.classes.values, mappered_labels2d.attributes.values
        )
        sess.run(tf.compat.v1.tables_initializer())
        self.run_op_benchmark(
            sess=sess,
            op_or_tensor=run_op,
            min_iters=self.ITERATIONS,
            store_trace=True,
            store_memory_usage=True,
        )

    @parameterized.expand(
        [
            [1, "/cpu:0"],
            [8, "/cpu:0"],
            [16, "/cpu:0"],
            [32, "/cpu:0"],
            [1, "/gpu:0"],
            [8, "/gpu:0"],
            [16, "/gpu:0"],
            [32, "/gpu:0"],
        ]
    )
    def benchmark_attribute_mapper_example_count(self, example_count, device_placement):
        """Benchmark different numbers of examples."""
        print("example_count {} device {}.".format(example_count, device_placement))
        with tf.device(device_placement), test_session(
            allow_soft_placement=True
        ) as sess:
            self._benchmark_class_attribute_mapper(
                sess=sess,
                class_mapping_count=10,
                match_class_count=10,
                match_attribute_count=20,
                example_count=example_count,
                frame_count=3,
                polygon_count=5,
                attribute_count=3,
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
    def benchmark_attribute_mapper_attribute_count(
        self, attribute_count, device_placement
    ):
        """Benchmark different numbers of attribute_count."""
        print("attribute_count {} device {}.".format(attribute_count, device_placement))
        with tf.device(device_placement), test_session(
            allow_soft_placement=True
        ) as sess:
            self._benchmark_class_attribute_mapper(
                sess=sess,
                class_mapping_count=10,
                match_class_count=10,
                match_attribute_count=20,
                example_count=32,
                frame_count=3,
                polygon_count=5,
                attribute_count=attribute_count,
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
    def benchmark_attribute_mapper_class_mapping_count(
        self, class_mapping_count, device_placement
    ):
        """Benchmark different numbers of class_mapping_count."""
        print(
            "class_mapping_count {} device {}.".format(
                class_mapping_count, device_placement
            )
        )
        with tf.device(device_placement), test_session(
            allow_soft_placement=True
        ) as sess:
            self._benchmark_class_attribute_mapper(
                sess=sess,
                class_mapping_count=class_mapping_count,
                match_class_count=10,
                match_attribute_count=20,
                example_count=32,
                frame_count=3,
                polygon_count=5,
                attribute_count=3,
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
    def benchmark_attribute_mapper_match_class_count(
        self, match_class_count, device_placement
    ):
        """Benchmark different numbers of match_class_count."""
        print(
            "match_class_count {} device {}.".format(
                match_class_count, device_placement
            )
        )
        with tf.device(device_placement), test_session(
            allow_soft_placement=True
        ) as sess:
            self._benchmark_class_attribute_mapper(
                sess=sess,
                class_mapping_count=10,
                match_class_count=match_class_count,
                match_attribute_count=20,
                example_count=32,
                frame_count=3,
                polygon_count=5,
                attribute_count=3,
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
    def benchmark_attribute_mapper_match_attribute_count(
        self, match_attribute_count, device_placement
    ):
        """Benchmark different numbers of match_attribute_count."""
        print(
            "match_attribute_count {} device {}.".format(
                match_attribute_count, device_placement
            )
        )
        with tf.device(device_placement), test_session(
            allow_soft_placement=True
        ) as sess:
            self._benchmark_class_attribute_mapper(
                sess=sess,
                class_mapping_count=10,
                match_class_count=10,
                match_attribute_count=match_attribute_count,
                example_count=32,
                frame_count=3,
                polygon_count=5,
                attribute_count=3,
            )


if __name__ == "__main__":
    tf.test.main()
