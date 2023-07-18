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

"""Class for parsing labels produced by the lanenet json_converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from nvidia_tao_tf1.blocks.multi_source_loader import types
from nvidia_tao_tf1.core.processors import ParseExampleProto


# TODO(vkallioniemi): Move (to lanenet?) once we've figured out what parts of TFRecordsDataSource
# can be shared.
class LaneLabelParser(object):
    """Parse tf.train.Example protos into lanenet Examples."""

    def __init__(
        self,
        image_dir,
        extension,
        height,
        width,
        max_height,
        max_width,
        should_normalize_labels=True,
    ):
        """
        Construct a parser for lane labels.

        Args:
            image_dir (str): Path to the directory where images are contained.
            extension (str): Image extension for images that get loaded (
                ".fp16", ".png", ".jpg" or ".jpeg").
            height (int): Height of images and labels.
            width (int): Width of images and labels.
            max_height (int): Height to pad image to if necessary.
            max_width (int): Width to pad image to if necessary.
            should_normalize_labels(bool): Whether or not the datasource should normalize the label
                coordinates.
        """
        assert height > 0
        assert width > 0
        self._image_dir = image_dir
        self._height = height
        self._width = width
        self._extension = extension
        self._parse_example = self._make_example_parser()
        self._should_normalize = should_normalize_labels
        self._max_height = max_height
        self._max_width = max_width

    def __call__(self, tfrecord):
        """
        Parse a tf.train.Example.

        Returns:
            (types.SequenceExample) Example compatible with Processors.
        """
        return self._parse(tfrecord)

    def _parse_image_reference(self, image_dir, image_id, extension):
        directory = os.path.normpath(image_dir)
        file_name = tf.strings.join([image_id, extension])
        frame_path = tf.strings.join([directory, file_name], os.sep)
        return frame_path

    def _parse(self, tfrecord):
        tfexample = self._parse_example(tfrecord)
        extension = self._extension
        frame_path = self._parse_image_reference(
            self._image_dir, tfexample["id"], extension
        )

        return types.SequenceExample(
            instances={
                types.FEATURE_CAMERA: types.Images2DReference(
                    path=frame_path,
                    extension=extension,
                    canvas_shape=types.Canvas2D(
                        height=tf.ones(self._max_height), width=tf.ones(self._max_width)
                    ),
                    input_height=[self._height],
                    input_width=[self._width],
                ),
                types.FEATURE_SESSION: types.Session(
                    uuid=tfexample["sequence_name"],
                    camera_name=tfexample["camera_location"],
                    frame_number=tfexample["frame_number"],
                ),
            },
            labels={
                types.LABEL_MAP: self._extract_polygon(
                    tfexample, image_height=self._height, image_width=self._width
                )
            },
        )

    def _make_example_parser(self):
        features = {
            "id": tf.io.FixedLenFeature([], dtype=tf.string),
            "width": tf.io.FixedLenFeature([], dtype=tf.int64),
            "height": tf.io.FixedLenFeature([], dtype=tf.int64),
            "target/classifier": tf.io.VarLenFeature(dtype=tf.string),
            "target/attributes": tf.io.VarLenFeature(dtype=tf.string),
            "target/polygon/length": tf.io.VarLenFeature(dtype=tf.int64),
            "target/polygon/x": tf.io.VarLenFeature(dtype=tf.float32),
            "target/polygon/y": tf.io.VarLenFeature(dtype=tf.float32),
            "target/polygonAttributes": tf.io.VarLenFeature(dtype=tf.string),
            "target/polygonAttributesCount": tf.io.VarLenFeature(dtype=tf.int64),
            "sequence_name": tf.io.FixedLenFeature([], dtype=tf.string),
            "frame_number": tf.io.FixedLenFeature([], dtype=tf.int64),
            "camera_location": tf.io.FixedLenFeature([], dtype=tf.string),
        }
        return ParseExampleProto(features=features, single=True)

    def _normalize_coordinates(
        self, x, y, polygon_width, polygon_height, image_width, image_height
    ):
        if self._should_normalize:
            x /= tf.cast(polygon_width, tf.float32)
            y /= tf.cast(polygon_height, tf.float32)

            x *= image_width
            y *= image_height

        return tf.stack([x, y], axis=1)

    def _extract_polygon(self, example, image_height, image_width):
        dense_coordinates = self._normalize_coordinates(
            x=example["target/polygon/x"],
            y=example["target/polygon/y"],
            polygon_width=example["width"],
            polygon_height=example["height"],
            image_height=image_height,
            image_width=image_width,
        )

        coordinates_per_polygon = example["target/polygon/length"]
        num_polygons = tf.shape(input=coordinates_per_polygon)[0]

        # TODO(ehall): Eventually this part will be removed in favor of adding all attributes below.
        # TODO(mlehr): Switched the if/elif, it is not working for pathnet with 'polygonAttributes'.
        # modulus `PathGenerator` does not support multiple attributes per polyline.
        if "target/attributes" in example:
            dense_attribute_ids = example["target/attributes"]
            sparse_attributes = types.vector_and_counts_to_sparse_tensor(
                dense_attribute_ids, tf.ones_like(dense_attribute_ids, dtype=tf.int64)
            )
        elif "target/polygonAttributes" in example:
            sparse_attributes = types.vector_and_counts_to_sparse_tensor(
                example["target/polygonAttributes"],
                example["target/polygonAttributesCount"],
            )
        else:
            raise ValueError(
                "Invalid TFRecords - neither attributes or polygonAttributes present."
            )

        dense_classes = example["target/classifier"]
        # Note: we rely on json converter script to ensure that each
        # polygon always contains a single class.
        # TODO(vkallioniemi): This can be done cheaper given the ^^ assumption.
        classes_per_polygon = tf.ones_like(dense_classes, dtype=tf.int64)
        sparse_classes = types.vector_and_counts_to_sparse_tensor(
            dense_classes, classes_per_polygon
        )

        # Turn coordinates to 3D sparse tensors of shape [S, V, C]
        # where S=Shape/Polygon, V=Vertex and C=Coordinate
        sparse_coordinates = types.sparsify_dense_coordinates(
            dense_coordinates, coordinates_per_polygon
        )

        return types.Polygon2DLabel(
            vertices=types.Coordinates2DWithCounts(
                coordinates=sparse_coordinates,
                canvas_shape=types.Canvas2D(
                    height=tf.ones(self._max_height), width=tf.ones(self._max_width)
                ),
                vertices_count=tf.SparseTensor(
                    indices=tf.reshape(
                        tf.cast(tf.range(num_polygons), dtype=tf.int64), [-1, 1]
                    ),
                    values=coordinates_per_polygon,
                    dense_shape=[num_polygons],
                ),
            ),
            classes=sparse_classes,
            attributes=sparse_attributes,
        )
