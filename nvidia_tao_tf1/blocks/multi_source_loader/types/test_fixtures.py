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

"""Functions for creating test fixtures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types.coordinates2d import (
    Coordinates2DWithCounts,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.images2d import Images2D
from nvidia_tao_tf1.blocks.multi_source_loader.types.images2d_reference import (
    Images2DReference,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.polygon2d_label import (
    Polygon2DLabel,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    FEATURE_CAMERA,
    FEATURE_SESSION,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    LABEL_MAP,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.sequence_example import (
    SequenceExample,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.session import Session
from nvidia_tao_tf1.core.types import Canvas2D, Transform


def make_identity_transform(count, height, width, timesteps=1):
    """Return a batch of identity Transforms for count examples."""
    return Transform(
        canvas_shape=make_canvas2d(count, height, width, timesteps=timesteps),
        color_transform_matrix=tf.stack(
            [tf.eye(4, dtype=tf.float32) for _ in range(count)]
        ),
        spatial_transform_matrix=tf.stack(
            [tf.eye(3, dtype=tf.float32) for _ in range(count)]
        ),
    )


def make_canvas2d(count, height, width, timesteps=1):
    """Return a batch of Canvas2D for count Examples.

    Args:
        count (int or tensor): Number of examples (batch size).
        height (int): Height of the canvas in pixels.
        width (int): Width of the canvas in pixels.
        timesteps (int): Sequence dimension size.

    Returns:
        Canvas2D.
    """
    height_shape = [height]
    width_shape = [width]

    if timesteps:
        height_shape = [timesteps] + height_shape
        width_shape = [timesteps] + width_shape

    add_count = False
    if type(count) == int:
        if count > 0:
            add_count = True
    elif count is not None:
        add_count = True
    if add_count:
        height_shape = [count] + height_shape
        width_shape = [count] + width_shape

    return Canvas2D(height=tf.zeros(height_shape), width=tf.zeros(width_shape))


def make_coordinates2d(
    shapes_per_frame, height, width, coordinates_per_polygon=3, coordinate_values=None
):
    """
    Create a batch of sparse labels.

    Each example can contain a different number of frames.

    Args:
        shapes_per_frame (list[list[int]]): List of lists containing the number of shapes to
            include in each frame. E.g.
            [[1, 2], [4, 4, 4]] - Two examples, where the first one contains 2 frames (first has 1
            shape, second 2) and the second example contains 3 frames (each with 4 shapes).
        height (int): Height of the canvas on which shapes reside.
        width (int): Width of the canvas on which shapes reside.
        coordinates_per_polygon (int): Number of the coordinates in each polygon.
        coordinate_values (list): List containing values of coordinates.

    Returns:
        (Coordinates2DWithCounts): Coordinates generated based on the passed in arguments.
    """
    if shapes_per_frame is None:
        shapes_per_frame = [[1]]
    indices = []
    coordinates = []
    example_count = len(shapes_per_frame)
    max_frame_count = 0
    max_shape_count = 0
    coordinate_counts = []
    for example_index in range(example_count):
        example_frame_shape_counts = shapes_per_frame[example_index]
        frame_count = len(example_frame_shape_counts)
        max_frame_count = max(max_frame_count, frame_count)

        for frame_index in range(frame_count):
            frame_shape_count = example_frame_shape_counts[frame_index]
            max_shape_count = max(frame_shape_count, max_shape_count)
            for shape_index in range(frame_shape_count):
                coordinate_counts.append(coordinates_per_polygon)
                for vertex_index in range(coordinates_per_polygon):
                    coordinates.append(
                        [
                            float(random.randint(0, width) / 2),
                            float(random.randint(0, height)) / 2,
                        ]
                    )
                    for coordinate_index in [0, 1]:
                        indices.append(
                            [
                                example_index,
                                frame_index,
                                shape_index,
                                vertex_index,
                                coordinate_index,
                            ]
                        )

    if coordinate_values is not None:
        assert len(coordinate_values) == len(coordinates)
        coordinates = coordinate_values

    dense_coordinates = tf.constant(coordinates, dtype=tf.float32)
    sparse_indices = tf.constant(indices, dtype=tf.int64)

    dense_shape = tf.constant(
        (
            example_count,
            max_frame_count,
            max_shape_count,
            coordinates_per_polygon,
            2,  # 2D: (x, y)
        ),
        dtype=tf.int64,
    )

    sparse_coordinates = tf.SparseTensor(
        indices=sparse_indices,
        values=tf.reshape(dense_coordinates, (-1,)),
        dense_shape=dense_shape,
    )
    vertices_count = tf.SparseTensor(
        indices=tf.constant(
            [[0, 0, j] for j in range(len(coordinate_counts))], dtype=tf.int64
        ),
        values=tf.constant(coordinate_counts),
        dense_shape=tf.constant([1, 1, len(coordinate_counts)], dtype=tf.int64),
    )

    return Coordinates2DWithCounts(
        coordinates=sparse_coordinates,
        canvas_shape=make_canvas2d(
            example_count, height, width, timesteps=max_frame_count
        ),
        vertices_count=vertices_count,
    )


def make_single_coordinates2d(shape_count, height, width, coordinates_per_polygon=3):
    """
    Create a single sparse label.


    Args:
        shape_count (int): Number of shapes to include.
        height (int): Height of the canvas on which shapes reside.
        width (int): Width of the canvas on which shapes reside.
        coordinates_per_polygon (int): Number of the coordinates in each polygon.

    Returns:
        (Coordinates2DWithCounts): Coordinates generated based on the passed in arguments.
    """
    indices = []
    coordinates = []
    coordinate_counts = []
    for shape_index in range(shape_count):
        coordinate_counts.append(coordinates_per_polygon)
        for vertex_index in range(coordinates_per_polygon):
            coordinates.append(
                [
                    float(random.randint(0, width) / 2),
                    float(random.randint(0, height)) / 2,
                ]
            )
            for coordinate_index in [0, 1]:
                indices.append([shape_index, vertex_index, coordinate_index])

    dense_coordinates = tf.constant(coordinates, dtype=tf.float32)
    sparse_indices = tf.constant(indices, dtype=tf.int64)

    dense_shape = tf.constant(
        (shape_count, coordinates_per_polygon, 2), dtype=tf.int64  # 2D: (x, y)
    )

    sparse_coordinates = tf.SparseTensor(
        indices=sparse_indices,
        values=tf.reshape(dense_coordinates, (-1,)),
        dense_shape=dense_shape,
    )
    vertices_count = tf.SparseTensor(
        indices=tf.constant(
            [[0, j] for j in range(len(coordinate_counts))], dtype=tf.int64
        ),
        values=tf.constant(coordinate_counts),
        dense_shape=tf.constant([1, len(coordinate_counts)], dtype=tf.int64),
    )

    return Coordinates2DWithCounts(
        coordinates=sparse_coordinates,
        canvas_shape=make_canvas2d(0, height, width),
        vertices_count=vertices_count,
    )


def make_tags(tags):
    """
    Create a sparse tensor representing tags associated with shapes.

    A shape can have variable number of tags associated with it. Tag can represent any value like
    a class or attribute.

    Args:
        tags (list[list[list[list[T]]]): List of tags of type T that need to be convertible to
        tf.DType. The depth into this structure encodes (from outermost to innermost):
            0: Examples
            1: Frames within an example.
            2: Shapes within a frame.
            3: Tags associated with a shape

    Returns:
        (tf.SparseTensor): A tensor encoding the passed in tags. The fields of this sparse tensor
            follow the following encoding:
        values:
        indices: A dense tensor of shape [E, F, S, C] and type tf.int64 where:
                E = Example
                F = Frame
                S = Shape that the classes are associated with.
                T = Tags.
        dense_shape: A dense tensor of shape (E, MF, MS, MT) and type T where
                E: Example count
                MF: Maximum frame count in indices.
                MS: Maximum shape count in indices.
                MT: Maximum tag count in indices.
    """
    indices = []
    values = []
    max_frame_count = 0
    max_shape_count = 0
    max_tag_count = 0
    example_count = len(tags)
    for example_index in range(example_count):
        frames = tags[example_index]
        frame_count = len(frames)
        max_frame_count = max(frame_count, max_frame_count)

        for frame_index in range(frame_count):
            shapes = frames[frame_index]
            shape_count = len(shapes)
            max_shape_count = max(shape_count, max_shape_count)

            for shape_index in range(shape_count):
                shape_tags = shapes[shape_index]
                tag_count = len(shape_tags)
                max_tag_count = max(tag_count, max_tag_count)

                values.extend(shape_tags)
                for tag_index in range(tag_count):
                    indices.append([example_index, frame_index, shape_index, tag_index])

    values = tf.constant(values)
    indices = tf.constant(indices, dtype=tf.int64)
    dense_shape = tf.constant(
        (example_count, max_frame_count, max_shape_count, max_tag_count), dtype=tf.int64
    )

    return tf.SparseTensor(
        indices=indices, values=tf.reshape(values, (-1,)), dense_shape=dense_shape
    )


def make_single_tags(tags):
    """
    Create a sparse tensor representing tags associated with shapes.

    A shape can have variable number of tags associated with it. Tag can represent any value like
    a class or attribute.

    Args:
        tags (list[list[T]]): List of tags of type T that need to be convertible to
        tf.DType. The depth into this structure encodes (from outermost to innermost):
            0: Shapes within a frame.
            1: Tags associated with a shape

    Returns:
        (tf.SparseTensor): A tensor encoding the passed in tags. The fields of this sparse tensor
            follow the following encoding:
        values:
        indices: A dense tensor of shape [S, C] and type tf.int64 where:
                S = Shape that the classes are associated with.
                T = Tags.
        dense_shape: A dense tensor of shape (MS, MT) and type T where
                MS: Maximum shape count in indices.
                MT: Maximum tag count in indices.
    """
    indices = []
    values = []
    max_frame_count = 0
    max_shape_count = 0
    max_tag_count = 0

    frame_count = len(tags)
    max_frame_count = max(frame_count, max_frame_count)

    for frame_index in range(frame_count):
        shapes = tags[frame_index]
        shape_count = len(shapes)
        max_shape_count = max(shape_count, max_shape_count)

        for shape_index in range(shape_count):
            shape_tags = shapes[shape_index]
            tag_count = len(shape_tags)
            max_tag_count = max(tag_count, max_tag_count)

            values.extend(shape_tags)
            for tag_index in range(tag_count):
                indices.append([shape_index, tag_index])

    values = tf.constant(values)
    indices = tf.constant(indices, dtype=tf.int64)
    dense_shape = tf.constant((max_shape_count, max_tag_count), dtype=tf.int64)

    return tf.SparseTensor(
        indices=indices, values=tf.reshape(values, (-1,)), dense_shape=dense_shape
    )


def make_polygon2d_label(
    shapes_per_frame,
    shape_classes,
    shape_attributes,
    height,
    width,
    coordinates_per_polygon=3,
    coordinate_values=None,
):
    """
    Make a Polygon2DLabel.

    Args:
        shapes_per_frame (list[list[int]]): List of lists containing the number of shapes to
            include in each frame. E.g.
            [[1, 2], [4, 4, 4]] - Two examples, where the first one contains 2 frames (first has 1
            shape, second 2) and the second example contains 3 frames (each with 4 shapes).
        shape_classes (list[T]): Classes of type T associated with each shape in
            shapes_per_frame. T needs to be a type convertible to tf.DType.
        shape_attributes (list[T]): Attributes of type T associated with each shape in
            shapes_per_frame. T needs to be a type convertible to tf.DType.
        height (int): Height of the canvas on which shapes reside.
        width (int): Width of the canvas on which shapes reside.
        coordinates_per_polygon (int): Number of the coordinates in each polygon.
        coordinate_values (list[float]): Coordinate values of the polygon of type float.

    Returns:
        (Polygon2DLabel): A label with polygons and their associated classes and attributes.
    """
    example_count = len(shapes_per_frame)
    classes = []
    attributes = []
    for example_index in range(example_count):
        example_frame_shapes = shapes_per_frame[example_index]
        example_frame_count = len(example_frame_shapes)

        example_classes = []
        example_attributes = []
        for frame_index in range(example_frame_count):
            frame_classes = []
            frame_attributes = []
            shape_count = example_frame_shapes[frame_index]
            for _ in range(shape_count):
                frame_classes.append(shape_classes)
                frame_attributes.append(shape_attributes)

            example_classes.append(frame_classes)
            example_attributes.append(frame_attributes)
        classes.append(example_classes)
        attributes.append(example_attributes)

    return Polygon2DLabel(
        vertices=make_coordinates2d(
            shapes_per_frame=shapes_per_frame,
            height=height,
            width=width,
            coordinates_per_polygon=coordinates_per_polygon,
            coordinate_values=coordinate_values,
        ),
        classes=make_tags(classes),
        attributes=make_tags(attributes),
    )


def make_images2d(example_count, frames_per_example, height, width):
    """
    Create a batch of Image2D.

    Args:
        example_count (int or tensor): Number of examples (batch size).
        frames_per_example (int): Number of frames within each example.
        height (int): Height of the image in pixels.
        width (int): Width of the image in pixels.

    Returns:
        (Images2D): Images where the images property has a tf.Tensor of type tf.float32 and
            shape [example_count, frames_per_example, 3, height, width].
    """
    shape = []
    if type(example_count) == int:
        if example_count > 0:
            shape.append(example_count)
    elif example_count is not None:
        shape.append(example_count)

    if frames_per_example:
        shape.append(frames_per_example)

    shape.extend([3, height, width])

    return Images2D(
        images=tf.ones(shape, tf.float32),
        canvas_shape=make_canvas2d(example_count, height, width, frames_per_example),
    )


def make_images2d_reference(example_count, frames_per_example, height, width):
    """
    Create a batch of Image2DReferences.

    Args:
        example_count (int or tensor): Number of examples (batch size).
        frames_per_example (int): Number of frames within each example.
        height (int): Height of the image in pixels.
        width (int): Width of the image in pixels.

    Returns:
        (Images2DReference): Image references where the images property has a tf.Tensor of type
            tf.float32 and shape [example_count, frames_per_example, 3, height, width].
    """
    shape = []
    if type(example_count) == int:
        if example_count > 0:
            shape.append(example_count)
    elif example_count is not None:
        shape.append(example_count)

    if frames_per_example:
        shape.append(frames_per_example)

    path = tf.constant("test_path", dtype=tf.string)
    extension = tf.constant(".fp16", dtype=tf.string)
    if shape:
        path = tf.broadcast_to(path, shape)
        extension = tf.broadcast_to(extension, shape)

    return Images2DReference(
        path=path,
        extension=extension,
        canvas_shape=make_canvas2d(example_count, height, width, frames_per_example),
        input_height=tf.constant(height, dtype=tf.int32),
        input_width=tf.constant(width, dtype=tf.int32),
    )


def make_example(
    height,
    width,
    example_count=1,
    shapes_per_frame=None,
    coordinates_per_polygon=3,
    coordinate_values=None,
    use_images2d_reference=False,
):
    """
    Create a batch of SequenceExamples.

    Args:
        height (int): Height of the images and labels to create.
        width (int): Width of the images and labels to create.
        example_count (int or tensor): Number of examples (batch size).
        shapes_per_frame (list[list[int]]): List of lists containing the number of shapes to
            include in each frame. E.g.
            [[1, 2], [4, 4, 4]] - Two examples, where the first one contains 2 frames (first has 1
            shape, second 2) and the second example contains 3 frames (each with 4 shapes).
        coordinates_per_polygon (int): Number of the coordinates in each polygon.
        coordinate_values (list): List containing values of coordinates.
        use_images2d_reference (boolean): If True, construct examples with Images2DReference.
            If False, construct with Images2D.

    Returns:
        (SequenceExample): Sequence examples configured based on parameters.
    """
    image_func = make_images2d_reference if use_images2d_reference else make_images2d
    return SequenceExample(
        instances={
            FEATURE_CAMERA: image_func(
                example_count=example_count,
                frames_per_example=1,
                height=height,
                width=width,
            ),
            FEATURE_SESSION: Session(
                uuid=tf.constant("session_uuid"),
                camera_name=tf.constant("camera_name"),
                frame_number=tf.constant(0),
            ),
        },
        labels={
            LABEL_MAP: Polygon2DLabel(
                vertices=make_coordinates2d(
                    shapes_per_frame=shapes_per_frame,
                    height=height,
                    width=width,
                    coordinates_per_polygon=coordinates_per_polygon,
                    coordinate_values=coordinate_values,
                ),
                classes=make_tags([[[["lane"]]]]),
                attributes=make_tags([[[["left", "exit"]]]]),
            )
        },
    )


def make_example_3d(height, width, label_name=LABEL_MAP):
    """
    Create a SequenceExample that does not have a time and batch dimension.

    Args:
        height (int): Height of the images and labels to create.
        width (int): Width of the images and labels to create.
        label_name (string): Name of the Polygon2DLabel to create.

    Returns:
        (SequenceExample): Sequence examples configured based on parameters.
    """
    return SequenceExample(
        instances={
            FEATURE_CAMERA: make_images2d(
                example_count=0, frames_per_example=0, height=height, width=width
            ),
            FEATURE_SESSION: Session(
                uuid=tf.constant("session_uuid"),
                camera_name=tf.constant("camera_name"),
                frame_number=tf.constant(0),
            ),
        },
        labels={
            label_name: Polygon2DLabel(
                vertices=make_single_coordinates2d(3, height=height, width=width),
                classes=make_single_tags([["lane", "lane", "lane"]]),
                attributes=make_single_tags([[["left", "exit", "entry"]]]),
            )
        },
    )
