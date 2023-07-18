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
"""Processor for applying spatial/temporal/color transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.data_format import CHANNELS_FIRST
from nvidia_tao_tf1.blocks.multi_source_loader.data_format import CHANNELS_LAST
from nvidia_tao_tf1.blocks.multi_source_loader.processors.processor import (
    Processor,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Example,
    FEATURE_CAMERA,
    LABEL_MAP,
    PolygonLabel,
    SequenceExample,
)
from nvidia_tao_tf1.core.processors import ColorTransform, PolygonTransform, SpatialTransform
from nvidia_tao_tf1.core.processors import Processor as ModulusProcessor
from nvidia_tao_tf1.core.types import Canvas2D, Transform


class TransformProcessor(Processor):
    """Processor that applies spatial and color transformations."""

    def __init__(self, transform):
        """
        Construct processor that uses a transformation.

        Args:
            transform (Transform): Transform to apply.
        """
        super(TransformProcessor, self).__init__()
        if transform is None:
            raise ValueError("Transform should not be None")

        self._transform = transform
        self._is_built = False
        # Default to channels last for backward compatibility of unit tests.
        self.data_format = CHANNELS_LAST

    def _build(self, data_format):
        self._color_transform = ColorTransform(
            min_clip=0.0, max_clip=1.0, data_format=str(data_format)
        )
        self._polygon_transform = PolygonTransform()
        self._spatial_transform = SpatialTransform(
            method="bilinear",
            background_value=0.5,
            data_format=str(data_format),
            verbose=False,
        )
        self._is_built = True

    @property
    def supported_formats(self):
        """Data formats supported by this processor.

        Returns:
            data_formats (list of 'DataFormat'): Input data formats that this processor supports.
        """
        return [CHANNELS_FIRST, CHANNELS_LAST]

    def can_compose(self, other):
        """
        Determine whether two processors can be composed into a single one.

        Args:
            other (Processor): Other processor instance.

        Returns:
            (Boolean): True if the other processor is also a TransformProcessor.
        """
        return isinstance(other, TransformProcessor)

    def compose(self, other):
        """Compose two TransformProcessors into a single one."""
        if not isinstance(other, TransformProcessor):
            raise TypeError(
                "Tried to compose TransformProcessor with type: {}".format(type(other))
            )

        composite = CompositeTransform([self._transform, other._transform])

        return TransformProcessor(transform=composite)

    def process(self, example):
        """
        Process examples by applying transformations in sequence..

        Args:
            example (Example): Examples to process in format specified by data_format.

        Returns:
            example (TransformedExample): Example with all transformations applied.
        """

        def _get_shape_as_list(tensor):
            # Try static shape inference first. If it fails, use run time shape.
            shape = tensor.shape.as_list()
            runtime_shape = tf.shape(input=tensor)
            for i, dim in enumerate(shape):
                if dim is None:
                    shape[i] = runtime_shape[i]
            return shape

        if not self._is_built:
            self._build(self.data_format)

        if isinstance(example, SequenceExample):
            # Depending on where in the data loader pipeline TransformProcessor is called,
            # feature_camera can be either Images2D or Images2DReference. Both namedtuples
            # have canvas_shape member, so we can use the same code for both.
            feature_camera = example.instances[FEATURE_CAMERA]

            # Get shapes (static shape if known, run time shape otherwise).
            canvas_height_shape = _get_shape_as_list(feature_camera.canvas_shape.height)
            canvas_width_shape = _get_shape_as_list(feature_camera.canvas_shape.width)

            # Check the number of dimensions:
            # 1: Spatial dimension only.
            # 2: Sequence and spatial dimensions.
            # 3: Batch, sequence, and spatial dimensions.
            rank = len(canvas_height_shape)
            assert 1 <= rank <= 3

            # Infer batch shape, width and height.
            batch_shape = [canvas_height_shape[0]] if rank == 3 else None
            height = canvas_height_shape[-1]
            width = canvas_width_shape[-1]

            identity_transformation = Transform(
                canvas_shape=Canvas2D(height=height, width=width),
                color_transform_matrix=tf.eye(
                    4, batch_shape=batch_shape, dtype=tf.float32
                ),
                spatial_transform_matrix=tf.eye(
                    3, batch_shape=batch_shape, dtype=tf.float32
                ),
            )

            transformation = self._transform(identity_transformation)
            # NOTE: We're encoding the canvas height as a vector of shape [Height] and width as
            # a vector of shape [Width]. This is done so that we can use TF static shapes to
            # pass shape information here as Python values (i.e. at graph construction time.)
            # Being able to set the shape of the transformed images here enables us to decouple
            # the dataloader and the estimator/model. The shape of all images coming out of
            # the dataloader will be fully defined at graph construction time.

            # Replace canvas width and height by transformed values, propagate the other
            # dimensions as is.
            canvas_height_shape[-1] = transformation.canvas_shape.height
            canvas_width_shape[-1] = transformation.canvas_shape.width

            transformation = Transform(
                canvas_shape=Canvas2D(
                    height=tf.zeros(canvas_height_shape),
                    width=tf.zeros(canvas_width_shape),
                ),
                color_transform_matrix=transformation.color_transform_matrix,
                spatial_transform_matrix=transformation.spatial_transform_matrix,
            )
            return example.transform(transformation)

        # Legacy LaneNet dataloader expect transformations to be applied
        # here. TODO(vkallioniemi): remove this functionality once we delete
        # the old dataloader.
        axis = self.data_format.axis_4d
        frame = example.instances[FEATURE_CAMERA]
        input_shape = frame.get_shape().as_list()

        identity_transformation = Transform(
            canvas_shape=Canvas2D(
                height=input_shape[axis.row], width=input_shape[axis.column]
            ),
            color_transform_matrix=tf.eye(4, dtype=tf.float32),
            spatial_transform_matrix=tf.eye(3, dtype=tf.float32),
        )

        transformation = self._transform(identity_transformation)
        return self._apply_transformation_to_example(transformation, example)

    def _color_transform_frames(self, frames, color_transform_matrix):
        """Return new frames by applying the color transform matrix against input frames."""
        axis = self.data_format.axis_4d
        input_shape = tf.shape(input=frames)
        batch_size = input_shape[axis.batch]

        color_transform_matrices = tf.tile(
            tf.expand_dims(color_transform_matrix, axis=0), [batch_size, 1, 1]
        )
        return self._color_transform(frames, ctms=color_transform_matrices)

    def _spatial_transform_frames(self, frames, spatial_transform_matrix, canvas_shape):
        """Return new frames by applying the spatial transform matrix against input frames."""
        axis = self.data_format.axis_4d
        input_shape = tf.shape(input=frames)
        batch_size = input_shape[axis.batch]

        stms = tf.tile(
            tf.expand_dims(spatial_transform_matrix, axis=0), [batch_size, 1, 1]
        )

        return self._spatial_transform(
            frames, stms=stms, shape=(int(canvas_shape.height), int(canvas_shape.width))
        )

    def _spatial_transform_label(self, label, spatial_transform_matrix):
        """Return new PolygonLabel by applying the spatial transform matrix against input label."""
        transformed_polygons = self._polygon_transform(
            label.polygons, spatial_transform_matrix
        )

        return PolygonLabel(
            polygons=transformed_polygons,
            vertices_per_polygon=label.vertices_per_polygon,
            class_ids_per_polygon=label.class_ids_per_polygon,
            attributes_per_polygon=label.attributes_per_polygon,
            polygons_per_image=label.polygons_per_image,
            attributes=label.attributes,
            attribute_count_per_polygon=label.attribute_count_per_polygon,
        )

    def _apply_transformation_to_example(self, transformation, example):
        """Return new Example by applying the transformation against the input example."""
        instances = example.instances
        labels = example.labels

        if FEATURE_CAMERA in example.instances:
            frames = example.instances[FEATURE_CAMERA]
            if transformation.spatial_transform_matrix is not None:
                frames = self._spatial_transform_frames(
                    frames,
                    transformation.spatial_transform_matrix,
                    canvas_shape=transformation.canvas_shape,
                )

            if transformation.color_transform_matrix is not None:
                frames = self._color_transform_frames(
                    frames, transformation.color_transform_matrix
                )

            instances[FEATURE_CAMERA] = frames

        if LABEL_MAP in example.labels:
            polygons = example.labels[LABEL_MAP]
            polygons = self._spatial_transform_label(
                polygons, transformation.spatial_transform_matrix
            )
            labels[LABEL_MAP] = polygons

        return Example(instances=instances, labels=labels)


class CompositeTransform(ModulusProcessor):
    """Sequence of transform processors composed into one."""

    def __init__(self, transforms, **kwargs):
        """Construct a pipeline of transforms applied in sequence."""
        super(CompositeTransform, self).__init__(**kwargs)
        self._transforms = transforms

    def __len__(self):
        """Return number of transforms this pipeline consists of."""
        return len(self._transforms)

    def call(self, transformation):
        """Produce a transformation by applying all transforms in sequence."""
        output = transformation
        for transform in self._transforms:
            output = transform(output)

        return output
