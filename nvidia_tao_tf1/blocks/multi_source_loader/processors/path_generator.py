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

"""Class for the path generation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.processors.polyline_clipper import (
    PolylineClipper,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Coordinates2DWithCounts,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types import (
    Example,
    FEATURE_CAMERA,
    LABEL_MAP,
)
from nvidia_tao_tf1.core.coreobject import TAOObject, save_args
from modulus.processors import PathGenerator as MaglevPathGenerator

NUM_ATTRIBUTES_FOR_CLASS = 1


class PathGenerator(TAOObject):
    """PathGenerator loads and stores label processor configuration specific to PathNet."""

    @save_args
    def __init__(
        self,
        nclasses,
        class_name_to_id,
        equidistant_interpolation=False,
        using_invalid_path_class=False,
        prior_assignment_constraint=False,
        npath_attributes=0,
        path_attribute_name_to_id=None,
        edges_per_path=2,
    ):
        """
        Construct PathGenerator.

        Args:
            nclasses (int): Number of classes.
            class_name_to_id (dict): Contains mapping between output class name and output id.
            equidistant_interpolation (bool): If ``True`` interpolates points along the path edges
                                              that are equally spaced. If ``False``, uses
                                              logarithmic spacing that gets closer with increasing
                                              rank order of the path points.
            using_invalid_path_class (bool): If `True`, then the invalid_path class is being used.
                                             Every prior with no ground truth assigned is assigned
                                             to this class. Default: ``False``.
            prior_assignment_constraint (bool): If ``True`` enable the prior assignment constraint
                                                strategy to group priors to classes according
                                                to the scheme.
                                                If `False` allow no grouping of priors to classes.
                                                default: False.
            For details about prior assignment constraint, please go to following file:
            (ai-infra/moduluspy/lib/src/generate_path_from_edges/generate_path_from_edges.cc) and
            (https://confluence.nvidia.com/display/DLVEH/Adding+Exit+Paths+to+Pathnet
            #AddingExitPathstoPathnet-Majorconceptintroducedforimprovement:Priorconstrainting:).
            npath_attributes (int): Number of path attributes.
            path_attribute_name_to_id (dict): Contains mapping between path attribute name and id.
            edges_per_path (int): Number of edges per path. 3 with center rail included else 2.
        """
        super(PathGenerator, self).__init__()
        self.nclasses = nclasses
        self.class_name_to_id = class_name_to_id
        self._equidistant_interpolation = equidistant_interpolation
        self._polyline_clipper = PolylineClipper(vertex_count_per_polyline=-1)
        self._path_priors = 0
        self._points_priors = 0
        self._prior_assignment_constraint = prior_assignment_constraint
        self._using_invalid_path_class = using_invalid_path_class
        self._edges_per_path = edges_per_path
        self.npath_attributes = npath_attributes
        self.path_attribute_name_to_id = path_attribute_name_to_id

        self._validate_class_name_to_id()

    def _validate_class_name_to_id(self):
        if len(self.class_name_to_id) == 0:
            raise ValueError(
                "There should be at least one output class "
                "in the target encoder. Found {}".format(len(self.class_name_to_id))
            )
        class_names = set()
        class_names = {class_name for class_name in self.class_name_to_id.keys()}
        if len(self.class_name_to_id) > len(class_names):
            raise ValueError("Duplicated output class name in the target encoder.")

    def _zip_point_coordinates(
        self, x, y, scale_x=1.0, scale_y=1.0, translate_x=0, translate_y=0
    ):
        """
        Utility function for normalizing/adjusting and zipping polyline coordinates.

        Args:
            x (float): x coordinate of the vertices of the polylines.
            y (float): y coordinate of the vertices of the polylines.
            scale_x (float): Factor for normalizing the x coordinates.
            scale_y (float): Factor for normalizing the y coordinates.
            translate_x (float): amount to translate the x coordinate.
            translate_y (float): amount to translate the y coordinate.

        Returns:
            polylines (tf.Tensor) with the normalized/adjusted vertices.
        """
        x *= tf.cast(scale_x, tf.float32)
        y *= tf.cast(scale_y, tf.float32)
        x -= tf.cast(translate_x, tf.float32)
        y -= tf.cast(translate_y, tf.float32)
        return tf.stack([x, y], axis=1)

    def _reorder_attributes(self, attributes_per_polyline, npath_attributes=0):
        """
        Enforce an order for attributes_per_polyline.

        Edge attributes are followed by path attributes.
        generate_path_from_edges.cc expects this order.

        Args:
            attributes_per_polyline (tf.Tensor): Tensor of shape [L,1] and type tf.int32.
                                                 Attributes for each polyline.
                                                 Every (npath_attributes + NUM_ATTRIBUTES_FOR_CLASS)
                                                 elements belong to one polyline.
                                                 The number of polylines is
                                                  L / (npath_attribues + NUM_ATTRIBUTES_FOR_CLASS).
            npath_attributes (int): The number of path attributes.
        Returns:
            attributes_per_polyline (tf.Tensor): Tensor of shape [L,1] and type tf.int32.
                                                 The first L elements are edge attributes.
                                                 The Nth L elements are the (N-1)th path attributes.

        """
        if npath_attributes > 0:

            nattribute_groups = npath_attributes + NUM_ATTRIBUTES_FOR_CLASS

            # L / nattribute_groups is the number of polylines.
            L = tf.shape(input=attributes_per_polyline)[0]
            base_index = tf.cast(
                nattribute_groups * tf.range(L / nattribute_groups), tf.int32
            )
            attributes_per_polyline_reshaped = tf.reshape(
                attributes_per_polyline, [L / nattribute_groups, nattribute_groups]
            )

            edge_attribute_index_within_a_polyline = tf.cast(
                tf.math.argmin(
                    input=tf.math.abs(attributes_per_polyline_reshaped), axis=1
                ),
                tf.int32,
            )

            # path_attribute index.
            path_attribute_index_within_a_polyline = (
                npath_attributes - edge_attribute_index_within_a_polyline
            )

            edge_attribute_index_1d = (
                base_index + edge_attribute_index_within_a_polyline
            )
            path_attribute_index_1d = (
                base_index + path_attribute_index_within_a_polyline
            )

            edge_attributes = tf.gather(
                attributes_per_polyline, edge_attribute_index_1d
            )

            path_attributes = tf.gather(
                attributes_per_polyline, path_attribute_index_1d
            )

            attributes_per_polyline = tf.reshape(
                tf.stack([edge_attributes, path_attributes]), [-1]
            )

        return attributes_per_polyline

    def encode_dense(self, example, priors_generator):
        """
        Encode dense path labels as targets compatible with PathNet loss and metrics.

        Path labels are PolygonLabel type.

        Args:
            example (Example): Example to apply the path generation operation on.
            priors_generator (PriorsGenerator): PriorsGenerator object.
        Returns:
            (`Example`): Example with the encoded labels.
                         Encoded labels are 2D tensors of type tf.float32 and
                         shape [C, P*4+TAG+SCALING] where:
                         C: Number of path priors.
                         P: Number of points per prior.
                         4: Length of point pair for the left and right edges (
                         currently encoded as (left_x, left_y, right_x, right_y))
                         TAG: Number of classes.
                         SCALING: Scaling factor.
        """
        frames = example.instances[FEATURE_CAMERA]
        labels = example.labels[LABEL_MAP]
        _, _, height, width = frames.get_shape().as_list()

        # Encode the paths from the image.
        polylines = labels.polygons
        vertices_per_polyline = labels.vertices_per_polygon
        class_ids_per_polyline = labels.class_ids_per_polygon
        attributes_per_polyline = labels.attributes_per_polygon

        target = self._encode_dense(
            priors_generator,
            height,
            width,
            height,
            width,
            polylines,
            vertices_per_polyline,
            class_ids_per_polyline,
            attributes_per_polyline,
        )
        labels = example.labels
        labels[LABEL_MAP] = target
        return Example(instances=example.instances, labels=labels)

    def _encode_dense(
        self,
        priors_generator,
        height,
        width,
        canvas_height,
        canvas_width,
        polylines,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline,
    ):
        image_boundaries = tf.constant(
            [
                [0, 0],
                [0, canvas_height],
                [canvas_width, canvas_height],
                [canvas_width, 0],
                [0, 0],
            ],
            tf.float32,
        )

        maglev_path_generator = MaglevPathGenerator(
            width=width,
            height=height,
            nclasses=self.nclasses,
            nall_priors=priors_generator.nall_priors,
            points_per_prior=priors_generator.points_per_prior,
            npath_attributes=self.npath_attributes,
            prior_threshold=priors_generator.prior_threshold,
            equal_spacing=self._equidistant_interpolation,
            prior_assignment_constraint=self._prior_assignment_constraint,
            using_invalid_path_class=self._using_invalid_path_class,
            edges_per_path=self._edges_per_path,
        )

        attributes_per_polyline = self._reorder_attributes(
            attributes_per_polyline, self.npath_attributes
        )

        # Clip polygons to model input boundaries.
        (
            clipped_polylines,
            _,
            clipped_vertices_per_polyline,
            clipped_class_ids_per_polyline,
            clipped_attributes_per_polyline,
        ) = self._polyline_clipper.clip(
            polylines,
            vertices_per_polyline,
            class_ids_per_polyline,
            attributes_per_polyline,
            maintain_vertex_number=False,
            polygon_mask=image_boundaries,
        )

        # Normalize the polyline before sending to label processor.
        x, y = tf.unstack(clipped_polylines, axis=1)
        clipped_polylines = self._zip_point_coordinates(
            x, y, scale_x=1.0 / canvas_width, scale_y=1.0 / canvas_height
        )
        target = maglev_path_generator(
            clipped_polylines,
            priors_generator.priors,
            clipped_vertices_per_polyline,
            clipped_class_ids_per_polyline,
            clipped_attributes_per_polyline,
        )

        return target

    def encode_sparse(self, labels2d, priors_generator, image_shape, temporal=None):
        """
        Encode sparse path labels as targets compatible with PathNet loss and metrics.

        Path labels are Polygon2DLabel type.

        Args:
            labels2d (Polygon2DLabel): A label containing 2D polygons/polylines and their
                associated classes and attributes. The first two dimensions of each tensor
                that this structure contains should be batch/example followed by a frame/time
                dimension. The rest of the dimensions encode type specific information. See
                Polygon2DLabel documentation for details.
            priors_generator (PriorsGenerator): PriorsGenerator object.
            image_shape (FrameShape): Namedtuple with height, width and channels.
            temporal (int): (optional) Temporal dimension of the batch. If not available,
                inferred from labels2d.vertices.canvas_shape.height.shape.

        Returns:
            (tf.Tensor): Dense tensor with encoded labels.
                         Encoded labels are 4D tensors of type tf.float32 and shape
                         [B, T, C, P*4+TAG+SCALING] where:
                         B: Batch size.
                         T: Number of frames.
                         C: Number of path priors.
                         P: Number of points per prior.
                         4: Length of point pair for the left and right edges (
                         currently encoded as (left_x, left_y, right_x, right_y))
                         TAG: Number of classes.
                         SCALING: Scaling factor.
        """
        assert isinstance(
            labels2d.vertices, Coordinates2DWithCounts
        ), "labels2d.vertices must be of type Coordinates2DWithCounts, was: {}".format(
            type(labels2d.vertices)
        )
        example_count = labels2d.vertices.canvas_shape.height.shape.as_list()[0]
        if temporal is None:
            max_timesteps_in_example = labels2d.vertices.canvas_shape.height.shape.as_list()[
                1
            ]
        else:
            max_timesteps_in_example = temporal

        polylines = labels2d.vertices.coordinates
        vertices_per_polyline = labels2d.vertices.vertices_count
        class_ids_per_polyline = labels2d.classes
        attributes_per_polyline = labels2d.attributes

        def _dense_wrapper(i, j):
            def _slice(spt, d):
                return tf.sparse.slice(
                    spt,
                    [i, j]
                    + [0]
                    * (d - 2),  # probably there is better way to do than use the d
                    [1, 1] + [spt.dense_shape[a] for a in range(2, d)],
                )

            i_polylines = tf.reshape(_slice(polylines, d=5).values, [-1, 2])
            i_class_ids_per_polyline = tf.reshape(
                _slice(class_ids_per_polyline, d=4).values, [-1]
            )
            i_attrs_per_polyline = tf.reshape(
                _slice(attributes_per_polyline, d=4).values, [-1]
            )
            i_vertices_per_polyline = tf.reshape(
                _slice(vertices_per_polyline, d=3).values, [-1]
            )
            canvas_shape = labels2d.vertices.canvas_shape
            canvas_height = canvas_shape.height[i].shape.as_list()[-1]
            canvas_width = canvas_shape.width[i].shape.as_list()[-1]

            # Encode the paths from the image.
            target = self._encode_dense(
                priors_generator,
                image_shape.height,
                image_shape.width,
                canvas_height,
                canvas_width,
                i_polylines,
                i_vertices_per_polyline,
                i_class_ids_per_polyline,
                i_attrs_per_polyline,
            )
            return target

        path_labels = []
        for i in range(example_count):
            frames = []
            for j in range(max_timesteps_in_example):
                frames.append(_dense_wrapper(i, j))

            if len(frames) > 1:
                frames = tf.stack(frames, 0)
            else:
                frames = tf.expand_dims(frames[0], 0)
            path_labels.append(frames)

        if len(path_labels) > 1:
            path_labels = tf.stack(path_labels, 0)
        else:
            path_labels = tf.expand_dims(path_labels[0], 0)

        # Set shape to keep shape inference in-tact.
        path_label_shape = path_labels.get_shape().as_list()
        path_labels.set_shape(
            [
                example_count,
                max_timesteps_in_example,
                path_label_shape[-2],
                path_label_shape[-1],
            ]
        )
        return path_labels
