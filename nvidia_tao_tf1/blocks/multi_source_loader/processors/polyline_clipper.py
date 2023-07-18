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

"""Class for clipping polylines representing priors or path labels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import py_func
from nvidia_tao_tf1.core.coreobject import TAOObject, save_args
from nvidia_tao_tf1.core.processors import ClipPolygon


class PolylineClipper(TAOObject):
    """Clip priors and labels to image boundaries."""

    @save_args
    def __init__(self, vertex_count_per_polyline=-1):
        """
        Initialize the clipper processor.

        Args:
            vertex_count_per_polyline (int): It is only used for the priors,
                                             not needed for path labels.
        """
        self._vertex_count_per_polyline = vertex_count_per_polyline
        self._clipper = ClipPolygon(closed=False)
        super(PolylineClipper, self).__init__()

    def _enforce_bottom_up_vertex_order(self, polylines):
        """
        Vertices should still be ordered from bottom to top of the image (always decreasing y).

        The clipper can cause this ordering to be reversed.

        This is only important for the priors as only the ground truths are reordered in
        the path generator op. Code below will only work when the number of vertices
        is equal and maintained in each path.

        Args:
            polylines (N x 2 tf.Tensor): Vertices of the polylines.

        Returns:
            reordered_polylines (N x 2 tf.Tensor): Vertices of the correctly ordered polylines.
        """
        polylines_reshaped = tf.reshape(
            tf.expand_dims(polylines, axis=-1), [-1, self._vertex_count_per_polyline, 2]
        )
        polylines_to_reverse = tf.less(
            polylines_reshaped[:, 0, 1],
            polylines_reshaped[:, self._vertex_count_per_polyline - 1, 1],
        )

        polylines_to_reverse = tf.tile(
            tf.expand_dims(polylines_to_reverse, axis=1),
            [1, self._vertex_count_per_polyline],
        )
        polylines_to_reverse = tf.tile(
            tf.expand_dims(polylines_to_reverse, axis=2), [1, 1, 2]
        )

        polylines_reordered = tf.compat.v1.where(
            polylines_to_reverse,
            tf.reverse(polylines_reshaped, [1]),
            polylines_reshaped,
        )
        return tf.reshape(polylines_reordered, tf.shape(input=polylines))

    def _resample_shortened_polylines(
        self,
        polylines,
        vertices_per_polyline,
        expected_vertices_per_polyline,
        polyline_index_map,
    ):
        """
        Check for polylines that changed number of vertices after clipping.

        The number of vertices in a polyline can change after clipping if the clip
        occurs between the the second and n-1st vertex. This is only a concern when
        clipping polylines that are intended to have specific vertex counts, like
        priors. Currently this function simply repeats the last vertex until the
        number of vertex requirement is met.

        Args:
            polylines (N x 2 np.Tensor): Vertices of the polylines.
            vertices_per_polyline (N x 1 np.Tensor): Number of vertices for each polyline.
            expected_vertices_per_polyline (N x 1 np.Tensor): Number of vertices for each polyline.
            polyline_index_map (N x 1 np.Tensor): Original id of polylines.

        Returns:
            resampled_polylines (M x 2 np.Tensor): Vertices of the resampled polylines.
            resampled_vertices(M x 1 np.Tensor): Number of vertices for each resampled polyline.
        """
        # TODO(blythe): Refactor to tensorflow functions.
        resampled_polylines = np.empty((0, 2), dtype=np.float32)
        resampled_vertices = np.empty((0,), dtype=np.int32)

        # Ensure that the number of expected vertices is always greater or equal
        # to the number of current vertices.
        assert not np.where(
            vertices_per_polyline > expected_vertices_per_polyline[polyline_index_map]
        )[
            0
        ], "Requesting to subtract vertices which is not a valid operation for this processor."

        # First just check that there is a polyline with a vertex number change.
        polyline_needs_resampling = np.where(
            vertices_per_polyline != expected_vertices_per_polyline[polyline_index_map]
        )[0]
        if polyline_needs_resampling.size != 0:

            # Count up the number of vertices before each polyline starts.
            vertex_counts = np.cumsum(np.hstack((0, vertices_per_polyline)))

            for path in range(len(vertices_per_polyline)):
                start_vertex = vertex_counts[path]
                end_vertex = start_vertex + vertices_per_polyline[path]

                # Repeat last vertex additional_vertices times to regain the right
                # number of vertices.
                # TODO(blythe): Use interpolation to resample vertices rather than just repetition.
                additional_vertices = (
                    expected_vertices_per_polyline[polyline_index_map[path]]
                    - vertices_per_polyline[path]
                )
                resampled_polyline = polylines[start_vertex:end_vertex, :]
                resampled_polyline = np.vstack(
                    (
                        resampled_polyline,
                        np.tile(polylines[end_vertex - 1, :], (additional_vertices, 1)),
                    )
                )

                resampled_polylines = np.vstack(
                    (resampled_polylines, resampled_polyline)
                )
                resampled_vertices = np.hstack(
                    (resampled_vertices, np.int32(resampled_polyline.shape[0]))
                )

        else:
            # If there are no splits, just pass the original input.
            resampled_polylines = polylines
            resampled_vertices = vertices_per_polyline

        return resampled_polylines, resampled_vertices

    def _remove_split_polylines(
        self, polylines, vertices_per_polyline, polyline_index_map
    ):
        """
        Check for split polylines and keep polyline segments closest to bottom of image.

        Split polylines can occur when a labeled polyline is clipped to within the image
        boundaries, but the original polyline has multiple segments that result from the
        clipping. This function choses only one segment from the original polyline to
        train on. This means that this function may receive N polylines, but only return
        M polylines, where M <= N.

        Args:
            polylines (N x 2 np.Tensor): Vertices of the polylines.
            vertices_per_polyline (N x 1 np.Tensor): Number of vertices for each polyline.
            polyline_index_map (N x 1 np.Tensor): Original id of polylines.

        Returns:
            unsplit_polylines (M x 2 np.Tensor): Vertices of the unsplit polylines.
            unsplit_vertices(M x 1 np.Tensor): Number of vertices for each unsplit polyline.
            unsplit_index_map (N x 1 np.Tensor): Original id of unsplit polylines.
        """
        # TODO(blythe): Refactor to tensorflow functions.
        unsplit_polylines = np.empty((0, 2), dtype=np.float32)
        unsplit_vertices = np.empty((0,), dtype=np.int32)
        unsplit_index_map = np.empty((0,), dtype=np.int32)

        # First just check that there is a split polyline.
        if np.unique(polyline_index_map).shape[0] < polyline_index_map.shape[0]:

            # Count up the number of vertices before each polyline starts.
            vertex_counts = np.cumsum(np.hstack((0, vertices_per_polyline)))

            # For each of the paths, determine whether there is a split or not.
            for path in np.unique(polyline_index_map):

                # Check if this particular polyline has a split or not.
                if len(np.where(polyline_index_map == path)[0]) > 1:

                    # For the split polylines, find and keep only the polyline segment closest
                    # to the bottom of the image. (y increases as you get closer to the
                    # bottom of the image)
                    max_y = -np.Inf
                    polylines_to_keep = []
                    vertices_to_keep = []
                    for segment_num in np.where(polyline_index_map == path)[0]:
                        start_vertex = vertex_counts[segment_num]
                        end_vertex = start_vertex + vertices_per_polyline[segment_num]
                        segment = polylines[start_vertex:end_vertex, :]

                        if np.max(segment[:, 1]) > max_y:
                            max_y = np.max(segment[:, 1])
                            polylines_to_keep = segment
                            vertices_to_keep = vertices_per_polyline[segment_num]
                            index_to_keep = polyline_index_map[segment_num]

                    unsplit_polylines = np.vstack(
                        (unsplit_polylines, polylines_to_keep)
                    )
                    unsplit_vertices = np.hstack((unsplit_vertices, vertices_to_keep))
                    unsplit_index_map = np.hstack((unsplit_index_map, index_to_keep))

                else:
                    # For the polylines without a split, just keep the polyline and vertex number.
                    segment_num = np.where(polyline_index_map == path)[0][0]
                    start_vertex = vertex_counts[segment_num]
                    end_vertex = start_vertex + vertices_per_polyline[segment_num]

                    unsplit_polylines = np.vstack(
                        (unsplit_polylines, polylines[start_vertex:end_vertex, :])
                    )
                    unsplit_vertices = np.hstack(
                        (unsplit_vertices, vertices_per_polyline[segment_num])
                    )
                    unsplit_index_map = np.hstack(
                        (unsplit_index_map, polyline_index_map[segment_num])
                    )

        else:
            # If there are no splits, just pass the original input.
            unsplit_polylines = polylines
            unsplit_vertices = vertices_per_polyline
            unsplit_index_map = polyline_index_map

        return unsplit_polylines, unsplit_vertices, unsplit_index_map

    def _gather_clipped_attributes(
        self, attributes, clipped_index, class_ids_per_polyline
    ):
        """
        Gather the attributes at the corresponding clipped_index.

        Args:
            attributes (tf.Tensor): 1D tensor of shape len(class_ids_per_polyline)
                                    * (npath_attributes + 1).
            clipped_index (tf.Tensor): 1D tensor with length of N holding the index of interest of
                                       the first attribute group.
            class_ids_per_polyline (tf.Tensor): 1D tensor, class ids for each polyline.

        Returns:
            attributes (tf.Tensor): Tensor of shape [1, len(clipped_index)* nattribute_groups].
        """

        def _clip_attributes(attributes, clipped_index, nattribute_groups):
            ncolumns = tf.reshape(tf.shape(input=clipped_index)[0], [1])
            base_index = tf.reshape(
                tf.tile(
                    tf.reshape(tf.range(nattribute_groups), [nattribute_groups, 1]),
                    [1, ncolumns[0]],
                ),
                [-1],
            )

            clipped_index = tf.tile(clipped_index, [nattribute_groups])
            index = tf.stack([base_index, clipped_index], axis=-1)
            reshaped_attributes = tf.reshape(attributes, [nattribute_groups, -1])
            clipped_attributes = tf.gather_nd(reshaped_attributes, index)

            return clipped_attributes

        nattribute_groups = tf.cond(
            pred=tf.equal(tf.shape(input=class_ids_per_polyline)[0], 0),
            true_fn=lambda: 1,
            false_fn=lambda: tf.compat.v1.div(
                tf.shape(input=attributes)[0], tf.shape(input=class_ids_per_polyline)[0]
            ),
        )

        clipped_attributes = tf.cond(
            pred=tf.equal(nattribute_groups, 1),
            true_fn=lambda: tf.gather(attributes, clipped_index),
            false_fn=lambda: _clip_attributes(
                attributes, clipped_index, nattribute_groups
            ),
        )

        return clipped_attributes

    def clip(
        self,
        polylines,
        vertices_per_polyline,
        class_ids_per_polyline,
        attributes_per_polyline,
        maintain_vertex_number,
        polygon_mask,
    ):  # noqa: D405 (pydocstring bug)
        """
        Clip the polylines to the model input boundaries.

        Args:
            polylines (tf.Tensor): Tensor of shape [N,2] and type tf.float32.
                                   Vertices of the polylines.
            vertices_per_polyline (tf.Tensor): Tensor of shape [L,1] and type tf.float32.
                                               Number of vertices for each polyline.
            class_ids_per_polyline (tf.Tensor): Tensor of shape [L,1] and type tf.float32.
                                                Class ids for each polyline.
            attributes_per_polyline (tf.Tensor): Tensor of shape [L,1] and type tf.float32.
                                                 Attributes for each polyline.
            maintain_vertex_number (bool): True if expecting same number of vertices out as in.
            polygon_mask (tf.Tensor): Tensor of shape [5, 2] and type tf.float32.
                                      Contains the 4 corners (first one repeated) of the cropping
                                      boundary for the polylines. Points start at 0, 0 and proceed
                                      counter-clockwise around the border ending at 0, 0 also.

        Returns:
            clipped_polylines (tf.Tensor): Tensor of shape [M,2] and type tf.float32.
                                           Vertices of the clipped polylines.
            clipped_polyline_index_map (tf.Tensor): Tensor of shape [M,1] and type tf.float32.
                                                    Index of surviving polylines into original
                                                    polyline tensor.
            clipped_vertices_per_polyline (tf.Tensor): Tensor of shape [M,1] and type tf.float32.
                                                       Number of vertices for each clipped polyline.
            clipped_class_ids_per_polyline (tf.Tensor): Tensor of shape [L,1] and type tf.float32.
                                                        Class ids for each clipped polyline.
            clipped_attributes_per_polyline (tf.Tensor): Tensor of shape [L,1] and type tf.float32.
                                                         Attributes for each clipped polyline.
        """
        (
            clipped_polylines,
            clipped_vertices_per_polyline,
            clipped_polyline_index_map,
        ) = self._clipper(
            polygons=polylines,
            points_per_polygon=vertices_per_polyline,
            polygon_mask=polygon_mask,
        )

        # Check for and handle the case where the number of polylines increases due to splitting.
        # Keep only the polyline segment closest to bottom of the image.
        clipped_polylines, clipped_vertices_per_polyline, clipped_polyline_index_map = py_func(
            self._remove_split_polylines,
            [
                clipped_polylines,
                clipped_vertices_per_polyline,
                clipped_polyline_index_map,
            ],
            [tf.float32, tf.int32, tf.int32],
        )

        # Check for and handle the case where the number vertices per polyline changes
        # due to splitting. Only modify the vertices if need to maintain the vertex
        # number as indicated by maintain_vertex_number input.
        if maintain_vertex_number:
            clipped_polylines, clipped_vertices_per_polyline = py_func(
                self._resample_shortened_polylines,
                [
                    clipped_polylines,
                    clipped_vertices_per_polyline,
                    vertices_per_polyline,
                    clipped_polyline_index_map,
                ],
                [tf.float32, tf.int32],
            )

            clipped_polylines = self._enforce_bottom_up_vertex_order(clipped_polylines)
        clipped_polylines.set_shape([clipped_polylines.shape[0], 2])

        # Check for and handle the case where the number of polylines decreases due to
        # being out of bounds. We just need to handle the meta data in this case.
        clipped_class_ids_per_polyline = tf.gather(
            class_ids_per_polyline, clipped_polyline_index_map
        )
        clipped_attributes_per_polyline = self._gather_clipped_attributes(
            attributes_per_polyline, clipped_polyline_index_map, class_ids_per_polyline
        )

        return (
            clipped_polylines,
            clipped_polyline_index_map,
            clipped_vertices_per_polyline,
            clipped_class_ids_per_polyline,
            clipped_attributes_per_polyline,
        )
