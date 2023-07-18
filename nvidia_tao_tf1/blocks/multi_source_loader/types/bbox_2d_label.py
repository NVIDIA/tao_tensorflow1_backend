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

"""Bbox label type.

This file also contains helper functions (some private, others public) for dealing with individual
label types typically associated with bounding box labels, such as (front, back) markers and depth.

It is possible in the future that these may become their own standalone types if such a need arises.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types.coordinates2d import (
    Coordinates2D,
)
from nvidia_tao_tf1.blocks.multi_source_loader.types.tensor_transforms import (
    map_and_stack,
)


def augment_marker_labels(marker_labels, stm):
    """Augment marker labels.

    Why is the below check enough? For DriveNet, all STMs start out as a 3x3 identity matrices M.
    In determining the final STM, input STMS are right multiplied sequentially with a
    flip LR STM, and a combination of translation/zoom STMs which use the same underlying
    representation. A quick matrix multiply will show you that applying both a translate and a
    zoom STM is pretty much the same as applying one such STM with different parameters.
    Furthermore, given that the parameters passed to get the translate and scale STMs are always
    positive, the end result R of multiplying the initial STM M by the flip LR STM x
    translate/zoom STM shows that R[0, 0] is positive if and only if no flip LR STM was applied.

    NOTE: If rotations are introduced, this reasoning is no longer sufficient.

    Args:
        marker_labels (tf.Tensor): Contains either front or back marker values.
        stm (tf.Tensor): 3x3 spatial transformation matrix with which to augment ``marker_labels``.

    Returns:
        augmented_marker_labels (tf.Tensor): Contains the marker values with the spatial
            transformations encapsulated by ``stm`` applied to them.
    """

    def no_flip():
        return marker_labels

    def flip():
        return tf.compat.v1.where(
            tf.equal(marker_labels, -1.0), marker_labels, 1.0 - marker_labels
        )

    with tf.control_dependencies(
        [
            tf.compat.v1.assert_equal(stm[0, 1], 0.0),
            tf.compat.v1.assert_equal(stm[1, 0], 0.0),
        ]
    ):
        augmentated_marker_labels = tf.cond(
            pred=stm[0, 0] < 0.0, true_fn=flip, false_fn=no_flip
        )

    return augmentated_marker_labels


def _get_begin_and_end_indices(sparse_tensor):
    """Helper function that returns the beginning and end indices per example.

    Args:
        sparse_tensor (tf.SparseTensor)

    Returns:
        begin_indices (tf.Tensor): i-th element indicates the index from which the i-th example's
            values in ``sparse_tensor`` starts.
        end_indices (tf.Tensor): i-th element indicates the last index pertaining to the i-th
            example's values in ``sparse_tensor``.
        indices_index (tf.Tensor): Range representation from 0 to the number of values in
            ``sparse_tensor``.
    """
    indices = tf.cast(sparse_tensor.indices, tf.int32)

    count_per_example = tf.math.bincount(indices[:, 0], dtype=tf.int64)
    example_count = tf.size(input=count_per_example)
    begin_indices = tf.cumsum(count_per_example, exclusive=True)
    end_indices = tf.cumsum(count_per_example)
    indices_index = tf.range(example_count)

    return begin_indices, end_indices, indices_index


def _transform_sparse(sparse_label, transformer, dtype=tf.float32):
    """Helper function to augment fields represented as tf.SparseTensor.

    Args:
        sparse_label (tf.SparseTensor): Field to transform.
        transformer (func): Signature is (index, values). ``index`` represents the outer-most index
            of the ``sparse_label`` to operate over, and ``values`` the values corresponding to
            this ``index``.
        dtype (tf.dtypes.Dtype).

    Returns:
        transformed_label (tf.SparseTensor): Augmented version of ``sparse_label``.
    """
    begin_indices, end_indices, indices_index = _get_begin_and_end_indices(sparse_label)

    def apply_spatial_transform(index):
        begin_index = begin_indices[index]
        end_index = end_indices[index]
        current_values = sparse_label.values[begin_index:end_index]

        return transformer(index, current_values)

    transformed_label = tf.SparseTensor(
        values=map_and_stack(apply_spatial_transform, indices_index, dtype=dtype),
        indices=sparse_label.indices,
        dense_shape=sparse_label.dense_shape,
    )

    return transformed_label


def _augment_sparse_marker_labels(markers, transform):
    """Helper function to use in conjunction with map_and_stack to augment markers.

    These markers are specifically expected to be a tf.SparseTensor with indices over
    [Example, Frame, Object, Value].


    Args:
        markers (tf.SparseTensor): Either front or back marker.
        transform (Transform): Transform to apply.

    Returns:
        transformed_markers (tf.SparseTensor): ``markers`` as transformed by ``transform``.
    """

    def transformer(index, current_values):
        spatial_transform_matrix = transform.spatial_transform_matrix[index, :]
        return augment_marker_labels(
            marker_labels=current_values, stm=spatial_transform_matrix
        )

    transformed_markers = _transform_sparse(markers, transformer)

    return transformed_markers


def _augment_depth(depth, transform):
    """Helper function to use in conjunction with map_and_stack to augment object depths.

    ``depth`` is specifically expected to be a tf.SparseTensor with indices over
    [Example, Frame, Object, Value].


    Args:
        depth (tf.SparseTensor): Contains depths of objects.
        transform (Transform): Transform to apply.

    Returns:
        transformed_depth (tf.SparseTensor): ``depth`` as transformed by ``transform``.
    """

    def transformer(index, current_values):
        spatial_transform_matrix = transform.spatial_transform_matrix[index, :]

        # Zoom factor is the square root of the determinant of the left-top 2x2 corner of
        # the spatial transformation matrix.
        abs_determinant = tf.abs(tf.linalg.det(spatial_transform_matrix[:2, :2]))

        scale_factor = tf.sqrt(abs_determinant)

        return scale_factor * current_values

    transformed_depth = _transform_sparse(depth, transformer)

    return transformed_depth


def _to_ltrb(coordinate_values):
    """Helper function to make sure coordinate values are series of [L, T, R, B].

    Args:
        coordinate_values (tf.Tensor): 1-D Tensor containing series of [L, T, R, B] or possibly
            [R, T, L, B] coordinates due to LR flipping.

    Returns:
        ltrb_values (tf.Tensor): Same shape as ``input`` but always in [L, T, R, B] order.
    """
    x1 = coordinate_values[::4]
    x2 = coordinate_values[2::4]
    y1 = coordinate_values[1::4]
    y2 = coordinate_values[3::4]
    xmin = tf.minimum(x1, x2)
    ymin = tf.minimum(y1, y2)
    xmax = tf.maximum(x1, x2)
    ymax = tf.maximum(y1, y2)

    ltrb_values = tf.reshape(tf.stack([xmin, ymin, xmax, ymax], axis=1), (-1,))

    return ltrb_values


# TODO(@williamz): Delete / move to types.Image2DReference or types.Session once the old
# DriveNet dataloader code is removed, and its consumers can adapt to the new dataloader more
# freely.
FRAME_FEATURES = ["frame_id"]

TARGET_FEATURES = [
    "vertices",  # Should this be named bbox_coords?
    "object_class",
    "occlusion",
    "truncation",
    "truncation_type",
    "is_cvip",
    "world_bbox_z",
    "non_facing",
    "front",
    "back",
]

ADDITIONAL_FEATURES = ["source_weight"]

_Bbox2DLabel = namedtuple(
    "Bbox2DLabel", FRAME_FEATURES + TARGET_FEATURES + ADDITIONAL_FEATURES
)


class Bbox2DLabel(_Bbox2DLabel):
    """Bbox label.

    frame_id (tf.Tensor): Frame id (str).
    vertices (Coordinates2D): Vertex coordinates for the bounding boxes. These follow the same
        definition as in Coordinates2D. However, instead of representing the 8 coordinate values
        of a bounding box explicitly, it will only contain series of [L, T, R, B] values. This is
        such that consumers of this label can reliably use those coordinates in that order.
    object_class (tf.SparseTensor): Class names associated with each bounding box.

    TODO(@williamz): Should we have the mapped values for this or preserve the original values?
    e.g. in a SQLite export from HumanLoop, these would be 'unknown', 'full', 'bottom', ..., but
    mapped to 0, 1, or 2.

    occlusion (tf.SparseTensor): Occlusion level of each bounding box. Is an int in {0, 1, 2}.
    truncation (tf.SparseTensor): Truncation level of each bounding box. Is a float in the range
        [0., 1.].
    truncation_type (tf.SparseTensor): An int (REALLY SHOULD BE A BOOLEAN TO BEGIN WITH??) that
        is 0 for not truncated, 1 for any form of truncation.
    is_cvip (tf.SparseTensor): Boolean tensor indicating whether a bounding box is the CVIP.
    world_bbox_z (tf.SparseTensor): Depth of an object.
    non_facing (tf.SparseTensor): Boolean tensor indicating whether an object (traffic light
        or road sign) is not facing us.
    front (tf.SparseTensor): Float tensor for where the front marker of an object is. Values are
        in [0., 1.] U {-1.}.
    back (tf.SparseTensor): Same as above, but for the rear marker of an object.
    """

    FRAME_FEATURES = FRAME_FEATURES
    TARGET_FEATURES = TARGET_FEATURES
    ADDITIONAL_FEATURES = ADDITIONAL_FEATURES

    def apply(self, transform, **kwargs):
        """
        Applies transformation to various bounding box level features.

        Args:
            transform (Transform): Transform to apply.

        Returns:
            (Bbox2DLabel): Transformed Bbox2DLabel.
        """
        transformed_coords = self.vertices.apply(transform)
        # To make sure downstream users can rely on the order being [L, T, R, B], we need to account
        # for possible LR-flip augmentations which would switch R with L.
        new_coords = _to_ltrb(transformed_coords.coordinates.values)

        fields_to_replace = dict()
        fields_to_replace["vertices"] = Coordinates2D(
            coordinates=tf.SparseTensor(
                values=new_coords,
                indices=transformed_coords.coordinates.indices,
                dense_shape=transformed_coords.coordinates.dense_shape,
            ),
            canvas_shape=transformed_coords.canvas_shape,
        )

        if isinstance(self.front, tf.SparseTensor):  # Could be an optional label.
            fields_to_replace["front"] = _augment_sparse_marker_labels(
                self.front, transform
            )
        if isinstance(self.back, tf.SparseTensor):  # Could be an optional label.
            fields_to_replace["back"] = _augment_sparse_marker_labels(
                self.back, transform
            )

        if isinstance(
            self.world_bbox_z, tf.SparseTensor
        ):  # Could be an optional label.
            fields_to_replace["world_bbox_z"] = _augment_depth(
                self.world_bbox_z, transform
            )

        return self._replace(**fields_to_replace)

    def _filter_vertices(self, valid_indices):
        """Helper function for returning filtered vertices.

        Args:
            valid_indices (tf.Tensor): 1-D boolean values with which to filter labels.

        Returns:
            (Coordinates2D): Filtered bbox vertices.
        """
        old_coords = self.vertices.coordinates

        old_values = tf.reshape(old_coords.values, [-1, 4])
        new_values = tf.boolean_mask(tensor=old_values, mask=valid_indices)
        new_values = tf.reshape(new_values, [-1])

        # 4 sparse indices per valid label index.
        valid_sparse_indices = tf.reshape(
            tf.stack([valid_indices for _ in range(4)], axis=1), [-1]
        )
        new_indices = tf.boolean_mask(
            tensor=old_coords.indices, mask=valid_sparse_indices
        )

        new_coords_sparse_tensor = tf.SparseTensor(
            values=new_values, indices=new_indices, dense_shape=old_coords.dense_shape
        )

        return self.vertices._replace(coordinates=new_coords_sparse_tensor)

    def filter(self, valid_indices):
        """
        Only keep those labels as indexed by ``valid_indices``.

        It is important to note that this filtering mechanism DOES NOT touch the dense_shape of the
        underlying tf.SparseTensor instances, as those may contain very important information. e.g.
        if the filtering happens to remove all objects from a particular frame, that should not
        bring the dense_shape's entry counting the total number of frames, as that may be used for
        things such as batch size, etc.

        Args:
            valid_indices (tf.Tensor): 1-D boolean values with which to filter labels.

        Returns:
            (Bbox2DLabel): Filtered Bbox2DLabel.
        """
        filtered_features = dict()

        for feature_name in TARGET_FEATURES:
            old_tensor = getattr(self, feature_name)
            if feature_name == "vertices":
                filtered_features[feature_name] = self._filter_vertices(valid_indices)
            elif isinstance(old_tensor, tf.SparseTensor):
                # Other features expected to be tf.SparseTensor.
                new_values = tf.boolean_mask(
                    tensor=old_tensor.values, mask=valid_indices
                )
                new_indices = tf.boolean_mask(
                    tensor=old_tensor.indices, mask=valid_indices
                )
                filtered_features[feature_name] = tf.SparseTensor(
                    values=new_values,
                    indices=new_indices,
                    dense_shape=old_tensor.dense_shape,
                )

        return self._replace(**filtered_features)


def filter_bbox_label_based_on_minimum_dims(bbox_2d_label, min_height, min_width):
    """Filter out entries in a label whose dimensions are less than some specified thresholds.

    Args:
        bbox_2d_label (Bbox2DLabel): Input label.
        min_height (float): Minimum height that an entry in ``box_2d_label`` must satisfy in order
            to be retained.
        min_width (float): Same but for width.

    Returns:
        filtered_label (Bbox2DLabel): Label with offending entries filtered out.
    """
    coords = _to_ltrb(bbox_2d_label.vertices.coordinates.values)
    width = coords[2::4] - coords[::4]
    height = coords[3::4] - coords[1::4]
    valid_indices = tf.logical_and(
        tf.greater_equal(width, min_width), tf.greater_equal(height, min_height)
    )

    return bbox_2d_label.filter(valid_indices=valid_indices)
