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

"""Utilities related to boxes/anchors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.image.python.ops import image_ops
from nvidia_tao_tf1.cv.ssd.utils.tensor_utils import tensor_strided_replace


def bipartite_match_row(similarity, scope=None):
    '''
    Returns a bipartite matching according to the similarity matrix.

    This is a greedy bi-partite matching algorithm.

    Arguments:
        similarity (tensor): A 2D tensor represents the similarity of each pair
            of boxes.

    Returns:
        matches (tensor): 1-D tensor with `matches.shape[0]==similarity.shape[0]`
            that represents which column box the corresponding row box should
            match to.
    '''
    with tf.name_scope(scope, 'BipartiteMatch'):
        matches, _ = image_ops.bipartite_match(
            -1.0 * similarity, num_valid_rows=tf.cast(tf.shape(similarity)[0], tf.float32))

        matches = tf.reshape(matches, [-1])
        matches = tf.cast(matches, tf.int32)

        return matches


def multi_match(similarity, threshold, scope=None):
    '''
    Returns argmax match according to similarity.

    Matches all elements along the second axis of `similarity` to their best
    matches along the first axis subject to the constraint that the weight of
    a match must be greater than or equal to `threshold` in order to produce a match.

    If the weight matrix contains elements that should be ignored, the row or column
    representing the respective elemet should be set to a value below `threshold`.

    Arguments:
        similarity (tensor): A 2D tensor represents the similarity of each pair
            of boxes.

        threshold (float): A float that represents the threshold (i.e. lower bound)
            that must be met by a pair of elements to produce a match.

    Returns:
        gt_idx, anchor_idx (tensor): Two 1D tensor of equal length that represent
        the matched indices. `gt_idx` contains the indices along the first axis of
         `similarity`, `anchor_idx` contains the indices along the second axis.
    '''
    with tf.name_scope(scope, 'MultiMatch'):
        gt_indices = tf.argmax(similarity, axis=0, output_type=tf.int32)
        num_col = tf.shape(similarity)[1]
        gt_indices_flatten = tf.range(0, num_col) + num_col * gt_indices
        overlaps = tf.gather(tf.reshape(similarity, [-1]), gt_indices_flatten)

        # Filter out the matches with a weight below the threshold.
        anchor_indices_thresh_met = tf.reshape(tf.where(overlaps >= threshold), [-1])
        gt_indices_thresh_met = tf.gather(gt_indices, anchor_indices_thresh_met)

        return tf.cast(gt_indices_thresh_met, tf.int32), \
            tf.cast(anchor_indices_thresh_met, tf.int32)


def area(boxlist, scope=None):
    """Computes area of boxes.

    Args:
        boxlist (tensor): Tensor of shape [N,4], holding xmin, ymin, xmax, ymax
        scope: name scope.

    Returns:
        a tensor with shape [N] representing box areas.
    """
    with tf.name_scope(scope, 'Area'):
        return (boxlist[:, 2] - boxlist[:, 0]) * (boxlist[:, 3] - boxlist[:, 1])


def intersection(boxlist1, boxlist2, scope=None):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxlist1 (tensor): Tensor of shape [N,4], holding xmin, ymin, xmax, ymax
        boxlist2 (tensor): Tensor of shape [M,4], holding xmin, ymin, xmax, ymax
        scope: name scope.

    Returns:
        a tensor with shape [N, M] representing pairwise intersections
    """
    with tf.name_scope(scope, 'Intersection'):
        x_min1, y_min1, x_max1, y_max1 = tf.split(
            value=boxlist1, num_or_size_splits=4, axis=1)
        x_min2, y_min2, x_max2, y_max2 = tf.split(
            value=boxlist2, num_or_size_splits=4, axis=1)
        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
        intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
        intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
        return intersect_heights * intersect_widths


def iou(boxlist1, boxlist2, scope=None):
    """Computes pairwise intersection-over-union between box collections.

    Args:
        boxlist1 (tensor): Tensor of shape [N,4], holding xmin, ymin, xmax, ymax
        boxlist2 (tensor): Tensor of shape [M,4], holding xmin, ymin, xmax, ymax
        scope: name scope.

    Returns:
        a tensor with shape [N, M] representing pairwise iou scores.
    """
    with tf.name_scope(scope, 'IOU'):
        intersections = intersection(boxlist1, boxlist2)
        areas1 = area(boxlist1)
        areas2 = area(boxlist2)
        unions = (
            tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
        return tf.where(
            tf.equal(intersections, 0.0),
            tf.zeros_like(intersections), tf.truediv(intersections, unions))


def np_elem_iou(boxes1, boxes2, border_pixels='half'):
    '''
    numpy version of element-wise iou.

    Computes the intersection-over-union similarity (also known as Jaccard similarity)
    of two sets of axis-aligned 2D rectangular boxes.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    In 'outer_product' mode, returns an `(m,n)` matrix with the IoUs for all possible
    combinations of the boxes in `boxes1` and `boxes2`.

    In 'element-wise' mode, `m` and `n` must be broadcast-compatible. Refer to the explanation
    of the `mode` argument for details.

    Arguments:
        boxes1 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates
            for one box in the format specified by `coords` or a 2D Numpy array of shape
            `(m, 4)` containing the coordinates for `m` boxes. If `mode` is set to 'element_wise',
            the shape must be broadcast-compatible with `boxes2`.
        boxes2 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for
            one box in the format specified by `coords` or a 2D Numpy array of shape `(n, 4)`
            containing the coordinates for `n` boxes. If `mode` is set to 'element_wise', the
            shape must be broadcast-compatible with `boxes1`.`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A 1D or 2D Numpy array (refer to the `mode` argument for details) of dtype float
        containing values in [0,1], the Jaccard similarity of the boxes in `boxes1` and
        `boxes2`. 0 means there is no overlap between two given boxes, 1 means their
        coordinates are identical.
    '''

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2:
        raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}."
                         .format(boxes1.ndim))
    if boxes2.ndim > 2:
        raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}."
                         .format(boxes2.ndim))

    if boxes1.ndim == 1:
        boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1:
        boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4):
        raise ValueError("Boxes list last dim should be 4 but got shape {} and {}, respectively."
                         .format(boxes1.shape, boxes2.shape))

    # Set the correct coordinate indices for the respective formats.
    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    # Compute the union areas.
    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    # Compute the IoU.

    min_xy = np.maximum(boxes1[:, [xmin, ymin]], boxes2[:, [xmin, ymin]])
    max_xy = np.minimum(boxes1[:, [xmax, ymax]], boxes2[:, [xmax, ymax]])

    # Compute the side lengths of the intersection rectangles.
    side_lengths = np.maximum(0, max_xy - min_xy + d)

    intersection_areas = side_lengths[:, 0] * side_lengths[:, 1]

    boxes1_areas = (boxes1[:, xmax] - boxes1[:, xmin] + d) * (boxes1[:, ymax] - boxes1[:, ymin] + d)
    boxes2_areas = (boxes2[:, xmax] - boxes2[:, xmin] + d) * (boxes2[:, ymax] - boxes2[:, ymin] + d)

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas


def corners_to_centroids(tensor, start_index, scope=None):
    '''
    Convert corner coordinates to centroids (tf tensor).

    Arguments:
        tensor (array): A tf nD tensor containing the four consecutive coordinates
            to be converted somewhere in the last axis.
        start_index (int): The index of the first coordinate in the last axis of `tensor`.
        scope (str, optional): The scope prefix of the name_scope.

    Returns:
        A tf nD tensor in centroids coordinates.
    '''

    with tf.name_scope(scope, 'CornerCoordToCentroids'):
        tt = tf.gather(tensor, tf.constant(start_index, dtype=tf.int32), axis=-1)
        bb = tf.gather(tensor, tf.constant(start_index+2, dtype=tf.int32), axis=-1)
        ll = tf.gather(tensor, tf.constant(start_index+1, dtype=tf.int32), axis=-1)
        rr = tf.gather(tensor, tf.constant(start_index+3, dtype=tf.int32), axis=-1)
        """
        original new_coords = [
            tf.truediv(tensor[..., start_index] + tensor[..., start_index+2], 2.0),
            tf.truediv(tensor[..., start_index+1] + tensor[..., start_index+3], 2.0),
            tensor[..., start_index+2] - tensor[..., start_index],
            tensor[..., start_index+3] - tensor[..., start_index+1]]
        """
        new_coords = [
            tf.truediv(tt + bb, 2.0),
            tf.truediv(ll + rr, 2.0),
            bb - tt,
            rr - ll
        ]
        return tensor_strided_replace(tensor, (start_index, start_index + 4), new_coords, -1)


def np_convert_coordinates(tensor, start_index, conversion):
    '''
    Convert coordinates for axis-aligned 2D boxes between two coordinate formats.

    Creates a copy of `tensor`, i.e. does not operate in place. Currently there are
    three supported coordinate formats that can be converted from and to each other:
        1) (xmin, xmax, ymin, ymax) - the 'minmax' format
        2) (xmin, ymin, xmax, ymax) - the 'corners' format
        2) (cx, cy, w, h) - the 'centroids' format

    Arguments:
        tensor (array): A Numpy nD array containing the four consecutive coordinates
            to be converted somewhere in the last axis.
        start_index (int): The index of the first coordinate in the last axis of `tensor`.
        conversion (str, optional): The conversion direction. Can be 'minmax2centroids',
            'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners',
            or 'corners2minmax'.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A Numpy nD array, a copy of the input tensor with the converted coordinates
        in place of the original coordinates and the unaltered elements of the original
        tensor elsewhere.
    '''
    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+1]) / 2.0  # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+2] + tensor[..., ind+3]) / 2.0  # Set cy
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind]  # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+2]  # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0  # Set xmin
        tensor1[..., ind+1] = tensor[..., ind] + tensor[..., ind+2] / 2.0  # Set xmax
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0  # Set ymin
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0  # Set ymax
    elif conversion == 'corners2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+2]) / 2.0  # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+1] + tensor[..., ind+3]) / 2.0  # Set cy
        tensor1[..., ind+2] = tensor[..., ind+2] - tensor[..., ind]  # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+1]  # Set h
    elif conversion == 'centroids2corners':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0  # Set xmin
        tensor1[..., ind+1] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0  # Set ymin
        tensor1[..., ind+2] = tensor[..., ind] + tensor[..., ind+2] / 2.0  # Set xmax
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0  # Set ymax
    elif conversion in ['minmax2corners', 'corners2minmax']:
        tensor1[..., ind+1] = tensor[..., ind+2]
        tensor1[..., ind+2] = tensor[..., ind+1]
    else:
        raise ValueError("Unexpected conversion value.")

    return tensor1
