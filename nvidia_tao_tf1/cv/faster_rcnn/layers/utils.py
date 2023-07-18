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
"""utils for FasterRCNN custom keras layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def non_max_suppression_fast(boxes_in, probs, overlap_thresh=0.9, max_boxes=300, scale=1.):
    """non-maximum-suppression in Python."""
    # if there are no boxes, return an empty list
    if len(boxes_in) == 0:
        return boxes_in, probs, np.zeros((0,)).astype(np.int64)
    # grab the coordinates of the bounding boxes
    boxes = boxes_in * scale
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # initialize the list of picked indexes
    pick = []
    # calculate the areas
    area = (x2 - x1 + 1.) * (y2 - y1 + 1.)
    # sort the bounding boxes
    idxs = np.argsort(probs, kind='stable')
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the intersection
        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])
        ww_int = np.maximum(0, xx2_int - xx1_int + 1.)
        hh_int = np.maximum(0, yy2_int - yy1_int + 1.)
        area_int = ww_int * hh_int
        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int
        # compute the ratio of overlap
        overlap = area_int / area_union
        # delete all indexes from the index list that have
        idxs = idxs[0:last]
        idxs = idxs[np.where(overlap <= overlap_thresh)[0]]
        if len(pick) >= max_boxes:
            break
    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes_in[np.array(pick)]
    probs = probs[np.array(pick)]
    return boxes, probs, np.array(pick)


def unique_with_inverse(x):
    '''get unique elements from an array and also return the original index.'''
    y, idx = tf.unique(x)
    num_segments = tf.shape(y)[0]
    num_elems = tf.shape(x)[0]
    return (y, idx,  tf.unsorted_segment_min(tf.range(num_elems), idx, num_segments))


def safe_gather(tensor, indices, axis=0):
    '''Add an assert to tf.gather to make sure there is no out-of-bound indexing.'''
    length = tf.shape(tensor)[axis]
    min_idx = tf.cond(tf.logical_and(tf.size(indices) > 0, length > 0),
                      true_fn=lambda: tf.cast(tf.reduce_min(indices), length.dtype),
                      false_fn=lambda: tf.cast(0, length.dtype))
    max_idx = tf.cond(tf.logical_and(tf.size(indices) > 0, length > 0),
                      true_fn=lambda: tf.cast(tf.reduce_max(indices), length.dtype),
                      false_fn=lambda: tf.cast(-1, length.dtype))
    op_1 = tf.debugging.assert_less(max_idx, length)
    op_2 = tf.debugging.assert_less_equal(0, min_idx)
    # if tensor is non-empty, check the out-of-bound of indices
    # and do gather.
    with tf.control_dependencies([op_1, op_2]):
        out_non_empty = tf.gather(tensor, indices, axis=axis)
    # if tensor is empty, then we should output an empty tensor
    # instead of returing the out-of-bound default values(0)
    # create empty tensor with the same rank
    empty_tensor = tf.zeros_like(out_non_empty)[..., :0]
    out = tf.cond(length > 0,
                  true_fn=lambda: out_non_empty,
                  false_fn=lambda: empty_tensor)
    return out


def stable_top_k(values, k, axis=-1):
    '''Stable descending sort of a tensor and then take top k indices.'''
    sort_idx = tf.argsort(values, axis=axis, direction='DESCENDING', stable=True)
    # the slice[:k] will retain the first k or the actual size if the actual size
    # is less than k
    return sort_idx[:k]


def normalize_bbox_tf(boxes, h, w):
    '''Normalize the bbox coordinates to the range of (0, 1).'''
    scale = tf.stack([h-1.0, w-1.0, h-1.0, w-1.0])
    return boxes / scale


def apply_deltas_to_anchors(ancs, deltas):
    '''Apply deltas to anchors in RPN.'''
    x = ancs[:, 0]
    y = ancs[:, 1]
    w = ancs[:, 2]
    h = ancs[:, 3]
    tx = deltas[:, 0]
    ty = deltas[:, 1]
    tw = deltas[:, 2]
    th = deltas[:, 3]
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    cx1 = tx * w + cx
    cy1 = ty * h + cy
    w1 = tf.exp(tw) * w
    h1 = tf.exp(th) * h
    x1 = cx1 - 0.5 * w1
    y1 = cy1 - 0.5 * h1
    return tf.stack((y1, x1, y1 + h1, x1 + w1), axis=-1)


def clip_bbox_tf(boxes, width, height):
    '''Clip Bboxes to the the boundary of the input images.'''
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    x1_clip = tf.minimum(tf.maximum(x1, 0.0), width-1.0)
    y1_clip = tf.minimum(tf.maximum(y1, 0.0), height-1.0)
    x2_clip = tf.minimum(tf.maximum(x2, 0.0), width-1.0)
    y2_clip = tf.minimum(tf.maximum(y2, 0.0), height-1.0)
    clipped_boxes = tf.concat((y1_clip, x1_clip, y2_clip, x2_clip), axis=1)
    clipped_boxes.set_shape((clipped_boxes.shape[0], 4))
    return clipped_boxes


def batch_op(inputs, graph_fn, batch_size, names=None):
    '''Batch processing of an Op with given batch size.'''
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [tf.gather(x, tf.constant(i, dtype=tf.int32), axis=0) for x in inputs]
        output_slice = graph_fn(inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def nms_core_py_func(boxes, probs, max_boxes, iou_threshold):
    '''NMS core in numpy and wrapped as TF Op with tf.py_func.'''

    def _nms_wrapper(_boxes, _probs, _max_boxes, _iou_threshold):
        return non_max_suppression_fast(_boxes,
                                        _probs,
                                        overlap_thresh=_iou_threshold,
                                        max_boxes=_max_boxes)[2]
    out = tf.py_func(_nms_wrapper, [boxes, probs, max_boxes, iou_threshold],
                     tf.int64)
    return out


def nms_tf(all_boxes, all_probs, pre_nms_top_N,
           post_nms_top_N, nms_iou_thres):
    '''NMS in TF Op.'''
    val_idx = tf.where(tf.logical_and(all_boxes[:, 2] - all_boxes[:, 0] > 1.0,
                                      all_boxes[:, 3] - all_boxes[:, 1] > 1.0))[:, 0]
    valid_boxes = safe_gather(all_boxes, val_idx)
    valid_probs = safe_gather(all_probs, val_idx)
    idx = stable_top_k(valid_probs, pre_nms_top_N)
    valid_boxes = safe_gather(valid_boxes, idx)
    valid_probs = safe_gather(valid_probs, idx)
    # In the rare case of no valid_boxes at all, tf.image.non_max_suppression will
    # raise error due to limitation of the TF API.
    # So use a conditional to make sure it is not empty.
    valid_boxes_non_empty = tf.cond(tf.size(valid_boxes) > 0,
                                    true_fn=lambda : valid_boxes,
                                    false_fn=lambda: tf.constant([[0, 0, 0, 0]], dtype=tf.float32))
    valid_probs_non_empty = tf.cond(tf.size(valid_boxes) > 0,
                                    true_fn=lambda: valid_probs,
                                    false_fn=lambda: tf.constant([0], dtype=tf.float32))
    sel_idx = tf.image.non_max_suppression(valid_boxes_non_empty,
                                           valid_probs_non_empty,
                                           post_nms_top_N,
                                           iou_threshold=nms_iou_thres)
    result = safe_gather(valid_boxes_non_empty, sel_idx)
    zero_pads = tf.zeros(
        (post_nms_top_N - tf.shape(sel_idx)[0], tf.shape(result)[1]),
        dtype=result.dtype
    )
    result = tf.concat((result, zero_pads), axis=0)
    return result


def make_anchors(sizes, ratios):
    """Generate base anchors with different size and aspect ratios."""
    sa = np.array(sizes).astype(np.float32)
    ra = np.array(ratios).astype(np.float32)
    sa = np.repeat(sa, ra.size*2).reshape(-1, ra.size, 2)
    ra = np.stack((ra, 1./ra), axis=-1)
    ancs = sa*ra
    return ancs


def iou_tf(boxes1, boxes2):
    '''Calculate IoU.'''
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1 + 1.0, 0.) * tf.maximum(y2 - y1 + 1.0, 0.)
    # 3. Compute unions
    b1_area = tf.maximum(b1_y2 - b1_y1 + 1.0, 0.) * tf.maximum(b1_x2 - b1_x1 + 1.0, 0.)
    b2_area = tf.maximum(b2_y2 - b2_y1 + 1.0, 0.) * tf.maximum(b2_x2 - b2_x1 + 1.0, 0.)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def calculate_delta_tf(box, gt_box):
    '''Calculate deltas of proposals w.r.t. gt boxes.'''
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0] + 1.0
    width = box[:, 3] - box[:, 1] + 1.0
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0] + 1.0
    gt_width = gt_box[:, 3] - gt_box[:, 1] + 1.0
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dx, dy, dw, dh], axis=1)
    return result


def unpad_tf(boxes):
    '''Remove paddings from the boxes.'''
    nonzeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=-1), tf.bool)
    nonzero_boxes = tf.boolean_mask(boxes, nonzeros)
    return nonzero_boxes, nonzeros


def generate_proposal_target(rois, gt_boxes, gt_class_ids,
                             iou_high_thres, iou_low_thres,
                             roi_train_bs, roi_positive_ratio,
                             deltas_scaling, bg_class_id):
    '''Generate target tensors for RCNN.'''
    # unpad
    rois_unpad, _ = unpad_tf(rois)
    gt_boxes_unpad, nonzero_gt = unpad_tf(gt_boxes)
    gt_class_ids = tf.boolean_mask(gt_class_ids, nonzero_gt)
    # compute IoU
    ious = iou_tf(rois_unpad, gt_boxes_unpad)
    iou_max_over_gt = tf.reduce_max(ious, axis=1)
    positive_roi_bool = (iou_max_over_gt >= iou_high_thres)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    negative_indices = tf.where(tf.logical_and(iou_max_over_gt < iou_high_thres,
                                               iou_max_over_gt >= iou_low_thres))[:, 0]
    positive_count = int(roi_train_bs * roi_positive_ratio)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]

    negative_count = roi_train_bs - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    negative_count = tf.shape(negative_indices)[0]

    positive_rois = safe_gather(rois_unpad, positive_indices)
    negative_rois = safe_gather(rois_unpad, negative_indices)

    positive_ious = safe_gather(ious, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_ious)[1], 0),
        true_fn=lambda: tf.argmax(positive_ious, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64))
    roi_gt_boxes = safe_gather(gt_boxes_unpad, roi_gt_box_assignment)
    roi_gt_class_ids = safe_gather(gt_class_ids, roi_gt_box_assignment)
    deltas = calculate_delta_tf(positive_rois, roi_gt_boxes)
    deltas = deltas * deltas_scaling.reshape((1, 4))
    pn_rois = tf.concat((positive_rois, negative_rois), axis=0)
    num_paddings = tf.maximum(0, roi_train_bs - tf.shape(pn_rois)[0])
    pn_rois = tf.pad(pn_rois, [(0, num_paddings), (0, 0)])

    # pad bg class id for negative ROIs, shape: (R,)
    total_roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, negative_count)],
                                    constant_values=bg_class_id)
    # convert class IDs to one hot format, shape: (R, C+1)
    total_roi_gt_class_ids = tf.one_hot(tf.cast(total_roi_gt_class_ids, tf.int32),
                                        int(bg_class_id+1), axis=-1)
    # zero-padding for class IDs, shape: (total, C+1)
    total_roi_gt_class_ids = tf.pad(total_roi_gt_class_ids, [(0, num_paddings), (0, 0)])
    # construct one-hot deltas to mask out negative and padded ROIs.
    # (R, C, 1)
    roi_gt_class_ids_oh = tf.one_hot(tf.cast(roi_gt_class_ids, tf.int32),
                                     int(bg_class_id))
    roi_gt_class_ids_expand = tf.expand_dims(roi_gt_class_ids_oh, axis=-1)
    # (R, 1, 4)
    deltas_expanded = tf.expand_dims(deltas, axis=-2)
    # (R, C, 4)
    deltas_masked = roi_gt_class_ids_expand * deltas_expanded
    deltas_boolean_mask = tf.concat((roi_gt_class_ids_expand,
                                     roi_gt_class_ids_expand,
                                     roi_gt_class_ids_expand,
                                     roi_gt_class_ids_expand),
                                    axis=-1)
    # (R, C, 8)
    deltas_masked = tf.concat((deltas_boolean_mask, deltas_masked), axis=-1)
    # padding negative and empty deltas. -> (total, C, 8)
    deltas_masked = tf.pad(deltas_masked,
                           [(0, negative_count + num_paddings), (0, 0), (0, 0)])
    # flatten deltas_masked: (R, C8)
    deltas_masked = tf.reshape(deltas_masked,
                               [tf.shape(deltas_masked)[0],
                                tf.shape(deltas_masked)[1]*tf.shape(deltas_masked)[2]])
    return [pn_rois,
            total_roi_gt_class_ids,
            deltas_masked,
            positive_count]


def generate_proposal_target_v1(rois, gt_boxes, gt_class_ids,
                                iou_high_thres, iou_low_thres,
                                roi_train_bs, roi_positive_ratio,
                                deltas_scaling, bg_class_id):
    '''Generate target tensors for RCNN.'''
    # unpad
    rois_unpad, _ = unpad_tf(rois)
    gt_boxes_unpad, nonzero_gt = unpad_tf(gt_boxes)
    gt_class_ids = tf.boolean_mask(gt_class_ids, nonzero_gt)
    # compute IoU
    # in the rare case of empty RoIs after unpadding, iou_tf will raise error
    # So fix this by assigning zero boxes and set ious to be all -1 matrix in this case
    # -1 implies no postive or negative RoIs at all.
    rois_unpad_nz = tf.cond(tf.size(rois_unpad) > 0,
                            true_fn=lambda: rois_unpad,
                            false_fn=lambda: tf.constant([[0., 0., 0., 0.]], dtype=tf.float32))
    _ious = iou_tf(rois_unpad_nz, gt_boxes_unpad)
    ious = tf.cond(tf.size(rois_unpad) > 0,
                   true_fn=lambda: _ious,
                   false_fn=lambda: -1.0*tf.ones([1, tf.shape(gt_boxes_unpad)[0]],
                                                 dtype=tf.float32))
    iou_max_over_gt = tf.reduce_max(ious, axis=1)
    positive_roi_bool = (iou_max_over_gt >= iou_high_thres)
    positive_indices = tf.cond(tf.size(tf.where(positive_roi_bool)) > 0,
                               true_fn=lambda: tf.where(positive_roi_bool)[:, 0],
                               false_fn=lambda: tf.constant([], dtype=tf.int64))
    negative_roi_bool = tf.logical_and(iou_max_over_gt < iou_high_thres,
                                       iou_max_over_gt >= iou_low_thres)
    negative_indices = tf.cond(tf.size(tf.where(negative_roi_bool)) > 0,
                               true_fn=lambda: tf.where(negative_roi_bool)[:, 0],
                               false_fn=lambda: tf.constant([], dtype=tf.int64))
    positive_limit = int(roi_train_bs * roi_positive_ratio)
    has_pos = tf.size(positive_indices) > 0
    has_neg = tf.size(negative_indices) > 0

    # case 1: both positive RoI and negative RoI are not empty
    pos_idx_limit = tf.random_shuffle(positive_indices)[:positive_limit]
    neg_limit_pn = tf.constant(roi_train_bs, dtype=tf.int32) - tf.shape(pos_idx_limit)[0]
    neg_idx_limit_pn = tf.random_shuffle(negative_indices)[:neg_limit_pn]
    maxval_n = tf.cond(has_neg,
                       true_fn=lambda: tf.shape(negative_indices)[0],
                       false_fn=lambda: tf.constant(1, dtype=tf.int32))
    neg_idx_unform_sampler_pn = tf.random.uniform([neg_limit_pn],
                                                  maxval=maxval_n,
                                                  dtype=tf.int32,
                                                  seed=42)
    neg_idx_replace_pn = safe_gather(negative_indices, neg_idx_unform_sampler_pn)
    neg_idx_pn = tf.cond(tf.shape(negative_indices)[0] >= neg_limit_pn,
                         true_fn=lambda: neg_idx_limit_pn,
                         false_fn=lambda: neg_idx_replace_pn)

    # case 2: only positive RoIs.
    maxval_p = tf.cond(has_pos,
                       true_fn=lambda: tf.shape(positive_indices)[0],
                       false_fn=lambda: tf.constant(1, dtype=tf.int32))
    uniform_sampler_p = tf.random.uniform(tf.constant([roi_train_bs]),
                                          maxval=maxval_p,
                                          dtype=tf.int32,
                                          seed=42)
    pos_idx_batch = tf.cond(tf.size(positive_indices) >= roi_train_bs,
                            true_fn=lambda: tf.random_shuffle(positive_indices)[:roi_train_bs],
                            false_fn=lambda: safe_gather(positive_indices, uniform_sampler_p))

    # case 3: only negative RoIs.
    uniform_sampler_n = tf.random.uniform(tf.constant([roi_train_bs]),
                                          maxval=maxval_n,
                                          dtype=tf.int32,
                                          seed=42)
    neg_idx_batch = tf.cond(tf.size(negative_indices) >= roi_train_bs,
                            true_fn=lambda: tf.random_shuffle(negative_indices)[:roi_train_bs],
                            false_fn=lambda: safe_gather(negative_indices, uniform_sampler_n))

    # case 4: both positive and negative RoIs are empty. leave it empty.
    # Finally, combine the 4 cases.
    positive_idxs_case_1_2 = tf.cond(tf.logical_and(has_pos, has_neg),
                                     true_fn=lambda: pos_idx_limit,
                                     false_fn=lambda: pos_idx_batch)
    positive_idxs_case_3_4 = tf.constant([], dtype=tf.int64)
    positive_idx_all = tf.cond(has_pos,
                               true_fn=lambda: positive_idxs_case_1_2,
                               false_fn=lambda: positive_idxs_case_3_4)
    negative_idxs_case_1_2 = tf.cond(tf.logical_and(has_pos, has_neg),
                                     true_fn=lambda: neg_idx_pn,
                                     false_fn=lambda: tf.constant([], dtype=tf.int64))
    negative_idxs_case_3_4 = tf.cond(has_neg,
                                     true_fn=lambda: neg_idx_batch,
                                     false_fn=lambda: tf.constant([], dtype=tf.int64))
    negative_idx_all = tf.cond(has_pos,
                               true_fn=lambda: negative_idxs_case_1_2,
                               false_fn=lambda: negative_idxs_case_3_4)

    positive_count = tf.shape(positive_idx_all)[0]
    negative_count = tf.shape(negative_idx_all)[0]
    positive_rois = safe_gather(rois_unpad, positive_idx_all)
    negative_rois = safe_gather(rois_unpad, negative_idx_all)
    positive_ious = safe_gather(ious, positive_idx_all)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_ious)[1], 0),
        true_fn=lambda: tf.argmax(positive_ious, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64))
    roi_gt_boxes = safe_gather(gt_boxes_unpad, roi_gt_box_assignment)
    roi_gt_class_ids = safe_gather(gt_class_ids, roi_gt_box_assignment)
    _deltas = tf.cond(tf.logical_and(tf.size(positive_rois) > 0, tf.size(roi_gt_boxes) > 0),
                      true_fn=lambda: calculate_delta_tf(positive_rois, roi_gt_boxes),
                      false_fn=lambda: tf.constant([], dtype=tf.float32))
    deltas = tf.cond(tf.size(_deltas) > 0,
                     true_fn=lambda: _deltas * deltas_scaling.reshape((1, 4)),
                     false_fn=lambda: tf.zeros([1, 4], dtype=tf.float32))
    pn_rois = tf.concat((positive_rois, negative_rois), axis=0)
    num_paddings = tf.maximum(0, roi_train_bs - tf.shape(pn_rois)[0])
    pn_rois = tf.cond(tf.size(pn_rois) > 0,
                      true_fn=lambda: tf.pad(pn_rois, [(0, num_paddings), (0, 0)]),
                      false_fn=lambda: tf.zeros([num_paddings, 4], dtype=tf.float32))
    # pad bg class id for negative ROIs, shape: (R,)
    total_roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, negative_count)],
                                    constant_values=bg_class_id)
    # convert class IDs to one hot format, shape: (R, C+1)
    total_roi_gt_class_ids = tf.one_hot(tf.cast(total_roi_gt_class_ids, tf.int32),
                                        int(bg_class_id+1), axis=-1)
    # zero-padding for class IDs, shape: (total, C+1)
    total_roi_gt_class_ids = tf.pad(total_roi_gt_class_ids, [(0, num_paddings), (0, 0)])
    # construct one-hot deltas to mask out negative and padded ROIs.
    # (R, C, 1)
    roi_gt_class_ids_oh = tf.cond(tf.size(roi_gt_class_ids) > 0,
                                  true_fn=lambda: tf.one_hot(tf.cast(roi_gt_class_ids, tf.int32),
                                                             int(bg_class_id)),
                                  false_fn=lambda: tf.zeros([1, int(bg_class_id)],
                                                            dtype=tf.float32))
    roi_gt_class_ids_expand = tf.expand_dims(roi_gt_class_ids_oh, axis=-1)
    # (R, 1, 4)
    deltas_expanded = tf.expand_dims(deltas, axis=-2)
    # (R, C, 4)
    deltas_masked = roi_gt_class_ids_expand * deltas_expanded
    deltas_boolean_mask = tf.concat((roi_gt_class_ids_expand,
                                     roi_gt_class_ids_expand,
                                     roi_gt_class_ids_expand,
                                     roi_gt_class_ids_expand),
                                    axis=-1)
    # (R, C, 8)
    deltas_masked = tf.concat((deltas_boolean_mask, deltas_masked), axis=-1)
    # padding negative and empty deltas. -> (total, C, 8)
    deltas_masked = tf.pad(deltas_masked,
                           [(0, roi_train_bs-tf.shape(deltas_masked)[0]), (0, 0), (0, 0)])
    # flatten deltas_masked: (R, C8)
    deltas_masked = tf.reshape(deltas_masked,
                               [tf.shape(deltas_masked)[0],
                                tf.shape(deltas_masked)[1]*tf.shape(deltas_masked)[2]])
    return [pn_rois,
            total_roi_gt_class_ids,
            deltas_masked,
            positive_count]


def proposal_target_wrapper(rois, gt_class_ids, gt_boxes, iou_high_thres,
                            iou_low_thres, roi_train_bs, roi_positive_ratio,
                            deltas_scaling, bg_class_id, bs_per_gpu,
                            gt_as_roi=False):
    '''proposal target wrapper function.'''
    if gt_as_roi:
        rois = tf.concat((rois, gt_boxes), axis=1)
    # remove zero padding in ROIs and GT boxes.
    result = batch_op([rois, gt_boxes, gt_class_ids],
                      lambda x: generate_proposal_target_v1(
                                    *x,
                                    iou_high_thres=iou_high_thres,
                                    iou_low_thres=iou_low_thres,
                                    roi_train_bs=roi_train_bs,
                                    roi_positive_ratio=roi_positive_ratio,
                                    deltas_scaling=deltas_scaling,
                                    bg_class_id=int(bg_class_id)),
                      bs_per_gpu)
    return result


def intersection_np(boxes_a1, boxes_b1):
    """Intersection of two sets of boxes."""
    boxes_x1 = np.maximum(boxes_a1[:, :, 0], boxes_b1[:, :, 0])
    boxes_y1 = np.maximum(boxes_a1[:, :, 1], boxes_b1[:, :, 1])
    boxes_x2 = np.minimum(boxes_a1[:, :, 2], boxes_b1[:, :, 2])
    boxes_y2 = np.minimum(boxes_a1[:, :, 3], boxes_b1[:, :, 3])
    area = np.maximum(boxes_x2 - boxes_x1 + 1., 0.) * np.maximum(boxes_y2 - boxes_y1 + 1., 0.)
    return area


def union_np(boxes_a1, boxes_b1, area_inter):
    """Union of two sets of boxes."""
    area_a = np.maximum(boxes_a1[:, :, 2] - boxes_a1[:, :, 0] + 1., 0.) * \
        np.maximum(boxes_a1[:, :, 3] - boxes_a1[:, :, 1] + 1., 0.)
    area_b = np.maximum(boxes_b1[:, :, 2] - boxes_b1[:, :, 0] + 1., 0.) * \
        np.maximum(boxes_b1[:, :, 3] - boxes_b1[:, :, 1] + 1., 0.)
    return area_a + area_b - area_inter


def iou_np(boxes_a, boxes_b, scale=1.0):
    """IoU of two sets of boxes."""
    m = boxes_a.shape[0]
    n = boxes_b.shape[0]
    boxes_a1 = np.broadcast_to(boxes_a.reshape(m, 1, 4), (m, n, 4))
    boxes_b1 = np.broadcast_to(boxes_b.reshape(1, n, 4), (m, n, 4))
    boxes_a1 = boxes_a1 * scale
    boxes_b1 = boxes_b1 * scale
    area_i = intersection_np(boxes_a1, boxes_b1)
    area_u = union_np(boxes_a1, boxes_b1, area_i)
    result = area_i / area_u
    return result


def encode_anchor(_anc_boxes, _gt_boxes, scale=1.):
    """Encode ground truth boxes with offset relative to anchor boxes."""
    anc_boxes = _anc_boxes * scale
    gt_boxes = _gt_boxes * scale
    cx = (gt_boxes[:, 0] + gt_boxes[:, 2] + 1.) / 2.0
    cy = (gt_boxes[:, 1] + gt_boxes[:, 3] + 1.) / 2.0
    cxa = (anc_boxes[:, 0] + anc_boxes[:, 2] + 1.) / 2.0
    cya = (anc_boxes[:, 1] + anc_boxes[:, 3] + 1.) / 2.0
    tx = (cx - cxa) / (anc_boxes[:, 2] - anc_boxes[:, 0] + 1.)
    ty = (cy - cya) / (anc_boxes[:, 3] - anc_boxes[:, 1] + 1.)
    tw = np.log((gt_boxes[:, 2] - gt_boxes[:, 0] + 1.) /
                (anc_boxes[:, 2] - anc_boxes[:, 0] + 1.))
    th = np.log((gt_boxes[:, 3] - gt_boxes[:, 1] + 1.) /
                (anc_boxes[:, 3] - anc_boxes[:, 1] + 1.))
    return np.stack((tx, ty, tw, th), axis=-1)


def unpad_np(boxes):
    '''Unpad in numpy.'''
    nonzeros = np.sum(np.absolute(boxes), axis=-1).astype(np.bool_)
    nonzero_boxes = boxes[nonzeros, :]
    return np.ascontiguousarray(np.copy(nonzero_boxes)), nonzeros


def decode_anchor_np(X, T):
    """apply deltas to anchors, numpy version."""
    # anchors
    x = X[:, 0]
    y = X[:, 1]
    w = X[:, 2]
    h = X[:, 3]
    # deltas
    tx = T[:, 0]
    ty = T[:, 1]
    tw = T[:, 2]
    th = T[:, 3]
    cx = x + w/2.
    cy = y + h/2.
    cx1 = tx * w + cx
    cy1 = ty * h + cy
    w1 = np.exp(tw) * w
    h1 = np.exp(th) * h
    x1 = cx1 - w1/2.
    y1 = cy1 - h1/2.
    return np.stack((x1, y1, w1, h1), axis=-1)


def _unique_no_sort(ar):
    '''numpy unique but without sorting.'''
    idx = np.unique(ar, return_index=True)[1]
    return np.array([ar[i] for i in sorted(idx)]), np.array(sorted(idx))


def anchor_target_process(num_anchors_for_bbox,
                          best_anchor_for_bbox,
                          y_is_box_valid,
                          y_rpn_overlap,
                          y_rpn_regr,
                          best_dx_for_bbox,
                          n_anchratios,
                          rpn_mini_batch):
    """post processing of anchor generation."""
    # we ensure that every bbox has at least one positive RPN region
    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            # no box with an IOU greater than zero ...
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            y_is_box_valid[best_anchor_for_bbox[idx, 0],
                           best_anchor_for_bbox[idx, 1],
                           best_anchor_for_bbox[idx, 2] + n_anchratios *
                           best_anchor_for_bbox[idx, 3]] = 1
            y_rpn_overlap[best_anchor_for_bbox[idx, 0],
                          best_anchor_for_bbox[idx, 1],
                          best_anchor_for_bbox[idx, 2] + n_anchratios *
                          best_anchor_for_bbox[idx, 3]] = 1
            start = 4 * (best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3])
            y_rpn_regr[best_anchor_for_bbox[idx, 0],
                       best_anchor_for_bbox[idx, 1],
                       start:start+4] = best_dx_for_bbox[idx, :]
    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap_save = np.copy(y_rpn_overlap)
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)
    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid_save = np.copy(y_is_box_valid)
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)
    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)
    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1,
                                       y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0,
                                       y_is_box_valid[0, :, :, :] == 1))
    num_pos = len(pos_locs[0])
    # one issue is that the RPN has many more negative than positive regions,
    # so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    num_regions = rpn_mini_batch
    if len(pos_locs[0]) > num_regions//2:
        val_locs = np.random.choice(np.arange(len(pos_locs[0])), len(pos_locs[0]) -
                                    num_regions//2, replace=False)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs],
                       pos_locs[2][val_locs]] = 0
        # also mask out the conf for pos
        y_rpn_overlap[0, pos_locs[0][val_locs],
                      pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions//2
    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = np.random.choice(np.arange(len(neg_locs[0])), len(neg_locs[0]) -
                                    (num_regions - num_pos), replace=False).tolist()
        y_is_box_valid[0, neg_locs[0][val_locs],
                       neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0
    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1,
                                       y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0,
                                       y_is_box_valid[0, :, :, :] == 1))
    num_pos = len(pos_locs[0])
    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)
    return np.copy(y_rpn_cls), np.copy(y_rpn_regr), y_is_box_valid_save, y_rpn_overlap_save


def _compute_rpn_target_np(input_bboxes, anchor_sizes, anchor_ratios,
                           rpn_stride, rpn_h, rpn_w, image_h, image_w,
                           rpn_train_bs, iou_high_thres, iou_low_thres):
    '''Compute RPN target via numpy.'''
    input_bboxes, _ = unpad_np(input_bboxes)
    downscale = float(rpn_stride)
    num_anchors = len(anchor_sizes) * len(anchor_ratios)
    n_anchratios = len(anchor_ratios)
    # initialise empty output objectives
    output_height = rpn_h
    output_width = rpn_w
    resized_width = image_w
    resized_height = image_h
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))
    num_bboxes = input_bboxes.shape[0]
    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)
    # get the GT box coordinates, and resize to account for image resizing
    gta = input_bboxes[:, (1, 0, 3, 2)]
    # rpn ground truth
    anc_x, anc_y = np.meshgrid(np.arange(output_width), np.arange(output_height))
    ancs = make_anchors(anchor_sizes, anchor_ratios).reshape(-1, 2)
    xy_s = np.stack((anc_x, anc_y), axis=-1) + 0.5
    anc_pos = downscale * xy_s.reshape(output_height, output_width, 1, 2)
    anc_pos = np.broadcast_to(anc_pos, (output_height, output_width, ancs.shape[0], 2))
    anc_left_top = anc_pos - ancs/2.0
    anc_right_bot = anc_pos + ancs/2.0
    full_anc = np.concatenate((anc_left_top, anc_right_bot), axis=-1)
    # remove outlier anchors(set iou to zero.)
    full_anc = full_anc.reshape(-1, 4).astype(np.float32)
    valid_anc_mask = ((full_anc[:, 0] >= 0.) &
                      (full_anc[:, 1] >= 0.) &
                      (full_anc[:, 2] <= resized_width - 1.) &
                      (full_anc[:, 3] <= resized_height - 1.))
    iou_area = iou_np(full_anc.reshape(-1, 4), gta)
    valid_anc_mask_reshape = valid_anc_mask.astype(np.float32).reshape(iou_area.shape[0], 1)
    iou_area = iou_area * np.broadcast_to(valid_anc_mask_reshape, iou_area.shape)
    per_anchor = np.amax(iou_area, axis=1)
    # positive anchors by iou threshold
    positive_anchors = np.where(per_anchor >= iou_high_thres)[0]
    positive_anchors_gt = np.argmax(iou_area[positive_anchors, :], axis=1)
    positive_anchors_gt_unique = np.unique(positive_anchors_gt)
    # negative anchors by iou threshold, excluding outlier anchors
    negative_anchors = np.where(np.logical_and(per_anchor <= iou_low_thres,
                                               valid_anc_mask.astype(np.float32) > 0.0))[0]
    # build outputs
    # build positive anchors
    if positive_anchors.size > 0:
        positive_anchors_idxs = np.unravel_index(positive_anchors,
                                                 (output_height, output_width, num_anchors))
        y_is_box_valid[positive_anchors_idxs[0],
                       positive_anchors_idxs[1],
                       positive_anchors_idxs[2]] = 1
        y_rpn_overlap[positive_anchors_idxs[0],
                      positive_anchors_idxs[1],
                      positive_anchors_idxs[2]] = 1
        start = positive_anchors_idxs[2]
        best_regr = encode_anchor(full_anc[positive_anchors, :],
                                  gta[positive_anchors_gt, :])
        y_rpn_regr_4d_view = y_rpn_regr.reshape((output_height,
                                                 output_width,
                                                 num_anchors,
                                                 4))
        y_rpn_regr_4d_view[positive_anchors_idxs[0],
                           positive_anchors_idxs[1],
                           start,
                           :] = best_regr
    # build negative anchors
    if negative_anchors.size > 0:
        negative_anchors_idxs = np.unravel_index(negative_anchors,
                                                 (output_height, output_width, num_anchors))
        y_is_box_valid[negative_anchors_idxs[0],
                       negative_anchors_idxs[1],
                       negative_anchors_idxs[2]] = 1
        y_rpn_overlap[negative_anchors_idxs[0],
                      negative_anchors_idxs[1],
                      negative_anchors_idxs[2]] = 0

    # build other data for missed positive anchors.
    num_anchors_for_bbox[positive_anchors_gt_unique] = 1
    per_gt = np.amax(iou_area, axis=0)
    per_gt_best_anchor_idxs = np.argmax(iou_area, axis=0)
    # for testing
    per_gt_best_anc_ = np.copy(per_gt_best_anchor_idxs)
    per_gt_best_anc_[np.where(per_gt <= 0.0)[0]] = -1
    best_anc_idxs = np.unravel_index(per_gt_best_anchor_idxs,
                                     (output_height,
                                      output_width,
                                      len(anchor_sizes),
                                      len(anchor_ratios)))
    best_anchor_for_bbox[...] = np.stack(best_anc_idxs, -1)[:, (0, 1, 3, 2)]
    best_dx_for_bbox[...] = encode_anchor(full_anc[per_gt_best_anchor_idxs, :], gta)
    valid_best_anc_mask_out = np.where(per_gt <= 0.0)[0]
    best_anchor_for_bbox[valid_best_anc_mask_out, :] = [-1, -1, -1, -1]
    unmapped_box = np.where(num_anchors_for_bbox == 0)[0]
    unmapped_anc = per_gt_best_anc_[unmapped_box]
    unmapped_anc, _anc_unique_idx = _unique_no_sort(unmapped_anc[::-1])
    unmapped_anc = unmapped_anc[::-1]
    _box_unique_idx = unmapped_box.shape[0] - 1 - _anc_unique_idx
    _box_unique_idx = _box_unique_idx[::-1].astype(np.int32)
    unmapped_box = unmapped_box[_box_unique_idx]
    res = anchor_target_process(num_anchors_for_bbox,
                                best_anchor_for_bbox,
                                y_is_box_valid,
                                y_rpn_overlap,
                                y_rpn_regr,
                                best_dx_for_bbox,
                                n_anchratios,
                                rpn_train_bs)
    return res[0:2] + (iou_area, positive_anchors_gt, per_gt_best_anc_, full_anc, unmapped_box,
                       unmapped_anc) + res[2:]


def compute_rpn_target_np(input_bboxes, anchor_sizes, anchor_ratios,
                          rpn_stride, rpn_h, rpn_w, image_h, image_w,
                          rpn_train_bs, iou_high_thres, iou_low_thres):
    '''Wrapper for compute_rpn_target_np to remove iou_area output as it it mainly for testing.'''
    res = _compute_rpn_target_np(input_bboxes, anchor_sizes, anchor_ratios,
                                 rpn_stride, rpn_h, rpn_w, image_h, image_w,
                                 rpn_train_bs, iou_high_thres, iou_low_thres)
    return res[0:2]


def rpn_to_roi_kernel_np(rpn_layer, regr_layer, std_scaling,
                         anchor_sizes, anchor_ratios,
                         rpn_stride, dim_ordering, input_w, input_h,
                         use_regr=True, max_boxes=300,
                         overlap_thresh=0.9, rpn_pre_nms_top_N=0,
                         activation_type='sigmoid'):
    """Proposal layer in numpy."""
    regr_layer = regr_layer / std_scaling
    if dim_ordering == 0:
        (rows, cols) = rpn_layer.shape[2:]
    elif dim_ordering == 1:
        (rows, cols) = rpn_layer.shape[1:3]
    anc_x, anc_y = np.meshgrid(np.arange(cols), np.arange(rows))
    ancs = make_anchors(anchor_sizes, anchor_ratios).reshape(-1, 2)
    anc_pos = rpn_stride*(np.stack((anc_x, anc_y), axis=-1) + 0.5).reshape(rows, cols, 1, 2)
    anc_pos = np.broadcast_to(anc_pos, (rows, cols, ancs.shape[0], 2))
    anc_left_top = anc_pos - ancs/2.0
    full_anc_xywh = np.concatenate((anc_left_top,
                                    np.broadcast_to(ancs, anc_left_top.shape)),
                                   axis=-1).astype(np.float32)
    if dim_ordering == 0:
        # if NCHW, it's in (1, A*4, H, W) shape. do reshape
        regr = np.transpose(regr_layer[0, ...], (1, 2, 0))
    else:
        # if NHWC, it's in (1, H, W, A*4). no need to reshape.
        regr = regr_layer[0, ...]

    all_boxes = decode_anchor_np(full_anc_xywh.reshape(-1, 4), regr.reshape(-1, 4))
    # (H, W, A, 4) -> (A, H, W, 4) to match the prob shape: (A, H, W)
    all_boxes = all_boxes.reshape(rows, cols,
                                  ancs.size//2, 4).transpose((2, 0, 1, 3)).reshape((-1, 4))
    # back to x1, y1, x2, y2 format.
    all_boxes[:, 2] += all_boxes[:, 0]
    all_boxes[:, 3] += all_boxes[:, 1]
    all_boxes_save = np.copy(all_boxes)
    rpn_deltas_save = np.copy(regr.reshape(-1, 4))
    # clip boxes
    all_boxes[:, 0] = np.minimum(np.maximum(0., all_boxes[:, 0]), input_w-1.)
    all_boxes[:, 1] = np.minimum(np.maximum(0., all_boxes[:, 1]), input_h-1.)
    all_boxes[:, 2] = np.minimum(np.maximum(0., all_boxes[:, 2]), input_w-1.)
    all_boxes[:, 3] = np.minimum(np.maximum(0., all_boxes[:, 3]), input_h-1.)
    if dim_ordering == 1:
        if activation_type == 'softmax':
            rpn_layer = rpn_layer[:, :, :, rpn_layer.shape[3]//2:]
        all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))
    else:
        if activation_type == 'softmax':
            rpn_layer = rpn_layer[:, rpn_layer.shape[1]//2:, :, :]
        all_probs = rpn_layer.reshape((-1))
    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]
    # delete boxes whose width is less than one pixel.
    idxs = np.where((x1 - x2 < -1.) & (y1 - y2 < -1.))[0]
    all_boxes = all_boxes[idxs, :]
    all_probs = all_probs[idxs]
    if all_probs.shape[0] > rpn_pre_nms_top_N > 0:
        sorted_idx = np.argsort(-1.0*all_probs, kind='stable')
        all_boxes = np.copy(all_boxes[sorted_idx[0:rpn_pre_nms_top_N], :])
        all_probs = np.copy(all_probs[sorted_idx[0:rpn_pre_nms_top_N]])
    result = non_max_suppression_fast(np.copy(all_boxes),
                                      np.copy(all_probs),
                                      overlap_thresh=overlap_thresh,
                                      max_boxes=max_boxes)
    return result[0], full_anc_xywh, all_boxes_save, rpn_deltas_save


def _rpn_to_roi(rpn_layer, regr_layer,
                input_w, input_h, anchor_sizes,
                anchor_ratios, std_scaling, rpn_stride,
                use_regr=True, max_boxes=300,
                overlap_thresh=0.9, rpn_pre_nms_top_N=0):
    """Wrapper for proposal numpy layer."""
    anchor_sizes = anchor_sizes  # noqa pylint: disable = W0127
    anchor_ratios = [np.sqrt(r) for r in anchor_ratios]
    assert rpn_layer.shape[0] == 1
    dim_ordering = 0
    new_result = rpn_to_roi_kernel_np(rpn_layer, regr_layer, std_scaling,
                                      anchor_sizes, anchor_ratios,
                                      rpn_stride, dim_ordering,
                                      use_regr=use_regr, max_boxes=max_boxes,
                                      overlap_thresh=overlap_thresh,
                                      rpn_pre_nms_top_N=rpn_pre_nms_top_N,
                                      activation_type='sigmoid',
                                      input_w=input_w,
                                      input_h=input_h)
    return (np.expand_dims(new_result[0], axis=0), new_result[1], new_result[2],
            new_result[3])


def rpn_to_roi(rpn_layer, regr_layer,
               input_w, input_h, anchor_sizes,
               anchor_ratios, std_scaling, rpn_stride,
               use_regr=True, max_boxes=300,
               overlap_thresh=0.9, rpn_pre_nms_top_N=0):
    '''Wrapper for _rpn_to_roi to remove outputs for testing.'''
    return _rpn_to_roi(rpn_layer, regr_layer,
                       input_w, input_h, anchor_sizes,
                       anchor_ratios, std_scaling, rpn_stride,
                       use_regr=use_regr, max_boxes=max_boxes,
                       overlap_thresh=overlap_thresh,
                       rpn_pre_nms_top_N=rpn_pre_nms_top_N)[0]


def calc_iou_np(R, bboxes, class_ids, iou_high_thres,
                iou_low_thres, deltas_scaling, gt_as_roi=False,
                num_classes=21):
    """compute IoU for proposal layer."""
    # (y1, x2, y2, x2) to (x1, y1, x2, y2)
    gta = np.copy(bboxes[:, (1, 0, 3, 2)])
    R = np.copy(R[:, (1, 0, 3, 2)])
    if gt_as_roi:
        R = np.concatenate((R, gta), axis=0)
    iou_area = iou_np(R, gta)
    best_gt = np.argmax(iou_area, axis=1)
    best_gt_iou = np.amax(iou_area, axis=1)
    # build outputs
    # positive ROI labels
    positive_idx = np.where(best_gt_iou >= iou_high_thres)[0]
    if positive_idx.size > 0:
        positive_gt_idx = best_gt[positive_idx]
        positive_class_idx = class_ids[positive_gt_idx]
        positive_class_idx = np.array(positive_class_idx).astype(np.int32)
        positive_class_label = np.zeros((positive_idx.size, num_classes))
        positive_class_label[np.arange(positive_class_idx.size), positive_class_idx] = 1
        positive_coords = np.zeros((positive_idx.size, num_classes-1, 4))
        positive_labels = np.zeros((positive_idx.size, num_classes-1, 4))
        deltas = encode_anchor(R[positive_idx, :],
                               gta[positive_gt_idx, :],
                               scale=1.)
        sx = deltas_scaling[0]
        sy = deltas_scaling[1]
        sw = deltas_scaling[2]
        sh = deltas_scaling[3]
        coords_scaling = np.array([sx, sy, sw, sh]).reshape(1, 4)
        positive_coords[np.arange(positive_idx.size),
                        positive_class_idx,
                        :] = deltas*coords_scaling
        positive_labels[np.arange(positive_idx.size),
                        positive_class_idx,
                        :] = [1, 1, 1, 1]
    # negative ROI labels
    _lb = best_gt_iou >= iou_low_thres
    _ub = best_gt_iou < iou_high_thres
    negative_idx = np.where(np.logical_and(_lb, _ub))[0]
    if negative_idx.size > 0:
        negative_class_label = np.zeros((negative_idx.size, num_classes))
        negative_class_label[:, -1] = 1
        negative_coords = np.zeros((negative_idx.size, num_classes-1, 4))
        negative_labels = np.zeros((negative_idx.size, num_classes-1, 4))
    # positive and negative ROIs.
    pn_idx = np.concatenate((positive_idx, negative_idx))
    # it's must be either greater than 0.5 or less than 0.5. Can't be both empty.
    assert pn_idx.size > 0, '''Both positive and negative ROIs are empty,
     this cannot happen anyway.'''
    pn_ROIs = R[pn_idx, :]
    X = pn_ROIs
    if positive_idx.size > 0 and negative_idx.size > 0:
        Y1 = np.concatenate((positive_class_label, negative_class_label), axis=0)
    elif positive_idx.size > 0:
        Y1 = positive_class_label
    else:
        Y1 = negative_class_label
    if positive_idx.size > 0 and negative_idx.size > 0:
        _pos_labels_concat = np.concatenate((positive_labels, positive_coords), axis=-1)
        _neg_labels_concat = np.concatenate((negative_labels, negative_coords), axis=-1)
        Y2 = np.concatenate((_pos_labels_concat, _neg_labels_concat), axis=0)
    elif positive_idx.size > 0:
        Y2 = np.concatenate((positive_labels, positive_coords), axis=-1)
    else:
        Y2 = np.concatenate((negative_labels, negative_coords), axis=-1)
    # shapes: (B, 4), (B, C), (B, C-1, 8)
    return X, Y1, Y2


def sample_proposals(class_id_gts, roi_mini_batch, positive_ratio=0.25):
    """Sample ROIs in proposal layer."""
    negative_idx = np.where(class_id_gts[:, -1] == 1)[0]
    positive_idx = np.where(class_id_gts[:, -1] == 0)[0]
    positive_max_num = int(roi_mini_batch * positive_ratio)
    if positive_idx.size > 0 and negative_idx.size > 0:
        if positive_idx.size >= positive_max_num:
            pos_idx = np.random.choice(positive_idx, positive_max_num, replace=False)
        else:
            pos_idx = positive_idx
        neg_num = roi_mini_batch - pos_idx.size

        if negative_idx.size >= neg_num:
            neg_idx = np.random.choice(negative_idx, neg_num, replace=False)
        else:
            neg_idx = np.random.choice(negative_idx, neg_num, replace=True)
        sample_proposals =  np.concatenate((pos_idx, neg_idx))
    elif positive_idx.size > 0:
        if positive_idx.size >= roi_mini_batch:
            pos_idx = np.random.choice(positive_idx, roi_mini_batch, replace=False)
        else:
            pos_idx = np.random.choice(positive_idx, roi_mini_batch, replace=True)
        sample_proposals =  pos_idx
    elif negative_idx.size > 0:
        if negative_idx.size >= roi_mini_batch:
            neg_idx = np.random.choice(negative_idx, roi_mini_batch, replace=False)
        else:
            neg_idx = np.random.choice(negative_idx, roi_mini_batch, replace=True)
        sample_proposals = neg_idx
    else:
        sample_proposals =  np.array([], dtype=np.float32)
    return sample_proposals


def proposal_target_py_func(rois, gt_bboxes, gt_class_ids, iou_high_thres,
                            iou_low_thres, roi_train_bs, roi_positive_ratio,
                            deltas_scaling, bg_class_id):
    '''compute proposal target in numpy.'''

    def _core_func(rois, gt_bboxes, gt_class_ids, iou_high_thres,
                   iou_low_thres, roi_train_bs, roi_positive_ratio,
                   deltas_scaling, bg_class_id):
        rois, _ = unpad_np(rois)
        gt_bboxes, nz = unpad_np(gt_bboxes)
        gt_class_ids = gt_class_ids[nz]
        rois_np, cls_np, deltas_np = calc_iou_np(rois, gt_bboxes, gt_class_ids,
                                                 iou_high_thres, iou_low_thres,
                                                 deltas_scaling,
                                                 num_classes=bg_class_id+1)
        roi_idxs_np = sample_proposals(cls_np, roi_train_bs, roi_positive_ratio)
        # pad in case there is no any RoI.
        if np.size(roi_idxs_np) == 0:
            rois_ret = np.zeros((roi_train_bs, 4), dtype=np.float32)
            cls_ret = np.zeros((roi_train_bs, bg_class_id+1), dtype=np.float32)
            deltas_ret = np.zeros((roi_train_bs, bg_class_id*8), dtype=np.float32)
            return rois_ret, cls_ret, deltas_ret

        return (rois_np[roi_idxs_np, ...].astype(np.float32),
                cls_np[roi_idxs_np, ...].astype(np.float32),
                deltas_np[roi_idxs_np, ...].astype(np.float32))

    return tf.py_func(_core_func,
                      [rois, gt_bboxes, gt_class_ids, iou_high_thres,
                       iou_low_thres, roi_train_bs, roi_positive_ratio,
                       deltas_scaling, bg_class_id],
                      (tf.float32, tf.float32, tf.float32))


def normalize_rois(rois, rpn_h, rpn_w):
    """Normalize the ROIs to the range of (0, 1)."""
    x = rois[:, 0]
    y = rois[:, 1]
    x2 = rois[:, 2]
    y2 = rois[:, 3]
    rois_n = np.stack((y/np.array(rpn_h-1.0).astype(np.float32),
                       x/np.array(rpn_w-1.0).astype(np.float32),
                       y2/np.array(rpn_h-1.0).astype(np.float32),
                       x2/np.array(rpn_w-1.0).astype(np.float32)),
                      axis=-1)
    return rois_n


def normalize_rois_no_perm(rois, rpn_h, rpn_w):
    """Normalize the ROIs to the range of (0, 1), without changing the coordinate ordering."""
    y = rois[:, 0]
    x = rois[:, 1]
    y2 = rois[:, 2]
    x2 = rois[:, 3]
    rois_n = np.stack((y/np.array(rpn_h-1.0).astype(np.float32),
                       x/np.array(rpn_w-1.0).astype(np.float32),
                       y2/np.array(rpn_h-1.0).astype(np.float32),
                       x2/np.array(rpn_w-1.0).astype(np.float32)),
                      axis=-1)
    return rois_n


def normalize_rois_no_perm_pf(rois, rpn_h, rpn_w):
    '''py_func wrapper for normalize_rois_no_perm.'''
    return tf.py_func(normalize_rois_no_perm, [rois, rpn_h, rpn_w], tf.float32)
