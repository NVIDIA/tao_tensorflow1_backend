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
"""Utilitity functions for FasterRCNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os
import random

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

from nvidia_tao_tf1.cv.faster_rcnn.layers.utils import iou_np


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="DEBUG")
logger = logging.getLogger(__name__)


def get_init_ops():
    """Return all ops required for initialization."""
    """copied from dlav.common.graph.initializer"""
    return tf.group(tf.local_variables_initializer(),
                    tf.tables_initializer(),
                    *tf.get_collection('iterator_init'))


def resize_all(img, new_height, new_width):
    """Resize both the width and height."""
    (height, width) = img.shape[:2]
    if height == new_height and width == new_width:
        return img, (1.0, 1.0)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, (float(new_height)/height, float(new_width)/width)


def resize_min(img, min_side):
    """Resize both the width and height."""
    (height, width) = img.shape[:2]
    if height <= width:
        target_size = (int(width * (min_side / height)), min_side)
        ratio = (min_side / height, min_side / height)
    else:
        target_size = (min_side, int(height * (min_side / width)))
        ratio = (min_side / width, min_side / width)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    return img, ratio


def preprocess_images(img,
                      image_h,
                      image_w,
                      image_c,
                      image_min,
                      image_scaling_factor,
                      image_mean_values,
                      image_channel_order,
                      expand_dims=True):
    """Resize and normalize image."""
    if (image_h > 0 and image_w > 0):
        img, ratio = resize_all(img, image_h, image_w)
    elif image_min > 0:
        img, ratio = resize_min(img, image_min)
    else:
        raise(
            ValueError(
                "Either image static shape(height and width) or dynamic shape"
                "(minimal side) should be specified."
            )
        )
    if image_c == 3:
        assert len(image_mean_values) == 3, \
            'Image mean values length should be 3 for color image, got {}'.format(image_mean_values)
        if image_channel_order == 'rgb':
            img = img[:, :, ::-1]
        if image_channel_order == 'bgr':
            image_mean_values = image_mean_values[::-1]
        img -= np.array(image_mean_values)
        img /= image_scaling_factor
    elif image_c == 1:
        assert len(image_mean_values) == 1, '''
        Image mean values length should be 1 for
         grascale image, got {}'''.format(image_mean_values)
        img -= np.array(image_mean_values)
        img /= image_scaling_factor
        img = np.expand_dims(img, axis=2)
    else:
        raise ValueError('Unsupported image type with channel number: {}'.format(image_c))
    # HWC to CHW
    img = np.transpose(img, (2, 0, 1))
    # add batch dim
    if expand_dims:
        img = np.expand_dims(img, axis=0)
    return img, ratio


def preprocess_image_batch(imgs,
                           image_h,
                           image_w,
                           image_c,
                           image_min,
                           image_scaling_factor,
                           image_mean_values,
                           image_channel_order):
    """Resize and normalize a batch of images."""
    img_batch = []
    ratio_batch = []
    original_shape = []
    for img in imgs:
        _img, _ratio = preprocess_images(img.astype(np.float32), image_h,
                                         image_w, image_c, image_min,
                                         image_scaling_factor,
                                         image_mean_values,
                                         image_channel_order,
                                         expand_dims=False)
        img_batch.append(_img)
        ratio_batch.append(_ratio)
        original_shape.append(img.shape[:2])
    # (N, C, H, W), (N, 2)
    return np.stack(img_batch, axis=0), ratio_batch, original_shape


def get_original_coordinates(ratio, x1, y1, x2, y2, orig_h=None, orig_w=None):
    """compute original bbox given resized bbox."""
    if type(ratio) is tuple:
        real_x1 = int(round(x1 / ratio[1]))
        real_y1 = int(round(y1 / ratio[0]))
        real_x2 = int(round(x2 / ratio[1]))
        real_y2 = int(round(y2 / ratio[0]))
    elif type(ratio) is float:
        real_x1 = int(round(x1 / ratio))
        real_y1 = int(round(y1 / ratio))
        real_x2 = int(round(x2 / ratio))
        real_y2 = int(round(y2 / ratio))
    else:
        raise TypeError('invalid data type for ratio.')

    if orig_h is not None:
        real_y1 = max(min(real_y1, orig_h-1), 0)
        real_y2 = max(min(real_y2, orig_h-1), 0)
    if orig_w is not None:
        real_x1 = max(min(real_x1, orig_w-1), 0)
        real_x2 = max(min(real_x2, orig_w-1), 0)
    return (real_x1, real_y1, real_x2, real_y2)


def union(au, bu, area_intersection):
    """Union of two boxes."""
    area_a = (au[2] - au[0] + 1.) * (au[3] - au[1] + 1.)
    area_b = (bu[2] - bu[0] + 1.) * (bu[3] - bu[1] + 1.)
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    """Intersection of two boxes."""
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x + 1.
    h = min(ai[3], bi[3]) - y + 1.
    if w < 0 or h < 0:
        return 0
    return w*h


def calc_iou(a, b, scale=1.0):
    """IoU of two boxes."""
    sa = [ai*scale for ai in a]
    sb = [bi*scale for bi in b]
    if sa[0] >= sa[2] or sa[1] >= sa[3] or sb[0] >= sb[2] or sb[1] >= sb[3]:
        return 0.0
    area_i = intersection(sa, sb)
    area_u = union(sa, sb, area_i)
    return float(area_i) / float(area_u + 1e-6)


def calc_map(pred, gt_cls, gt_bbox, gt_diff, image_h, image_w, id_to_class,
             iou_thres=0.5):
    """Compute mAP."""
    T = {}
    P = {}
    valid_idx = np.where(gt_cls >= 0)[0]
    gt_cls = gt_cls[valid_idx]
    gt_bbox = gt_bbox[valid_idx, :]
    gt_diff = gt_diff[valid_idx]
    gt_bbox_cls_diff = np.concatenate(
        (gt_bbox, np.expand_dims(gt_cls, axis=1), np.expand_dims(gt_diff, axis=1)),
        axis=-1
    )
    gt = []
    for _bbox in gt_bbox_cls_diff:
        _gt = dict()
        _gt['bbox_matched'] = False
        _gt['x1'] = _bbox[1]
        _gt['x2'] = _bbox[3]
        _gt['y1'] = _bbox[0]
        _gt['y2'] = _bbox[2]
        _gt['class'] = id_to_class[_bbox[4]]
        _gt['difficult'] = _bbox[5]
        gt.append(_gt)
    pred_probs = np.array([s['prob'] for s in pred])
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]
    for box_idx in box_idx_sorted_by_prob:
        pred_box = pred[box_idx]
        pred_class = pred_box['class']
        pred_x1 = max(min(int(pred_box['x1']), image_w-1), 0)
        pred_x2 = max(min(int(pred_box['x2']), image_w-1), 0)
        pred_y1 = max(min(int(pred_box['y1']), image_h-1), 0)
        pred_y2 = max(min(int(pred_box['y2']), image_h-1), 0)
        pred_prob = pred_box['prob']
        if pred_class not in P:
            P[pred_class] = []
            T[pred_class] = []
        P[pred_class].append(pred_prob)
        max_ovp = -1.0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt):
            gt_class = gt_box['class']
            gt_x1 = int(gt_box['x1'])
            gt_x2 = int(gt_box['x2'])
            gt_y1 = int(gt_box['y1'])
            gt_y2 = int(gt_box['y2'])
            if gt_class != pred_class:
                continue
            iou = calc_iou((pred_x1, pred_y1, pred_x2, pred_y2),
                           (gt_x1, gt_y1, gt_x2, gt_y2))
            if iou > max_ovp:
                max_ovp = iou
                best_gt_idx = gt_idx
        if max_ovp >= iou_thres:
            if gt[best_gt_idx]['difficult']:
                # if best match is a difficult GT, ignore it.
                logger.warning("Got label marked as difficult(occlusion > 0), "
                               "please set occlusion field in KITTI label to 0 "
                               "and re-generate TFRecord dataset, if you want to "
                               "include it in mAP calculation "
                               "during validation/evaluation.")
                P[pred_class].pop()
            elif not gt[best_gt_idx]['bbox_matched']:
                # TP
                T[pred_class].append(1)
                gt[best_gt_idx]['bbox_matched'] = True
            else:
                # FP
                T[pred_class].append(0)
        else:
            # FP
            T[pred_class].append(0)
    for gt_box in gt:
        if not gt_box['bbox_matched'] and not gt_box['difficult']:
            if gt_box['class'] not in P:
                P[gt_box['class']] = []
                T[gt_box['class']] = []
            T[gt_box['class']].append(1)
            P[gt_box['class']].append(0)
    return T, P


def calc_rpn_recall(RPN_RECALL,
                    id_to_class,
                    rois_output,
                    gt_class_ids,
                    gt_bboxes,
                    overlap_thres=0.5):
    '''Mark TP and FN for each RoI to calculate RPN recall.'''
    # unpad zero gt boxes.
    val_idx = np.where(gt_class_ids >= 0)[0]
    gt_class_ids = gt_class_ids[val_idx]
    gt_bboxes = gt_bboxes[val_idx, :]
    ious = iou_np(rois_output, gt_bboxes)
    # for each gt box, find its best matched RoI
    ious_max = np.amax(ious, axis=0)
    overlapped_ious = (ious_max >= overlap_thres)
    for k in range(overlapped_ious.size):
        class_name = id_to_class[gt_class_ids[k]]
        if class_name not in RPN_RECALL:
            RPN_RECALL[class_name] = []
        RPN_RECALL[class_name].append(int(overlapped_ious[k]))


def voc_ap(rec, prec, use_07_metric=False, class_name=None, vis_path=None):
    """Compute VOC AP given precision and recall.

    Args:
        rec(array): recall vector.
        prec(array): precision vector.
        use_07_metric(bool): whether to use voc07 metric.
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        if class_name and vis_path:
            rec_arr = np.array(rec)
            prec_arr = np.array(prec)
            plt.plot(rec_arr, prec_arr, label=class_name)
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        if class_name and vis_path:
            plt.plot(mrec, mpre, label=class_name)
    return ap


def calc_ap(
    T, P, p_thres,
    use_voc07_metric=False,
    class_name=None,
    vis_path=None
):
    """compute the AP for a single class."""
    prec = []
    rec = []
    TP = 0.
    FP = 0.
    FN = 0.
    # sort according to prob.
    Ta = np.array(T)
    Pa = np.array(P)
    s_idx = np.argsort(-Pa)
    P = Pa[s_idx].tolist()
    T = Ta[s_idx].tolist()
    npos = np.sum(Ta)
    if p_thres <= 0.0:
        raise ValueError('''Object confidence score threshold should be
                          positive, got {}.'''.format(p_thres))
    for t, p in zip(T, P):
        if t == 1 and p >= p_thres:
            TP += 1
        elif t == 1 and p < p_thres:
            FN += 1
        elif t == 0 and p >= p_thres:
            FP += 1

        if TP+FP == 0.:
            precision = 0.
        else:
            precision = float(TP) / (TP+FP)

        if npos > 0:
            recall = float(TP) / float(npos)
        else:
            recall = 0.0

        prec.append(precision)
        rec.append(recall)
    rec_arr = np.array(rec)
    prec_arr = np.array(prec)
    ap = voc_ap(
        rec_arr, prec_arr,
        use_voc07_metric,
        class_name, vis_path
    )
    return ap, precision, recall, TP, FP, FN


def dump_kitti_labels(img_file,
                      all_dets,
                      ratio_2,
                      dump_dir,
                      prob_thres):
    '''Dump KITTI labels when doing KPI tests.'''
    label_file = '.'.join(os.path.basename(img_file).split('.')[0:-1]) + '.txt'
    assert dump_dir, 'KITTI label dump directory cannot be empty.'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    label_path = os.path.join(dump_dir, label_file)
    with open(label_path, 'w') as f:
        for det in all_dets:
            if det['prob'] >= prob_thres:
                f.write('{} 0 0 0 {:.2f} {:.2f} {:.2f} {:.2f}'
                        ' 0 0 0 0 0 0 0 {:.7f}\n'.format(
                            det['class'],
                            det['x1']*ratio_2[0],
                            det['y1']*ratio_2[1],
                            det['x2']*ratio_2[0],
                            det['y2']*ratio_2[1],
                            det['prob'])
                        )


def set_random_seed(seed):
    """set radom seed."""
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def debug_roi(rois, scores, img_name, output_dir):
    '''debug RoI.'''
    rois = np.reshape(rois, (-1, 4))
    scores = np.squeeze(scores, axis=0)
    y1 = rois[:, 0]
    x1 = rois[:, 1]
    y2 = rois[:, 2]
    x2 = rois[:, 3]
    for i in range(y1.shape[0]):
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        cv2.rectangle(img, (x1[i], y1[i]), (x2[i], y2[i]), [255, 255, 255], 2)
        basename = os.path.basename(img_name)
        output_file = os.path.join(output_dir, 'rois_{}_'.format(i)+basename)
        print('================', i, scores[i])
        cv2.imwrite(output_file, img)


def apply_regr(x, y, w, h, tx, ty, tw, th):
    """apply deltas to anchor boxes."""
    cx = x + w/2.
    cy = y + h/2.
    cx1 = tx * w + cx
    cy1 = ty * h + cy
    # clip to exp**50(~10**21) to avoid overflow in FP32.
    w1 = math.exp(np.minimum(tw, 50.)) * w
    h1 = math.exp(np.minimum(th, 50.)) * h
    x1 = cx1 - w1/2.
    y1 = cy1 - h1/2.
    return x1, y1, w1, h1


def read_images(imgs, channel_means, scaling, flip_channel=True):
    '''Read and preprocess images.'''
    np_imgs = []
    for im in imgs:
        with Image.open(im) as f:
            np_imgs.append(np.array(f))
    # NHWC to NCHW
    np_imgs = np.transpose(np.array(np_imgs, dtype=np.float32), (0, 3, 1, 2))
    # RGB to BGR
    if flip_channel:
        perm = np.array([2, 1, 0])
        np_imgs = np_imgs[:, perm, :, :]
        channel_means = channel_means[::-1]
    np_imgs -= np.array(channel_means).reshape([1, 3, 1, 1]).astype(np.float32)
    np_imgs /= scaling
    return np_imgs


def gen_det_boxes(
    id_to_class, nmsed_classes,
    nmsed_boxes, nmsed_scores,
    image_idx, num_dets,
):
    """Generate detected boxes."""

    def _gen_det_box(roi_idx):
        """Generate_detected_box."""
        # in case it is an invalid class ID
        if nmsed_classes[image_idx, roi_idx] not in id_to_class:
            return None
        cls_name = id_to_class[nmsed_classes[image_idx, roi_idx]]
        y1, x1, y2, x2 = nmsed_boxes[image_idx, roi_idx, :]
        det = {'x1': x1,
               'x2': x2,
               'y1': y1,
               'y2': y2,
               'class': cls_name,
               'prob': nmsed_scores[image_idx, roi_idx]}
        return det

    return list(filter(lambda x: x is not None, map(_gen_det_box, range(nmsed_classes.shape[1]))))


def get_detection_results(all_dets, gt_class_ids, gt_bboxes, gt_diff, image_h,
                          image_w, image_idx, id_to_class, T, P, iou_list):
    """Helper function to get the detection results."""
    for idx, iou in enumerate(iou_list):
        t, p = calc_map(
            all_dets,
            gt_class_ids[image_idx, ...],
            gt_bboxes[image_idx, ...],
            gt_diff[image_idx, ...],
            image_h,
            image_w,
            id_to_class,
            iou_thres=iou
        )
        for key in t.keys():
            if key not in T[idx]:
                T[idx][key] = []
                P[idx][key] = []
            T[idx][key].extend(t[key])
            P[idx][key].extend(p[key])


def compute_map_list(
    T, P, score_thres,
    use_voc07_metric,
    RPN_RECALL,
    iou_list,
    vis_path=None
):
    """Helper function to compute mAPs."""
    mAPs = []
    for idx, iou in enumerate(iou_list):
        all_aps = []
        print('='*90)
        if RPN_RECALL:
            print('{:<20}{:<20}{:<20}{:<20}{:<20}'.format('Class',
                                                          'AP',
                                                          'precision',
                                                          'recall',
                                                          'RPN_recall'))
        else:
            print('{:<20}{:<20}{:<20}{:<20}'.format('Class',
                                                    'AP',
                                                    'precision',
                                                    'recall'))
        print('-'*90)
        for key in sorted(list(T[idx].keys())):
            ap, prec, rec, _TP, _FP, _FN = calc_ap(
                T[idx][key], P[idx][key],
                score_thres, use_voc07_metric,
                class_name=key,
                vis_path=vis_path
            )
            if RPN_RECALL and (key in RPN_RECALL) and len(RPN_RECALL[key]):
                rpn_recall = sum(RPN_RECALL[key]) / len(RPN_RECALL[key])
                print('{:<20}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}'.format(
                    key, ap, prec, rec, rpn_recall)
                )
            else:
                print('{:<20}{:<20.4f}{:<20.4f}{:<20.4f}'.format(
                    key, ap, prec, rec)
                )
            print('-'*90)
            all_aps.append(ap)
        if vis_path is not None:
            plt.legend()
            plt.title("Precision-Recall curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.grid()
            save_path = os.path.join(vis_path, "PR_curve.png")
            plt.savefig(save_path)
            print(f"PR-curve image saved to {save_path}")
            plt.clf()
        mAP = np.mean(np.array(all_aps))
        mAPs.append(mAP)
        print('mAP@{} = {:<20.4f}'.format(iou, mAP))
    if len(iou_list) > 1:
        print('mAP@[{}:{}] = {:<20.4f}'.format(iou_list[0], iou_list[-1], np.mean(np.array(mAPs))))
    return mAPs
