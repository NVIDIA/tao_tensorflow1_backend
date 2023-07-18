# Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.

'''Numpy implementation of YOLOv3 label encoder.'''

import numpy as np
import tensorflow as tf


def iou(boxes1, boxes2):
    '''
    numpy version of vectorized iou.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    returns an `(m, n)` matrix with the IoUs for all pairwise boxes

    Arguments:
        boxes1 (array of shape (m, 4)): x_min, y_min, x_max, y_max
        boxes2 (array of shape (n, 4)): x_min, y_min, x_max, y_max

    Returns:
        IOU (array of shape (m, n)): IOU score
    '''

    # Compute the IoU.
    xmin1, ymin1, xmax1, ymax1 = np.split(boxes1, 4, axis=1)
    xmin2, ymin2, xmax2, ymax2 = np.split(boxes2, 4, axis=1)

    xmin = np.maximum(xmin1, xmin2.T)
    ymin = np.maximum(ymin1, ymin2.T)
    xmax = np.minimum(xmax1, xmax2.T)
    ymax = np.minimum(ymax1, ymax2.T)

    intersection = np.maximum(xmax - xmin, 0) * np.maximum(ymax - ymin, 0)
    boxes1_areas = (xmax1 - xmin1) * (ymax1 - ymin1)
    boxes2_areas = (xmax2 - xmin2) * (ymax2 - ymin2)
    union = boxes1_areas + boxes2_areas.T - intersection

    return intersection / union


class YOLOv3InputEncoder:
    '''
    Encoder class.

    Transforms ground truth labels for object detection in images
    (2D bounding box coordinates and class labels) to the format required for
    training an YOLO v3 model.

    In the process of encoding the ground truth labels, a template of anchor boxes
    is being built, which are subsequently matched to the ground truth boxes
    via an intersection-over-union threshold criterion.

    args:
        img_height: image height (how many pixels in height)
        img_width: image width (how many pixels in width)
        n_classes: Number of all possible classes.
        feature_map_stride: List of length n and format [(h_stride, w_stride), ...], n is number of
            feature maps. Stride is `input_size / fmap_size` and must be integer. The original paper
            input image is (416, 416) and feature map size is [(13, 13), (26, 26), (52, 52)]. The
            corresponding feature_map_stride should be [(32, 32), (16, 16), (8, 8)]
        anchors: List of 3 elements indicating the anchor boxes shape on feature maps. first element
            is for smallest feature map (i.e. to detect large objects). Last element is for largest
            feature map (i.e. to detect small objects). Each element is a list of tuples of size 2,
            in the format of (w, h). The length of the list can be any integer larger than 0. All
            w and h needs to be in range (0, 1) and this is (anchor_w / img_w, anchor_h / img_h)
    '''

    def __init__(self,  # pylint: disable=W0102
                 n_classes,
                 feature_map_stride=[(32, 32), (16, 16), (8, 8)],
                 anchors=[[(0.279, 0.216), (0.375, 0.476), (0.897, 0.784)],
                          [(0.072, 0.147), (0.149, 0.108), (0.142, 0.286)],
                          [(0.024, 0.031), (0.038, 0.072), (0.079, 0.055)]]):
        '''See class documentation for details.'''

        assert len(feature_map_stride) == len(anchors), "anchors and feature maps mismatch!"
        self.n_classes = n_classes
        self.fmap_num_anchors = [len(x) for x in anchors]
        self.feature_map_stride = feature_map_stride

        # maps box to fmap and corresponding box ind. {box_ind => (fmap, box_ind inside fmap)}
        self.fmap_box_dict = {}
        box_cnt = 0
        for idx, i in enumerate(anchors):
            for j, _ in enumerate(i):
                self.fmap_box_dict[box_cnt] = (idx, j)
                box_cnt += 1

        # w, h shape (9, 1)
        w, h = np.split(np.array(anchors).reshape(-1, 2), 2, axis=-1)

        # yxyx
        self.anchor_boxes = np.concatenate([-h / 2.0, -w / 2.0, h / 2.0, w / 2.0], axis=-1)
        self.one_hot_cls = np.eye(n_classes)

    def _loc_map_fn(self, output_img_size, cx, cy, anchor_idx):
        """Helper function to get location of anchor match.

        Returns:
            fmap_id, fmap_y, fmap_x, anchor_id_in_fmap
        """

        fmap_idx, anchor_idx_fmap = self.fmap_box_dict[anchor_idx]
        w_step = self.feature_map_stride[fmap_idx][1] / float(output_img_size[0])
        h_step = self.feature_map_stride[fmap_idx][0] / float(output_img_size[1])
        fmap_x = int(np.floor(np.clip(cx / w_step, 0, 1.0 / w_step - 1e-3)))
        fmap_y = int(np.floor(np.clip(cy / h_step, 0, 1.0 / h_step - 1e-3)))
        return fmap_idx, fmap_y, fmap_x, anchor_idx_fmap

    def __call__(self, output_img_size, gt_label):
        '''
        Processing one image groundtruthing.

        Args:
            output_img_size: (w, h) 2 integers representing image size
            gt_label: (#boxes, [class_idx, is_difficult, x_min, y_min, x_max, y_max])
        Returns:
            encoded_target: `(#anchor_boxes, [cy, cx, h, w, objectness, cls])`
        '''

        encoding_template = []

        for fmap_id, num_anchor in enumerate(self.fmap_num_anchors):
            template = np.zeros((output_img_size[1] // self.feature_map_stride[fmap_id][0],
                                 output_img_size[0] // self.feature_map_stride[fmap_id][1],
                                 num_anchor,
                                 5 + self.n_classes), dtype=np.float)
            encoding_template.append(template)

        # all shape (#boxes, 1)
        cls_id, _, xmin, ymin, xmax, ymax = np.split(gt_label, 6, axis=-1)

        cy = (ymin + ymax) / 2.0
        cx = (xmin + xmax) / 2.0
        h = ymax - ymin
        w = xmax - xmin

        gt_shape = np.concatenate([-h / 2.0, -w / 2.0, h / 2.0, w / 2.0], axis=-1)
        ious = iou(gt_shape, self.anchor_boxes)

        for gt_idx, gt_iou in enumerate(ious):
            gt_cx = cx[gt_idx, 0]
            gt_cy = cy[gt_idx, 0]
            gt_h = h[gt_idx, 0]
            gt_w = w[gt_idx, 0]
            # process best match
            loc = self._loc_map_fn(output_img_size, gt_cx, gt_cy, np.argmax(gt_iou))
            gt_array = np.concatenate(([gt_cy, gt_cx, gt_h, gt_w, 1.0],
                                       self.one_hot_cls[int(round(cls_id[gt_idx, 0]))]))
            encoding_template[loc[0]][loc[1], loc[2], loc[3], :] = gt_array

        encoding_template = [x.reshape(-1, 5 + self.n_classes) for x in encoding_template]
        return np.concatenate(encoding_template, axis=0)


class YOLOv3InputEncoderTensor:
    '''
    Encoder class.

    Transforms ground truth labels for object detection in images
    (2D bounding box coordinates and class labels) to the format required for
    training an YOLO v3 model.

    In the process of encoding the ground truth labels, a template of anchor boxes
    is being built, which are subsequently matched to the ground truth boxes
    via an intersection-over-union threshold criterion.

    args:
        img_height: image height (how many pixels in height)
        img_width: image width (how many pixels in width)
        n_classes: Number of all possible classes.
        feature_map_size: List of length n and format [(h,w), ...], n is number of feature maps and
            (h, w) is the last two dims of NCHW feature maps' shape
        anchors: List of 3 elements indicating the anchor boxes shape on feature maps. first element
            is for smallest feature map (i.e. to detect large objects). Last element is for largest
            feature map (i.e. to detect small objects). Each element is a list of tuples of size 2,
            in the format of (w, h). The length of the list can be any integer larger than 0.
    '''

    def __init__(self,  # pylint: disable=W0102
                 img_height,
                 img_width,
                 n_classes,
                 feature_map_size=None,
                 anchors=None):
        '''See class documentation for details.'''

        self.n_classes = n_classes
        self.fmap_size = tf.convert_to_tensor(feature_map_size)
        self.image_size = tf.convert_to_tensor([[img_height, img_width]], dtype=tf.float32)
        self.fmap_size_ratio = tf.truediv(
            self.image_size,
            tf.cast(self.fmap_size, dtype=tf.float32)
        )
        self.box_inds_in_fmap = tf.constant(sum([[j for j in range(len(anchors[i]))]
                                                 for i in range(len(anchors))], []), dtype=tf.int32)
        self.fmap_anchor_count = tf.constant([len(i) for i in anchors], dtype=tf.int32)
        # compute cumsum of box_idx_offset
        self.box_idx_offset = tf.constant([0], dtype=tf.int32)
        box_total = tf.constant(0, dtype=tf.int32)
        for i in range(len(feature_map_size)):
            box_per_fmap = feature_map_size[i][0] * feature_map_size[i][1] * len(anchors[i])
            _sum = self.box_idx_offset[-1:] + box_per_fmap
            box_total = box_total + box_per_fmap
            self.box_idx_offset = tf.concat([self.box_idx_offset, _sum], axis=-1)
        self.box_idx_offset = self.box_idx_offset[:-1]
        self.encoding_template = tf.zeros([box_total, 5 + n_classes], tf.float32)
        anchors = np.array(anchors)
        self.anchor_fmap_mapping = tf.constant(sum([[i] * len(anchors[i])
                                                    for i in range(len(anchors))], []),
                                               dtype=tf.int32)
        anchors = anchors.reshape(-1, 2)
        w = anchors[:, 0]
        h = anchors[:, 1]
        self.anchor_boxes = tf.constant(np.stack([-h / 2.0, -w / 2.0, h / 2.0, w / 2.0], axis=1),
                                        dtype=tf.float32)

    def __call__(self, ground_truth_labels):
        '''
        Converts ground truth bounding box data into a suitable format to train a YOLO v3 model.

        Arguments:
            ground_truth_labels (list): A python list of length `batch_size` that contains one 2D
                Numpy array for each batch image. Each such array has `k` rows for the `k` ground
                truth bounding boxes belonging to the respective image, and the data for each ground
                truth bounding box has the format `(class_id, xmin, ymin, xmax, ymax)` (i.e. the
                'corners' coordinate format), and `class_id` must be an integer greater than 0 for
                all boxes as class ID 0 is reserved for the background class.


        Returns:
            `y_encoded`, a 3D numpy array of shape
            `(batch_size, #boxes, [cy, cx, h, w, objectness, objectness_negative, cls])` that
            serves as the ground truth label tensor for training, where `#boxes` is the total number
            of boxes predicted by the model per image. cx, cy, h, w are centroid coord. cls is
            one-hot class encoding. objectness = 1 if matched to GT, else 0. objectness_negative = 1
            if not matched to GT and not neutral, else 0.
        '''
        encoded = []
        for gt_label in ground_truth_labels:  # For each batch item...
            match_y = tf.cond(tf.equal(tf.shape(gt_label)[0], 0),
                              lambda: self.encoding_template,
                              lambda label=gt_label: self.__process_one_img(label), strict=True)
            encoded.append(match_y)
        return tf.stack(encoded, axis=0)

    def __process_one_img(self, gt_label):
        '''
        TF graph for processing one image groundtruthing.

        Args:
            gt_label: 2D Numpy array for this image with `k` rows for the `k` ground
                truth bounding boxes belonging to the image, and the data for each ground
                truth bounding box has the format `(class_id, xmin, ymin, xmax, ymax)` (i.e. the
                'corners' coordinate format), and `class_id` must be an integer greater than 0 for
                all boxes as class ID 0 is reserved for the background class.
        Returns:
            encoded_target: `(#anchor_boxes, [cy, cx, h, w, objectness, objectness_negative, cls])`
        '''

        # nightmare level TF graph build.
        # Commented-out code is for single box match. which is easier to understand.
        # Real code matches all possible boxes with given GT togather.

        gt_label = tf.cast(gt_label, tf.float32)
        classes_one_hot = tf.one_hot(tf.reshape(tf.cast(gt_label[:, 0], tf.int32), [-1]),
                                     self.n_classes)
        cy = tf.truediv(gt_label[:, 3:4] + gt_label[:, 5:6], 2.0)
        cx = tf.truediv(gt_label[:, 2:3] + gt_label[:, 4:5], 2.0)
        h = gt_label[:, 5:6] - gt_label[:, 3:4]
        w = gt_label[:, 4:5] - gt_label[:, 2:3]
        objectness = tf.ones_like(w)
        # gt encoded as [:, 4+n_cls]
        one_hot_gt = tf.concat([cy, cx, h, w, objectness, classes_one_hot], axis=-1)

        # force center to (0, 0)
        gt_centroid_0 = tf.concat([tf.truediv(-h, 2.0), tf.truediv(-w, 2.0), tf.truediv(h, 2.0),
                                   tf.truediv(w, 2.0)], axis=-1)
        num_gt = tf.shape(gt_centroid_0)[0]
        iou_result = iou_tf(gt_centroid_0, self.anchor_boxes)

        # iou_match = tf.reshape(tf.argmax(iou_result, axis=-1, output_type=tf.int32), [-1])
        # size (#gt_box, #anchors)
        fmap_match_all = tf.tile(tf.reshape(self.anchor_fmap_mapping, [1, -1]), [num_gt, 1])
        # fmap_match = tf.gather(self.anchor_fmap_mapping, iou_match)

        # fmap_size_ratio = tf.gather(self.fmap_size_ratio, fmap_match, axis=0)
        # size (#gt_box, #anchors, 2)
        fmap_ratio_all = tf.gather(self.fmap_size_ratio, fmap_match_all, axis=0)

        # fmap_size = tf.gather(self.fmap_size, fmap_match, axis=0)
        # size (#gt_box, #anchors, 2)
        fmap_size_all = tf.gather(self.fmap_size, fmap_match_all, axis=0)

        # fmap_shift = tf.gather(self.box_idx_offset, fmap_match)
        # size (#gt_box, #anchors)
        fmap_shift_all = tf.gather(self.box_idx_offset, fmap_match_all)

        # anchor_count = tf.gather(self.fmap_anchor_count, fmap_match)
        anchor_count_all = tf.gather(self.fmap_anchor_count, fmap_match_all)

        cycx = one_hot_gt[..., :2]

        # adjusted_box_center = tf.truediv(cycx, fmap_size_ratio)
        # size (#gt, #anchor, 2)
        adjusted_box_center_all = tf.truediv(
            tf.reshape(cycx * self.image_size, [-1, 1, 2]),
            fmap_ratio_all
        )

        # box_center_cell = tf.cast(tf.maximum(tf.floor(adjusted_box_center - 1e-5), 0.0), tf.int32)
        box_center_limit = tf.truediv(
            tf.reshape(tf.ones_like(cycx) * self.image_size, [-1, 1, 2]),
            fmap_ratio_all
        )
        box_center_cell_all = tf.cast(
            tf.floor(tf.maximum(tf.minimum(adjusted_box_center_all, box_center_limit - 1e-3), 0.)),
            tf.int32
        )

        # cell_shift = (box_center_cell[..., 0] * fmap_size[..., 1] + box_center_cell[..., 1])
        # * anchor_count

        cell_shift_all = (box_center_cell_all[..., 0] * fmap_size_all[..., 1] +
                          box_center_cell_all[..., 1]) * anchor_count_all

        # anchor_shift = tf.gather(self.box_inds_in_fmap, iou_match)
        anchor_shift_all = tf.tile(tf.reshape(self.box_inds_in_fmap, [1, -1]), [num_gt, 1])

        # box_ind = fmap_shift + cell_shift + anchor_shift, (#gt, #anchors)
        box_ind_all = fmap_shift_all + cell_shift_all + anchor_shift_all
        iou_match = tf.reshape(tf.argmax(iou_result, axis=-1, output_type=tf.int32), [-1, 1])
        best_match_box = tf.gather_nd(
            box_ind_all,
            tf.concat([tf.reshape(tf.range(num_gt), [-1, 1]), iou_match], axis=-1)
        )
        encoded = tensor_slice_replace(
            self.encoding_template,
            one_hot_gt,
            best_match_box,
            tf.range(num_gt)
        )
        return encoded


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


def iou_tf(boxlist1, boxlist2, scope=None):
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


def tensor_slice_replace(a, b, a_idx, b_idx, scope=None):
    '''
    Returns a new tensor same as `a` but with `a[a_idx] = b[b_idx]`.

    Args:
        a, b (tensor): `a` and `b` must have same shape except for
            the first dimension.
        a_idx, b_idx (tensor): 1D tensors. `a_idx` and `b_idx` must
            have the same shape and all elements in `a_idx` should
            be smaller than `a.shape[0]`. Similar for `b_idx`

    Returns:
        c (tensor): A tensor same as `a` but with `a_idx` repalced
            by `b[b_idx]`.
    '''
    with tf.name_scope(scope, 'SliceReplace'):
        a_all_idx = tf.range(tf.shape(a)[0])
        _, a_remaining_idx = tf.setdiff1d(a_all_idx, a_idx)
        return tf.dynamic_stitch([a_remaining_idx, a_idx],
                                 [tf.gather(a, a_remaining_idx),
                                  tf.gather(b, b_idx)])
