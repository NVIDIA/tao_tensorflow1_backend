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

"""IVA FasterRCNN Inputs loader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types.bbox_2d_label import Bbox2DLabel
import nvidia_tao_tf1.core as tao_core
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.build_dataloader import build_dataloader
from nvidia_tao_tf1.cv.faster_rcnn.data_augmentation.augmentation import hflip_bboxes, random_hflip
from nvidia_tao_tf1.cv.faster_rcnn.layers.utils import batch_op, calculate_delta_tf, \
                                         compute_rpn_target_np, iou_tf, \
                                         make_anchors, safe_gather, unique_with_inverse, \
                                         unpad_tf


class InputsLoader(object):
    """Data loader class for FasterRCNN.

    A data loader for FasterRCNN that can load TFRecord labels along with images.
    The format of the data source is the same as DetectNet_v2. Although ported from DetectNet_v2
    data loader, we made some customizations for FasterRCNN data loader as below:
    1. We append a background class in the class mapping and assign it the largest class ID.
    2. We can support both BGR, RGB and grayscale images as output of data loader.
    3. The images before mean subtraction and scaling is in the range 0-255 rather than (0, 1) as in
        DetectNet_v2.
    4. We apply a customizable per-channel mean subtraction to the image, then apply a scaling
        to it(also customizable). The mean values and scaling are configurable in spec file.
    5. We pad the groundtruth boxes per image to a maximum limit(defaults to 100) in order to
        support multi-batch and multi-gpu training in FasterRCNN.
    """

    def __init__(self,
                 dataset_config,
                 augmentation_config,
                 batch_size,
                 image_channel_num,
                 image_channel_means,
                 image_scaling_factor,
                 flip_channel,
                 max_objs_per_img=100,
                 training=True,
                 enable_augmentation=False,
                 session=None,
                 map_to_target_class_names=False,
                 visualizer=None,
                 rank=0):
        """Initialize data loader.

        Args:
            dataset_config(proto): the data source proto, the same as in DetectNet_v2.
            augmentation_config(proto): the data augmentation proto, also the same as DetectNet_v2.
            batch_size(int): the dataset batch size to output for the data loader.
            image_channel_num(int): the image channel number, can be 3(color) or 1(grayscale).
            image_channel_means(list of float): per channel image mean values for subtraction.
            image_scaling_factor(float): A scaling factor for preprocessing after mean subtraction.
            flip_channel(bool): Whether or not to flip the channels for image
                (RGB to BGR or vice-versa). The original image channel order is RGB in DetectNet_v2,
                If we want to use BGR ordering, we need a flip.
            max_objs_per_img(int): The maximum number of objects in an image. This is used to pad
                the groundtruth box numbers to the same number for batched training.
            training(bool): Training phase or test phase.
            enable_augmentation(bool): Whether or not to enable data augmentation.
            session(Keras session): the Keras(TF) session used to generate the numpy array from TF
                tensors for evaluation.
            map_to_target_class_names(bool): Whether or not to apply the map from source class
                names to target class names. This is useful when the detectnet_v2 data loader
                forgets to handle this properly. Defaults to False.
            visualizer(object): The visualizer object.
            rank(int): Horovod rank.
        """
        dataloader = build_dataloader(
            dataset_proto=dataset_config,
            augmentation_proto=augmentation_config)
        self.images, self.ground_truth_labels, self.num_samples = \
            dataloader.get_dataset_tensors(batch_size, training=training,
                                           enable_augmentation=enable_augmentation)
        if self.num_samples == 0:
            return

        cls_mapping_dict = dataset_config.target_class_mapping
        self.classes = sorted({str(x) for x in cls_mapping_dict.values()})
        # append background class so it maps to the largest number
        assert 'background' not in self.classes, '''
        Cannot have class name {} for ground truth objects'''.format('background')
        self.classes.append('background')
        cls_map = tao_core.processors.LookupTable(
            keys=self.classes,
            values=list(range(len(self.classes))),
            default_value=-1
        )
        cls_map.build()
        # use dynamic shape
        self.H = tf.shape(self.images)[2]
        self.W = tf.shape(self.images)[3]
        # preprocess input.
        self.images *= 255.0
        # save the original images for ease of testing/debugging
        self._original_images = self.images

        # do data augmentation if using dynamic shape
        if augmentation_config.preprocessing.output_image_min > 0:
            flipped_images, is_flipped = random_hflip(
                self.images[0, ...],
                augmentation_config.spatial_augmentation.hflip_probability,
                42
            )
            self.images = tf.expand_dims(flipped_images, axis=0)
        # Vis the augmented images in TensorBoard
        if rank == 0 and visualizer is not None:
            if visualizer.enabled:
                # NHWC format
                vis_image = tf.transpose(self.images, (0, 2, 3, 1))
        if image_channel_num == 3:
            if flip_channel:
                perm = tf.constant([2, 1, 0])
                self.images = tf.gather(self.images, perm, axis=1)
                image_channel_means = image_channel_means[::-1]
            self.images -= tf.constant(np.array(image_channel_means).reshape([1, 3, 1, 1]),
                                       dtype=tf.float32)
        elif image_channel_num == 1:
            self.images -= tf.constant(image_channel_means, dtype=tf.float32)
        else:
            raise ValueError('''Image channel number can only be 1
                              or 3, got {}.'''.format(image_channel_num))
        self.images /= image_scaling_factor

        gt_boxes = []
        gt_classes = []
        gt_diff = []
        if isinstance(self.ground_truth_labels, list):
            for l in self.ground_truth_labels:
                obj_id = cls_map(l['target/object_class'])
                x1 = tf.clip_by_value(tf.cast(tf.round(l['target/coordinates_x1']), tf.int32), 0,
                                      self.W - 1)
                x2 = tf.clip_by_value(tf.cast(tf.round(l['target/coordinates_x2']), tf.int32), 0,
                                      self.W - 1)
                y1 = tf.clip_by_value(tf.cast(tf.round(l['target/coordinates_y1']), tf.int32), 0,
                                      self.H - 1)
                y2 = tf.clip_by_value(tf.cast(tf.round(l['target/coordinates_y2']), tf.int32), 0,
                                      self.H - 1)
                # only select valid labels
                select = tf.logical_and(tf.not_equal(obj_id, -1),
                                        tf.logical_and(tf.less(x1, x2), tf.less(y1, y2)))
                label = tf.stack([y1, x1, y2, x2], axis=1)
                label = tf.boolean_mask(label, select)
                obj_id = tf.boolean_mask(obj_id, select)
                # pad to the same number of boxes for each image
                # such that we can concat and work with batch size > 1
                obj_count = tf.shape(label)[0]
                # Runtime check that objects number does not exceed this limit
                assert_op = tf.debugging.assert_less_equal(
                    obj_count,
                    max_objs_per_img,
                    message=('Maximum number of objects in ' +
                             'image exceeds the limit ' +
                             '{}'.format(max_objs_per_img)))
                with tf.control_dependencies([assert_op]):
                    num_pad = tf.maximum(max_objs_per_img - obj_count, 0)
                gt_classes.append(tf.pad(obj_id, [(0, num_pad)], constant_values=-1))
                gt_boxes.append(tf.pad(label, [(0, num_pad), (0, 0)]))
            self.frame_ids = [l['frame/id'] for l in self.ground_truth_labels]
        elif isinstance(self.ground_truth_labels, Bbox2DLabel):
            source_classes = self.ground_truth_labels.object_class
            # apply source class to target class mapping in case the
            # detectnet_v2 data loader forgets to apply it. Note that applying
            # the mapping twice has no problem so we are safe to do this.
            if map_to_target_class_names:
                target_classes_names = self._map_to_model_target_classes(
                    source_classes.values,
                    cls_mapping_dict)
                mapped_classes = tf.SparseTensor(
                    values=cls_map(target_classes_names),
                    indices=source_classes.indices,
                    dense_shape=source_classes.dense_shape)
            else:
                mapped_classes = tf.SparseTensor(
                    values=cls_map(source_classes.values),
                    indices=source_classes.indices,
                    dense_shape=source_classes.dense_shape)
            mapped_labels = self.ground_truth_labels._replace(object_class=mapped_classes)
            valid_indices = tf.not_equal(mapped_classes.values, -1)
            filtered_labels = mapped_labels.filter(valid_indices)
            filtered_obj_ids = tf.sparse.reshape(filtered_labels.object_class, [batch_size, -1, 1])
            filtered_coords = tf.sparse.reshape(filtered_labels.vertices.coordinates,
                                                [batch_size, -1, 4])
            filtered_occlusion = tf.sparse.reshape(
                filtered_labels.occlusion,
                [batch_size, -1, 1]
            )
            filtered_coords = tf.sparse.SparseTensor(
                values=tf.cast(tf.round(filtered_coords.values), tf.int32),
                indices=filtered_coords.indices,
                dense_shape=filtered_coords.dense_shape)
            filtered_occlusion = tf.sparse.SparseTensor(
                values=tf.cast(filtered_occlusion.values, tf.int32),
                indices=filtered_occlusion.indices,
                dense_shape=filtered_occlusion.dense_shape,
            )
            labels_all = tf.sparse.concat(
                axis=-1,
                sp_inputs=[filtered_obj_ids, filtered_occlusion, filtered_coords]
            )
            labels_split = tf.sparse.split(sp_input=labels_all, num_split=batch_size, axis=0)
            labels_split = [tf.sparse.reshape(x, [-1, 6]) for x in labels_split]
            labels = [tf.sparse.to_dense(self.get_non_empty_rows_2d_sparse(x))
                      for x in labels_split]
            for l in labels:
                obj_id = l[:, 0]
                # difficult/occlusion flag
                occ_diff = l[:, 1]
                x1 = tf.clip_by_value(l[:, 2], 0, self.W - 1)
                x2 = tf.clip_by_value(l[:, 4], 0, self.W - 1)
                y1 = tf.clip_by_value(l[:, 3], 0, self.H - 1)
                y2 = tf.clip_by_value(l[:, 5], 0, self.H - 1)
                # only select valid labels
                select = tf.logical_and(tf.not_equal(obj_id, -1),
                                        tf.logical_and(tf.less(x1, x2), tf.less(y1, y2)))
                label = tf.stack([y1, x1, y2, x2], axis=1)
                label = tf.boolean_mask(label, select)
                obj_id = tf.boolean_mask(obj_id, select)
                occ_diff = tf.boolean_mask(occ_diff, select)
                # pad to the same number of boxes for each image
                # such that we can concat and work with batch size > 1
                obj_count = tf.shape(label)[0]
                # Runtime check that objects number does not exceed this limit
                assert_op = tf.debugging.assert_less_equal(
                    obj_count,
                    max_objs_per_img,
                    message=('Maximum number of objects in ' +
                             'image exceeds the limit ' +
                             '{}'.format(max_objs_per_img)))
                with tf.control_dependencies([assert_op]):
                    num_pad = tf.maximum(max_objs_per_img - obj_count, 0)
                gt_classes.append(tf.pad(obj_id, [(0, num_pad)], constant_values=-1))
                gt_diff.append(tf.pad(occ_diff, [(0, num_pad)], constant_values=0))
                gt_boxes.append(tf.pad(label, [(0, num_pad), (0, 0)]))
            self.frame_ids = self.ground_truth_labels.frame_id
        else:
            raise TypeError('Groundtruth labels must be either list or Bbox2DLabel type.')
        self.gt_boxes = tf.cast(tf.stack(gt_boxes, axis=0), tf.float32)
        # flip bboxes if using dynamic shape
        if augmentation_config.preprocessing.output_image_min > 0:
            flipped_boxes = tf.cond(
                is_flipped,
                true_fn=lambda: hflip_bboxes(self.gt_boxes[0], self.W),
                false_fn=lambda: self.gt_boxes[0],
            )
            self.gt_boxes = tf.expand_dims(flipped_boxes, axis=0)
        self.gt_classes = tf.stack(gt_classes, axis=0)
        # we encode PASCAL VOC difficult objects as KITTI occlusion field >= 1
        self.gt_diff = tf.cast(tf.greater_equal(tf.stack(gt_diff, axis=0), 1), tf.int32)
        self.max_objs_per_img = max_objs_per_img
        self.session = session
        if rank == 0 and visualizer is not None:
            if visualizer.enabled:
                normalizer = tf.cast(
                    tf.stack([self.H, self.W, self.H, self.W], axis=0),
                    tf.float32
                )
                vis_boxes = self.gt_boxes / normalizer
                vis_image = tf.cast(
                    tf.image.draw_bounding_boxes(vis_image, vis_boxes),
                    tf.uint8
                )
                visualizer.image("augmented_images", vis_image, data_format="channels_last")

    def generate_rpn_targets(self, rpn_target_generator):
        '''Generate the RPN target tensors for training RPN.'''
        self.rpn_score_tensor, self.rpn_deltas_tensor = rpn_target_generator(self.gt_boxes)
        return self.rpn_score_tensor, self.rpn_deltas_tensor

    def get_array(self):
        '''get numpy array from the TF tensors for evaluation.'''
        return self.session.run([self.images, self.gt_classes, self.gt_boxes])

    def get_array_with_diff(self):
        '''get numpy array from the TF tensors for evaluation, with difficult tag.'''
        return self.session.run([self.images, self.gt_classes, self.gt_boxes, self.gt_diff])

    def get_array_and_frame_ids(self):
        '''get the array and frame IDs for a batch.'''
        return self.session.run([self.frame_ids, self.images, self.gt_classes, self.gt_boxes])

    def get_array_diff_and_frame_ids(self):
        '''get the array, difficult tag, and frame IDs for a batch.'''
        return self.session.run(
            [self.frame_ids, self.images, self.gt_classes, self.gt_boxes, self.gt_diff]
        )

    def get_non_empty_rows_2d_sparse(self, input_tensor):
        """
        Helper function to retrieve non-empty rows of a 2d sparse tensor.

        Args:
            input_tensor (tf.sparse.SparseTensor): must be 2-D
        Returns:
            output_tensor (tf.sparse.SparseTensor): output tensor with all rows non-empty
        """
        cols = input_tensor.dense_shape[1]
        empty_tensor = tf.sparse.SparseTensor(
            indices=tf.zeros(dtype=tf.int64, shape=[0, 2]),
            values=tf.zeros(dtype=input_tensor.dtype, shape=[0]),
            dense_shape=[0, cols])
        return tf.cond(tf.equal(tf.size(input_tensor.indices), 0),
                       true_fn=lambda: empty_tensor,
                       false_fn=lambda: self._get_non_empty_rows_2d_sparse_non_empty(input_tensor))

    def _get_non_empty_rows_2d_sparse_non_empty(self, input_tensor):
        """
        Helper function to retrieve non-empty rows of a 2d sparse tensor.

        Args:
            input_tensor (tf.sparse.SparseTensor): must be 2-D and non-empty
        Returns:
            output_tensor (tf.sparse.SparseTensor): output tensor with all rows non-empty
        """
        old_inds = input_tensor.indices
        _, new_rows = tf.unique(old_inds[:, 0], out_idx=tf.int64)
        num_new_rows = tf.reduce_max(new_rows) + 1
        cols = old_inds[:, 1]
        out_tensor = tf.sparse.SparseTensor(
            indices=tf.stack([new_rows, cols], axis=1),
            values=input_tensor.values,
            dense_shape=[num_new_rows, input_tensor.dense_shape[1]])
        return out_tensor

    def _map_to_model_target_classes(self, src_classes, target_class_mapping):
        """Map object classes as they are defined in the data source to the model target classes.

        Args:
            src_classes(str Tensor):  Source class names.
            target_class_mapping(Protobuf map): Map from source class names to target
                                                class names.

        Returns
            A str Tensor contains the mapped class names.
        """
        datasource_target_classes = list(target_class_mapping.keys())

        if len(datasource_target_classes) > 0:
            mapped_target_classes = list(target_class_mapping.values())
            default_value = tf.constant('-1')

            lookup = tao_core.processors.LookupTable(
                keys=datasource_target_classes,
                values=mapped_target_classes,
                default_value=default_value
            )
        return lookup(src_classes)


class RPNTargetGenerator(object):
    """Generate RPN Targets for training RPN heads.

    A Class used to generate the target tensors for RPN. Basically, this class partitions the
    anchor boxes into two categories: positive anchors and negative anchors. This paritition is by
    IoU between anchor boxes and groundtruth boxes. For each anchor box, if the IoU is larger than
    a specific threshold, then we regard it as a positive anchor. If an anchor's IoU with any of the
    groundtruth box is smaller than a specific threshold(different from the previous threshold),
    then we regard this anchor as a negative anchor. This is the foward pairing. But to ensure
    each groundtruth box is matched to at least one anchor box, we also need the backgward
    pairing. In the backward pairing, for each groundtruth box, we find the largest IoU with all
    the anchor boxes and the corresponding anchor box. As long as this largest IoU is not zero,
    we regard it as a successful pairing. For this successful pairing, we force assign this
    corresponding anchor as positive anchor regardless of its best IoU with all the groundtruth
    boxes(the best IoU can be very low, as long as it is not zero). With this kind of bipartite
    pairing, we can ensure each groundtruth box is paired with at lest one anchor boxes and RPN
    and RCNN will not miss any objects in groundtruth, as long as we use a reasonably good anchor
    design.
    """

    def __init__(self, image_w, image_h, rpn_w, rpn_h,
                 rpn_stride, anchor_sizes, anchor_ratios,
                 bs_per_gpu, iou_high_thres, iou_low_thres,
                 rpn_train_bs, max_objs_per_image=100):
        '''Initialize the RPNTargetGenerator.

        Args:
            image_w(int/Tensor): the input image width.
            image_h(int/Tensor): the input image height.
            rpn_w(int/Tensor): the width of the input feature map of RPN.
            rpn_h(int/Tensor): the height of the input feature map of RPN.
            rpn_stride(int): the rpn stride relative to input image(16).
            anchor_sizes(list): the list of anchor sizes, at input image scale.
            anchor_ratios(list): the list of anchor ratios.
            be_per_gpu(int): the image batch size per GPU.
            iou_high_thres(float): the higher threshold above which we regard anchors as positive.
            iou_low_thres(float): the lower threshold below which we regard anchors as negative.
            rpn_train_bs(int): the anchor batch size used to train the RPN.
            max_objs_per_image(int): the maximum number of groundtruth objects in one image.
        '''

        self.image_w = tf.cast(image_w, tf.float32)
        self.image_h = tf.cast(image_h, tf.float32)
        self.rpn_w = rpn_w
        self.rpn_h = rpn_h
        self.rpn_stride = rpn_stride
        self.anchor_sizes = anchor_sizes
        self.anchor_ratios = [np.sqrt(ar) for ar in anchor_ratios]
        self.num_anchors = len(self.anchor_sizes) * len(self.anchor_ratios)
        self.num_anc_ratios = len(self.anchor_ratios)
        self.bs_per_gpu = bs_per_gpu
        self.iou_high_thres = iou_high_thres
        self.iou_low_thres = iou_low_thres
        self.rpn_train_bs = rpn_train_bs
        self.max_objs_per_image = max_objs_per_image

    def generate_anchors(self):
        '''Generate anchors and use it later as constants.'''
        anc_x, anc_y = np.meshgrid(np.arange(self.rpn_w), np.arange(self.rpn_h))
        ancs = make_anchors(self.anchor_sizes, self.anchor_ratios).reshape(-1, 2)
        anc_pos = self.rpn_stride*(np.stack((anc_x, anc_y), axis=-1) + 0.5)
        anc_pos = anc_pos.reshape(self.rpn_h, self.rpn_w, 1, 2)
        # (H, W, A, 2)
        anc_pos = np.broadcast_to(anc_pos, (self.rpn_h, self.rpn_w, ancs.shape[0], 2))
        anc_left_top = anc_pos - ancs/2.0
        anc_right_bot = anc_pos + ancs/2.0
        # (y1, x1, y2, x2)
        full_anc = np.concatenate((anc_left_top[:, :, :, ::-1],
                                   anc_right_bot[:, :, :, ::-1]),
                                  axis=-1)
        self.full_anc = full_anc.reshape((-1, 4))

    def generate_anchors_tf(self):
        """Generate anchors using tensorflow ops in case input image shape is dynamic."""
        anc_x, anc_y = tf.meshgrid(tf.range(tf.cast(self.rpn_w, tf.float32)),
                                   tf.range(tf.cast(self.rpn_h, tf.float32)))
        ancs = make_anchors(self.anchor_sizes, self.anchor_ratios).reshape(-1, 2)
        anc_pos = self.rpn_stride * (tf.stack([anc_x, anc_y], axis=-1) + 0.5)
        anc_pos = tf.reshape(anc_pos, [self.rpn_h, self.rpn_w, 1, 2])
        anc_pos = tf.broadcast_to(anc_pos, [self.rpn_h, self.rpn_w, ancs.shape[0], 2])
        anc_left_top = anc_pos - (ancs / 2.0)
        anc_right_bot = anc_pos + (ancs / 2.0)
        full_anc = tf.concat([
            anc_left_top[:, :, :, ::-1],
            anc_right_bot[:, :, :, ::-1]
            ],
            axis=-1)
        self.full_anc = tf.reshape(full_anc, [-1, 4])

    def build_rpn_target_batch(self, input_gt_boxes):
        '''Batched processing of generating RPN target tensors.

        Args:
           input_gt_boxes(Tensor): the input groundtruth boxes, shape: (N, G, 4).

        Returns:
            the RPN target tensors: scores target tensor and deltas target tensor.
            shape:
                rpn_scores_gt: (N, A, H, W).
                rpn_deltas_gt: (N, A4, H, W).
        '''
        rpn_scores_gt, rpn_deltas_gt = batch_op([input_gt_boxes],
                                                self.rpn_target_tf,
                                                self.bs_per_gpu)
        return rpn_scores_gt, rpn_deltas_gt

    def _rpn_target_tf(self, input_gt_boxes):
        '''Generate RPN target tensors for single image.

        Since we are going to do some paddings for each image and the padding size are different
        for different image, so here we do this in a single image and then batch this set of Ops to
        achieve the batched processing. This function implemented the anchor pairing step with TF
        Ops to generate the positive and negative anchors, and also assign the groundtruth labels
        and bbox coordinate deltas for them. For positive anchors, we will assign both class IDs and
        bbox coordinate deltas to them, while for negative anchors we will only assign class IDs for
        them. Below is a quick breakdown of the code:
            1. Wrap the numpy anchors as TF constants.
            2. Calculate the IoU between anchor boxes and groundtruth boxes.
            3. Find the positive anchors and negative anchors by some IoU thresholds.
            4. Assign class IDs and(or) bbox deltas to each anchor.
            5. For each groundtruth box, find the best anchor that has largest IoU with it,
                and make sure the IoU is positive. Then force this anchor to be positive anchor
                and assign class IDs and bbox deltas to it.
            6. GT format tranformation to make the shape and dimension compatible with that of the
                RPN loss function.

        Args:
            input_gt_boxes(Tensor): the input groundtruth boxes, shape: (G, 4).

        Returns:
            the RPN scores and deltas target tensors for this single image.
        '''

        _ahw = tf.reshape(self.rpn_h*self.rpn_w*self.num_anchors, (1,))
        # (H, W, A, 4)
        if (
            isinstance(self.image_h, int) and
            isinstance(self.image_w, int) and
            isinstance(self.rpn_h, int) and
            isinstance(self.rpn_w, int)
           ):
            # if static shape
            self.generate_anchors()
            anchors_yxyx = tf.constant(self.full_anc, dtype=tf.float32)
        else:
            # if dynamic shape
            self.generate_anchors_tf()
            anchors_yxyx = self.full_anc
        valid_anc_mask = tf.logical_and(tf.logical_and(anchors_yxyx[:, 0] >= 0.,
                                                       anchors_yxyx[:, 1] >= 0.),
                                        tf.logical_and(anchors_yxyx[:, 2] <= self.image_h - 1.,
                                                       anchors_yxyx[:, 3] <= self.image_w - 1.))
        # unpad gt boxes and class IDs.
        input_gt_boxes, _ = unpad_tf(input_gt_boxes)

        ious = iou_tf(anchors_yxyx, input_gt_boxes)
        # set IoUs for invalid anchors to 0
        ious *= tf.expand_dims(tf.cast(valid_anc_mask, tf.float32), axis=1)
        iou_max_over_gt = tf.reduce_max(ious, axis=-1)
        positive_anchor_idxs = tf.where(iou_max_over_gt >= self.iou_high_thres)[:, 0]
        positive_gt_idxs = tf.cond(tf.size(positive_anchor_idxs) > 0,
                                   true_fn=lambda: tf.argmax(safe_gather(ious,
                                                                         positive_anchor_idxs),
                                                             axis=-1),
                                   false_fn=lambda: tf.constant([], dtype=tf.int64))
        positive_gt_idxs_unique, _ = tf.unique(positive_gt_idxs)

        negative_anchor_idxs = tf.where(tf.logical_and(iou_max_over_gt <= self.iou_low_thres,
                                                       valid_anc_mask))[:, 0]
        # build outputs
        # positive anchors

        def lambda_t1():
            return tf.scatter_nd(tf.reshape(tf.cast(positive_anchor_idxs, tf.int32),
                                            (-1, 1)),
                                 tf.ones_like(positive_anchor_idxs, dtype=tf.float32),
                                 _ahw)

        def lambda_f1():
            return tf.zeros(shape=_ahw, dtype=tf.float32)

        y_is_bbox_valid = tf.cond(tf.size(positive_anchor_idxs) > 0,
                                  true_fn=lambda_t1,
                                  false_fn=lambda_f1)

        def lambda_t2():
            return tf.scatter_nd(tf.reshape(tf.cast(positive_anchor_idxs, tf.int32),
                                            (-1, 1)),
                                 tf.ones_like(positive_anchor_idxs, tf.float32),
                                 _ahw)

        def lambda_f2():
            return tf.zeros(shape=_ahw, dtype=tf.float32)

        y_rpn_overlap = tf.cond(tf.size(positive_anchor_idxs) > 0,
                                true_fn=lambda_t2,
                                false_fn=lambda_f2)

        def lambda_t3():
            return calculate_delta_tf(safe_gather(anchors_yxyx, positive_anchor_idxs),
                                      safe_gather(input_gt_boxes, positive_gt_idxs))

        def lambda_f3():
            return tf.constant([], dtype=tf.float32)

        best_regr = tf.cond(tf.size(positive_anchor_idxs) > 0,
                            true_fn=lambda_t3,
                            false_fn=lambda_f3)

        def lambda_t4():
            return tf.scatter_nd(tf.reshape(tf.cast(positive_anchor_idxs, tf.int32),
                                            (-1, 1)),
                                 best_regr,
                                 tf.stack([_ahw[0], 4]))

        def lambda_f4():
            return tf.zeros(shape=tf.stack([_ahw[0], 4]), dtype=tf.float32)

        y_rpn_regr = tf.cond(tf.size(positive_anchor_idxs) > 0,
                             true_fn=lambda_t4,
                             false_fn=lambda_f4)
        # negative anchors

        def lambda_t5():
            return tf.scatter_nd(tf.reshape(tf.cast(negative_anchor_idxs, tf.int32),
                                            (-1, 1)),
                                 tf.ones_like(negative_anchor_idxs, tf.float32),
                                 _ahw)

        def lambda_f5():
            return tf.zeros(shape=_ahw, dtype=tf.float32)

        y_is_bbox_valid_n = tf.cond(tf.size(negative_anchor_idxs) > 0,
                                    true_fn=lambda_t5,
                                    false_fn=lambda_f5)

        # either positive or negative anchors are valid for training.
        y_is_bbox_valid += y_is_bbox_valid_n
        # find best match for missed GT boxes.
        # scatter_nd requires shape known at graph define time, so we use
        # self.max_objs_per_image and then slice to the actual gt box num

        def lambda_t6():
            return tf.scatter_nd(tf.reshape(tf.cast(positive_gt_idxs_unique,
                                                    tf.int32),
                                            (-1, 1)),
                                 tf.ones_like(positive_gt_idxs_unique, tf.float32),
                                 tf.constant([self.max_objs_per_image]))

        def lambda_f6():
            return tf.zeros(shape=[self.max_objs_per_image], dtype=tf.float32)

        num_anchors_for_bbox = tf.cond(tf.size(positive_gt_idxs_unique) > 0,
                                       true_fn=lambda_t6,
                                       false_fn=lambda_f6)
        input_gt_boxes_mask = tf.ones_like(input_gt_boxes[:, 0], dtype=tf.float32)
        input_gt_boxes_mask_idxs = tf.where(input_gt_boxes_mask > 0.0)[:, 0]
        num_anchors_for_bbox = safe_gather(num_anchors_for_bbox, input_gt_boxes_mask_idxs)

        iou_max_over_anc = tf.reduce_max(ious, axis=0)
        iou_max_best_anc = tf.argmax(ious, axis=0)
        best_dx_for_bbox = calculate_delta_tf(safe_gather(anchors_yxyx, iou_max_best_anc),
                                              input_gt_boxes)
        iou_max_best_anc = tf.where(iou_max_over_anc <= 0.0,
                                    -1.0 * tf.ones_like(iou_max_best_anc, tf.float32),
                                    tf.cast(iou_max_best_anc, tf.float32))
        unmapped_gt_box_idxs_ = tf.where(tf.logical_and(tf.equal(num_anchors_for_bbox, 0.0),
                                                        iou_max_over_anc > 0.0))[:, 0]
        unmapped_ancs_for_boxes_ = safe_gather(iou_max_best_anc, unmapped_gt_box_idxs_)
        # as more than one gt boxes can map to the same anchors, we should remove duplication
        # so that an anchor is mapped to a single box, we use the last box as used in V1
        # at this moment for ease of testing.
        # find the last occurence of each mapped anchor
        unmapped_ancs_for_boxes_rev, _, _orig_idx = \
            unique_with_inverse(unmapped_ancs_for_boxes_[::-1])
        unmapped_ancs_for_boxes = unmapped_ancs_for_boxes_rev[::-1]
        bbox_sel_idx = (tf.shape(unmapped_ancs_for_boxes_)[0] - 1 - _orig_idx)[::-1]
        unmapped_gt_box_idxs = safe_gather(unmapped_gt_box_idxs_, bbox_sel_idx)

        # short alias
        uafb = unmapped_ancs_for_boxes
        uafb_int32 = tf.cast(uafb, tf.int32)

        def lambda_t7():
            return tf.scatter_nd(tf.reshape(uafb_int32, (-1, 1)),
                                 tf.ones_like(uafb, tf.float32),
                                 _ahw)

        def lambda_f7():
            return tf.zeros(shape=_ahw, dtype=tf.float32)

        y_is_bbox_valid_patch = tf.cond(tf.size(uafb) > 0,
                                        true_fn=lambda_t7,
                                        false_fn=lambda_f7)

        y_is_bbox_valid = tf.where(y_is_bbox_valid_patch > 0.0,
                                   y_is_bbox_valid_patch,
                                   y_is_bbox_valid)
        y_rpn_overlap = tf.where(y_is_bbox_valid_patch > 0.0,
                                 y_is_bbox_valid_patch,
                                 y_rpn_overlap)

        def lambda_t8():
            return tf.scatter_nd(tf.reshape(uafb_int32, (-1, 1)),
                                 safe_gather(best_dx_for_bbox, unmapped_gt_box_idxs),
                                 tf.stack([_ahw[0], 4]))

        def lambda_f8():
            return tf.zeros(shape=tf.stack([_ahw[0], 4]), dtype=tf.float32)

        y_rpn_regr_patch = tf.cond(tf.size(uafb) > 0,
                                   true_fn=lambda_t8,
                                   false_fn=lambda_f8)

        y_rpn_regr = tf.where(tf.tile(tf.reshape(y_is_bbox_valid_patch, (-1, 1)), [1, 4]) > 0.0,
                              y_rpn_regr_patch,
                              y_rpn_regr)
        # (H, W, A) to (A, H, W)
        y_rpn_regr = tf.reshape(y_rpn_regr, (self.rpn_h, self.rpn_w, self.num_anchors*4))
        y_rpn_regr = tf.transpose(y_rpn_regr, (2, 0, 1))
        y_rpn_overlap = tf.reshape(y_rpn_overlap, (self.rpn_h, self.rpn_w, self.num_anchors))
        y_rpn_overlap = tf.transpose(y_rpn_overlap, (2, 0, 1))
        y_rpn_overlap_save = y_rpn_overlap
        y_is_bbox_valid = tf.reshape(y_is_bbox_valid, (self.rpn_h, self.rpn_w, self.num_anchors))
        y_is_bbox_valid = tf.transpose(y_is_bbox_valid, (2, 0, 1))
        y_is_bbox_valid_save = y_is_bbox_valid
        pos_locs = tf.where(tf.logical_and(tf.equal(y_rpn_overlap, 1.0),
                                           tf.equal(y_is_bbox_valid, 1.0)))
        neg_locs = tf.where(tf.logical_and(tf.equal(y_rpn_overlap, 0.0),
                                           tf.equal(y_is_bbox_valid, 1.0)))
        num_pos = tf.shape(pos_locs)[0]
        pos_limit = tf.floor_div(self.rpn_train_bs, 2)
        real_pos_num = tf.minimum(num_pos, pos_limit)
        num_pos_delete = num_pos - real_pos_num
        delete_idxs = tf.random_shuffle(tf.where(tf.reshape(y_rpn_overlap,
                                                            (-1,)))[:, 0])[:num_pos_delete]

        def lambda_t9():
            return tf.scatter_nd(tf.reshape(tf.cast(delete_idxs, tf.int32), (-1, 1)),
                                 tf.ones_like(delete_idxs, tf.float32),
                                 _ahw)

        def lambda_f9():
            return tf.zeros(shape=_ahw, dtype=tf.float32)

        delete_mask_ = tf.cond(tf.size(delete_idxs) > 0,
                               true_fn=lambda_t9,
                               false_fn=lambda_f9)

        delete_mask = tf.reshape(1.0 - delete_mask_,
                                 (self.num_anchors, self.rpn_h, self.rpn_w))
        y_rpn_overlap = y_rpn_overlap * delete_mask

        y_is_bbox_valid = y_is_bbox_valid * delete_mask

        num_neg = tf.shape(neg_locs)[0]
        neg_limit = tf.minimum(self.rpn_train_bs - real_pos_num, num_neg)
        num_neg_delete = num_neg - neg_limit
        neg_idxs = tf.where(tf.logical_and(tf.equal(tf.reshape(y_rpn_overlap, (-1,)), 0.0),
                                           tf.equal(tf.reshape(y_is_bbox_valid, (-1,)), 1.0)))[:, 0]
        neg_delete_idxs = tf.random_shuffle(neg_idxs)[:num_neg_delete]

        def lambda_t10():
            return tf.scatter_nd(tf.reshape(tf.cast(neg_delete_idxs, tf.int32), (-1, 1)),
                                 tf.ones_like(neg_delete_idxs, tf.float32),
                                 _ahw)

        def lambda_f10():
            return tf.zeros(shape=_ahw, dtype=tf.float32)

        neg_delete_mask_ = tf.cond(tf.size(neg_delete_idxs) > 0,
                                   true_fn=lambda_t10,
                                   false_fn=lambda_f10)

        neg_delete_mask = tf.reshape(1.0 - neg_delete_mask_,
                                     (self.num_anchors, self.rpn_h, self.rpn_w))
        y_is_bbox_valid *= neg_delete_mask

        rpn_class_targets = tf.concat([y_is_bbox_valid, y_rpn_overlap], axis=0)
        y_rpn_overlap_tiled = tf.tile(tf.reshape(y_rpn_overlap,
                                                 (self.num_anchors, 1, self.rpn_h, self.rpn_w)),
                                      [1, 4, 1, 1])
        y_rpn_overlap_tiled = tf.reshape(y_rpn_overlap_tiled,
                                         (self.num_anchors*4, self.rpn_h, self.rpn_w))
        rpn_bbox_targets = tf.concat([y_rpn_overlap_tiled, y_rpn_regr], axis=0)
        return (rpn_class_targets, rpn_bbox_targets, ious,
                positive_gt_idxs, iou_max_best_anc, anchors_yxyx,
                unmapped_gt_box_idxs, unmapped_ancs_for_boxes,
                y_is_bbox_valid_save, y_rpn_overlap_save)

    def rpn_target_tf(self, input_gt_boxes):
        '''Wrapper for _rpn_target_tf, to remove ious output as it is for debug and testing.'''
        return self._rpn_target_tf(input_gt_boxes)[0:2]

    def rpn_target_py_func(self, inpu_gt_boxes):
        '''py_func implementation of rpn_target generator.'''

        def py_func_core(input_boxes):
            rpn_cls, rpn_deltas = \
                 compute_rpn_target_np(input_boxes, self.anchor_sizes, self.anchor_ratios,
                                       self.rpn_stride, self.rpn_h, self.rpn_w,
                                       self.image_h, self.image_w,
                                       self.rpn_train_bs, self.iou_high_thres,
                                       self.iou_low_thres)
            return rpn_cls[0, ...].astype(np.float32), rpn_deltas[0, ...].astype(np.float32)

        return tf.py_func(py_func_core, [inpu_gt_boxes], (tf.float32, tf.float32))
