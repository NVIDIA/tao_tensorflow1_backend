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
"""Loss functions used by FpeNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
import tensorflow as tf

from nvidia_tao_tf1.blocks.losses.loss import Loss
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.cv.fpenet.dataloader.fpenet_dataloader import (
    build_augmentation_config,
    get_all_transformations_matrices,
    get_transformation_ops
)


class FpeLoss(Loss):
    """Loss functions used by FpeNet."""

    @save_args
    def __init__(self,
                 loss_type='l1',
                 kpts_coeff=0.01,
                 weights_dict=None,
                 mask_occ=False,
                 elt_loss_info=None,
                 **kwargs):
        """Init function.

        Args:
            loss_type (str): Type of loss to use ('l1', 'square_euclidean', 'wing_loss').
            kpts_coeff (float): Coefficent the loss is multiplied with.
            weights_dict (dict of floats): Contains the weights for the 'eyes',
            the 'mouth', and the rest of the 'face'. These dict keys must be
            present, and the elements must sum up to 1.
            mask_occ (Boolean): If True, will mask all occluded points.
            elt_loss_info (dict): Dictionary about ELT loss from experiment spec.
                elt_alpha (float): Weight for ELT loss.
                enable_elt_loss (Bool): Flag to enable ELT loss.
                modulus_spatial_augmentation: Augmentation config.
        Raises:
            ValueError: If loss type is not a supported type (not 'l1',
            'square_euclidean' or 'wing_loss').
        """
        super(FpeLoss, self).__init__(**kwargs)
        self.kpts_coeff = kpts_coeff

        if weights_dict:
            assert type(
                weights_dict) == dict, 'Please provide a dict type object.'
            assert sum(weights_dict.values()) == 1.0,\
                'The sum of all class weights must be exactly 1.0.'
            assert all(key in weights_dict for key in ('face', 'eyes', 'mouth')),\
                'Please provide the correct dict entries as float values.'
        self.weights_dict = weights_dict
        self.mask_occ = mask_occ
        self.elt_loss_info = elt_loss_info
        self.loss_type = loss_type
        if loss_type == 'l1':
            self.loss_func = self.l1_loss
        elif loss_type == 'square_euclidean':
            self.loss_func = self.sqeuclidean_loss
        elif loss_type == 'wing_loss':
            self.loss_func = self.wing_loss
        else:
            raise ValueError('%s loss type not supported.' % loss_type)

    def __call__(self,
                 y_true,
                 y_pred,
                 occ_true,
                 occ_masking_info,
                 num_keypoints=80,
                 loss_name='landmarks'):
        """Model loss __call__ method.

        Args:
            y_true (tensor): The real ground truth value.
            y_pred (tensor): The predicted value.
            occ_true (tensor): Ground truth occlusions value.
            occ_masking_info (tensor): Ground truth to configure masking or
                                       no masking per set data.
            num_keypoints (int): Total number of keypoints.
            loss_name (str): String for loss type to add to logging.
        Returns:
            loss (tensor): Scalar loss.
        """
        # modify occlusions based on occ_masking_info
        # samples with no occlusion are set to 1.0, otherise 0.0
        # making occ for all samples not to be masked as 1.0
        occ_true_masked = tf.maximum(occ_true,
                                     tf.transpose(
                                        tf.tile([occ_masking_info],
                                                [num_keypoints, 1])
                                        )
                                     )

        # Mask occluded points
        if self.mask_occ:
            y_true = tf.multiply(y_true,
                                 tf.expand_dims(occ_true_masked, 2))
            y_pred = tf.multiply(y_pred,
                                 tf.expand_dims(occ_true_masked, 2))

        # Compute total loss.
        loss = self.loss_weighted(y_true, y_pred, self.kpts_coeff,
                                  self.loss_func, self.loss_type,
                                  self.weights_dict, num_keypoints, loss_name)
        return loss

    @staticmethod
    def l1_loss(y_true, y_pred, kpts_coeff):
        """Compute l1 loss.

        Args:
            y_true (tensor): The real ground truth value.
            y_pred (tensor): The predicted value.
            kpts_coeff (float): Coefficent the loss is multiplied with
                (dummy value here, in order to be compatible across parameters).

        Returns:
            loss (tensor): A scalar l1 loss computed with y_true and y_pred.
        """
        loss = K.mean(K.sum(K.abs(y_true - y_pred), axis=0))
        return loss

    @staticmethod
    def sqeuclidean_loss(y_true, y_pred, kpts_coeff):
        """Compute squared euclidean distance.

        Args:
            y_true (tensor): The real ground truth value.
            y_pred (tensor): The predicted value.
            kpts_coeff (float): Coefficent the loss is multiplied with.

        Returns:
            loss (tensor): A scalar distance error computed with y_true and y_pred.
        """
        loss = kpts_coeff * K.mean(K.sum(K.square(y_true - y_pred), axis=0))
        return loss

    @staticmethod
    def wing_loss(y_true, y_pred, kpts_coeff):
        """
        Compute wing loss as described in below paper.

        http://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_Wing_Loss_for_CVPR_2018_paper.pdf.

        Args:
            y_true (tensor): The real ground truth value.
            y_pred (tensor): The predicted value.
            kpts_coeff (float): Coefficent the loss is multiplied with.

        Returns:
            loss (tensor): A scalar distance error computed with y_true and y_pred.
        """
        # non-negative w sets the range of the nonlinear part to (−w, w)
        w = 10.0
        # epsilon limits the curvature of the nonlinear region
        epsilon = 2.0

        x = y_true - y_pred
        c = w * (1.0 - tf.math.log(1.0 + w/epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(
            tf.greater(w, absolute_x),
            w * tf.math.log(1.0 + absolute_x/epsilon),
            absolute_x - c
        )
        wing_loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)
        loss = kpts_coeff * wing_loss

        return loss

    @staticmethod
    def loss_weighted(y_true,
                      y_pred,
                      kpts_coeff,
                      loss_func,
                      loss_type,
                      weights_dict=None,
                      num_keypoints=80,
                      loss_name='landmarks'):
        """Compute squared euclidean distance.

        Args:
            y_true (tensor): The real ground truth value.
            y_pred (tensor): The predicted value.
            kpts_coeff (float): Coefficent the loss is multiplied with.
            loss_type (string): Type of loss- 'l1', 'square_euclidean' or `wing_loss'.
            weights_dict (dict of floats): Contains the weights for the 'eyes', the 'mouth',
                the 'pupil' and the rest of the 'face'. These dict keys must be present,
                and the elements must sum up to 1.
                ordering of points listed here-
                https://docs.google.com/document/d/13q8NciZtGyx5TgIgELkCbXGfE7PstKZpI3cENBGWkVw/edit#
            num_keypoints (int): Number of facial keypoints for computing the loss.
            loss_name (str): String for loss type to add to logging.

        Returns:
            loss (tensor): A scalar distance error computed with y_true and y_pred.
            mouth_loss (tensor): Loss for the mouth only.
            eyes_loss (tensor): Loss for the eyes only.
        """
        # Loss for all key points except for those on the mouth and the eyelids:
        eyelids_start_idx = 36
        face_true = y_true[:, 0:eyelids_start_idx, :]
        face_pred = y_pred[:, 0:eyelids_start_idx, :]
        face_loss = loss_func(face_true, face_pred, kpts_coeff)

        # Loss for 6 keypoints eyelids on each eye:
        eyelids_nkpts = 6 * 2
        eyelids_end_idx = eyelids_start_idx + eyelids_nkpts
        eyelids_true = y_true[:, eyelids_start_idx:eyelids_end_idx, :]
        eyelids_pred = y_pred[:, eyelids_start_idx:eyelids_end_idx, :]
        eyes_loss = loss_func(eyelids_true, eyelids_pred, kpts_coeff)

        # Loss for all keypoints on the mouth:
        mouth_start_idx = eyelids_end_idx
        mouth_end_idx = 68
        mouth_true = y_true[:, mouth_start_idx:mouth_end_idx, :]
        mouth_pred = y_pred[:, mouth_start_idx:mouth_end_idx, :]
        mouth_loss = loss_func(mouth_true, mouth_pred, kpts_coeff)

        # More facial points with 80 keypoints
        if (num_keypoints == 80):
            # Loss for pupils points
            pupils_start_idx = mouth_end_idx
            pupils_end_idx = 76
            pupils_true = y_true[:, pupils_start_idx:pupils_end_idx, :]
            pupils_pred = y_pred[:, pupils_start_idx:pupils_end_idx, :]
            pupils_loss = loss_func(pupils_true, pupils_pred, kpts_coeff)
            eyes_loss = eyes_loss + pupils_loss

            # Loss on remaining 4 ear points
            ears_start_idx = pupils_end_idx
            ears_end_idx = 80
            ears_true = y_true[:, ears_start_idx:ears_end_idx, :]
            ears_pred = y_pred[:, ears_start_idx:ears_end_idx, :]
            ears_loss = loss_func(ears_true, ears_pred, kpts_coeff)
            face_loss = face_loss + ears_loss

        if weights_dict:
            tf.compat.v1.summary.scalar(
                name=str('%s_face_loss' % loss_type), tensor=face_loss)
            tf.compat.v1.summary.scalar(
                name=str('%s_eyelids_loss' % loss_type), tensor=eyes_loss)
            tf.compat.v1.summary.scalar(
                name=str('%s_mouth_loss' % loss_type), tensor=mouth_loss)

            loss = (weights_dict['face'] * face_loss +
                    weights_dict['eyes'] * eyes_loss +
                    weights_dict['mouth'] * mouth_loss)
        else:
            loss = loss_func(y_true, y_pred, kpts_coeff)

        net_loss_name = str('%s_net_loss' % loss_type)
        if weights_dict:
            net_loss_name = str('weighted_%s' % net_loss_name)
        if loss_name == 'elt':
            net_loss_name = str('elt_%s' % net_loss_name)
        tf.compat.v1.summary.scalar(name=net_loss_name, tensor=loss)
        return loss, mouth_loss, eyes_loss


class FpeNetEltLoss(FpeLoss):
    """
    ELT loss used by FpeNet.

    Defined in- "Improving Landmark Localization with Semi-Supervised Learning"
    CVPR'2018
    """

    @save_args
    def __init__(self,
                 elt_loss_info,
                 image_height=80,
                 image_width=80,
                 image_channel=1,
                 num_keypoints=80,
                 **kwargs):
        """Init function.

        Args:
            elt_loss_info (dict): Information on ELT params.
                    elt_alpha (float): Weight for ELT loss.
                    enable_elt_loss (Bool): Flag to enable ELT loss.
                    modulus_spatial_augmentation: Augmentation config.
            image_height (int): Image height.
            image_width (int): Image width.
            image_channel (int): Number of image channels.
            num_keypoints (int): Number of facial keypoints.
        Returns:
            None
        """

        self.enable_elt_loss = elt_loss_info['enable_elt_loss']
        self.elt_alpha = elt_loss_info['elt_alpha']
        self.image_width = image_width
        self.image_height = image_height
        self.image_channel = image_channel
        self.num_keypoints = num_keypoints

        augmentation_info = elt_loss_info['modulus_spatial_augmentation']
        self.augmentation_config = build_augmentation_config(augmentation_info)

        frame_shape = [self.image_height, self.image_width, self.image_channel]
        frame_shape = map(float, frame_shape)
        self._stm_op, self._ctm_op, self._blur_op, \
            self._gamma_op, self._shift_op = \
            get_transformation_ops(self.augmentation_config, frame_shape)

    def transform_images(self, images):
        """Transforms the images with a random affine transformation.

        Args:
            images (Tensor): Decoded input images of shape (NCHW).

        Returns:
            transformed_images (Tensor): Transformed images tensor.
            sm (Tensor): 3x3 spatial transformation/augmentation matrix.
        """
        # get spatial transoformation matrix
        sm, _ = get_all_transformations_matrices(self.augmentation_config,
                                                 self.image_height,
                                                 self.image_width,
                                                 enable_augmentation=True)
        # Apply augmentations to frame tensors.
        transformed_images = []
        for i in range(images.shape[0]):
            transformed_image = self._apply_augmentations_to_frame(images[i, :, :, :], sm)
            transformed_images.append(tf.transpose(transformed_image, perm=[2, 0, 1]))

        transformed_images = tf.stack(transformed_images)
        transformed_images = transformed_images

        # return transformed images and transform matrix
        return(transformed_images, sm)

    def transform_points(self, ground_truth_labels, sm):
        """
        Transforms the (x,y) keypoints using a given tranformation matrix.

        Args:
            ground_truth_labels (Tensor) : a matrix of key_point locations (N x num_kpts x 2)
            sm (Tensor): 3x3 spatial transformation/augmentation matrix.
        Returns:
            kpts_norm (Tensor): Transformed points matrix of key_point locations (N x num_kpts x 2).
        """
        kpts_norm = []
        for i in range(ground_truth_labels.shape[0]):
            kpts_norm.append(self._apply_augmentations_to_kpts(ground_truth_labels[i, :, :], sm))

        kpts_norm = tf.stack(kpts_norm)

        return(kpts_norm)

    def _apply_augmentations_to_frame(self, input_tensor, sm):
        """
        Apply spatial and color transformations to an image.

        Spatial transform op maps destination image pixel P into source image location Q
        by matrix M: Q = P M. Here we first compute a forward mapping Q M^-1 = P, and
        finally invert the matrix.

        Args:
            input_tensor (Tensor): Input image frame tensors (HWC).
            sm (Tensor): 3x3 spatial transformation/augmentation matrix.

        Returns:
            image (Tensor, CHW): Augmented input tensor.
        """
        # Convert image to float if needed (stm_op requirement).
        if input_tensor.dtype != tf.float32:
            input_tensor = tf.cast(input_tensor, tf.float32)

        dm = tf.matrix_inverse(sm)
        # NOTE: Image and matrix need to be reshaped into a batch of one for this op.
        # Apply spatial transformations.

        input_tensor = tf.transpose(input_tensor, perm=[1, 2, 0])
        image = self._stm_op(images=tf.stack([tf.image.grayscale_to_rgb(input_tensor)]),
                             stms=tf.stack([dm]))
        image = tf.image.rgb_to_grayscale(image)

        image = tf.reshape(image, [self.image_height, self.image_width,
                                   self.image_channel])
        return image

    def _apply_augmentations_to_kpts(self, key_points, mapMatrix):
        """
        Apply augmentation to keypoints.

        This methods get matrix of keypoints and returns a matrix of
        their affine transformed location.

        Args:
            key_points: a matrix of key_point locations in the format (#key-points, 2)
            num_keypoints: number of keypoints
            MapMatrix: affine transformation of shape (2 * 3)

        Returns:
            A matrix of affine transformed key_point location in the
            format (#key-points, 2)
        """
        kpts = tf.concat([tf.transpose(key_points),
                          tf.ones([1, self.num_keypoints],
                          dtype=tf.float32)], axis=0)
        new_kpt_points = tf.matmul(tf.transpose(mapMatrix), kpts)
        new_kpt_points = tf.slice(new_kpt_points, [0, 0], [2, -1])

        return tf.transpose(new_kpt_points)
