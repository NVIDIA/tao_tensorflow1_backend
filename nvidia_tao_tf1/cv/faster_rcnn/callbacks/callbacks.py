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
'''Callbacks for FasterRCNN.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.callbacks import Callback
import numpy as np
from tqdm import tqdm

from nvidia_tao_tf1.cv.common import utils as iva_utils
from nvidia_tao_tf1.cv.faster_rcnn.utils import utils


class ModelSaver(Callback):
    '''Custom checkpointer for FasterRCNN, as a Keras callback.'''

    def __init__(self, model_path, key, interval):
        '''Initialize the checkpointer.'''
        self.model_path = model_path
        self.key = key
        self.ckpt_interval = interval

    def on_epoch_end(self, epoch, logs=None):
        '''On specified epoch end, save the encoded model.'''
        if (epoch + 1) % self.ckpt_interval == 0:
            model_file = self.model_path.format(epoch + 1)
            iva_utils.encode_from_keras(self.model,
                                        model_file,
                                        str.encode(self.key),
                                        only_weights=False)


class ValidationCallback(Callback):
    """A callback for online validation during training."""

    def __init__(self, val_model, val_data_loader,
                 val_interval, batch_size, prob_thres,
                 use_11point, id_to_class_map,
                 gt_matching_iou_list):
        """Initialize the validation hook."""
        self.val_model = val_model
        self.val_data_loader = val_data_loader
        self.val_interval = val_interval
        self.batch_size = batch_size
        self.prob_thres = prob_thres
        self.use_11point = use_11point
        self.id_to_class_map = id_to_class_map
        self.iou_list = gt_matching_iou_list
        self.max_iters = (
            (val_data_loader.num_samples + batch_size - 1) // batch_size
        )

    def on_epoch_end(self, epoch, logs=None):
        '''On specified epoch end, do validation.'''
        if (epoch + 1) % self.val_interval == 0:
            print("Doing validation at epoch {}(1-based index)...".format(epoch + 1))
            prev_lp = keras.backend.learning_phase()
            keras.backend.set_learning_phase(0)
            # set train model weights to validation model weights
            self.val_model.set_weights(self.model.get_weights())
            # do validation on validation model
            maps = self.do_validation()
            keras.backend.set_learning_phase(prev_lp)
            print("Validation done!")
            # Update logs with mAPs
            for _map, _iou in zip(maps, self.iou_list):
                logs[f"mAP@{_iou:.2f}"] = _map
            if len(maps) > 1:
                logs[f"mAP@[{self.iou_list[0]:.2f}:{self.iou_list[-1]:.2f}]"] = np.mean(maps)
            # Unified alias mAP for TAO API logging
            logs['mAP'] = np.mean(maps)

    def do_validation(self):
        """Conduct validation during training."""
        T = [dict() for _ in self.iou_list]
        P = [dict() for _ in self.iou_list]
        RPN_RECALL = {}
        for _ in tqdm(range(self.max_iters)):
            images, gt_class_ids, gt_bboxes, gt_diff = self.val_data_loader.get_array_with_diff()
            image_h, image_w = images.shape[2:]
            # get the feature maps and output from the RPN
            nmsed_boxes, nmsed_scores, nmsed_classes, num_dets, rois_output = \
                self.val_model.predict(images)
            # apply the spatial pyramid pooling to the proposed regions
            for image_idx in range(nmsed_boxes.shape[0]):
                all_dets = utils.gen_det_boxes(
                    self.id_to_class_map, nmsed_classes,
                    nmsed_boxes, nmsed_scores,
                    image_idx, num_dets,
                )
                # get detection results for each IoU threshold, for each image
                utils.get_detection_results(
                    all_dets, gt_class_ids, gt_bboxes, gt_diff, image_h,
                    image_w, image_idx, self.id_to_class_map, T, P, self.iou_list
                )
                # # calculate RPN recall for each class, this will help debugging
                utils.calc_rpn_recall(
                    RPN_RECALL,
                    self.id_to_class_map,
                    rois_output[image_idx, ...],
                    gt_class_ids[image_idx, ...],
                    gt_bboxes[image_idx, ...]
                )
        # finally, compute and print all the mAP values
        maps = utils.compute_map_list(
            T, P, self.prob_thres,
            self.use_11point, RPN_RECALL,
            self.iou_list
        )
        return maps
