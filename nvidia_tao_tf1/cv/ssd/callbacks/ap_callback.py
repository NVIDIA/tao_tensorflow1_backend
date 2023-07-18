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

"""Perform continuous RetinaNet training on a tfrecords dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.callbacks import Callback
import numpy as np

from nvidia_tao_tf1.cv.common.inferencer.evaluator import Evaluator
from nvidia_tao_tf1.cv.common.utils import ap_mode_dict
from nvidia_tao_tf1.cv.ssd.builders import eval_builder


class ap_callback(Callback):
    '''Callback function to calculate model mAP per k epoch.'''

    def __init__(self, val_dataset, every_k, im_width, im_height,
                 batch_size, classes, ap_mode, model_eval,
                 confidence_thresh, clustering_iou, top_k,
                 nms_max_output_size, matching_iou, verbose):
        '''init function, val_dataset should not be encoded.'''
        self.val_dataset = val_dataset
        self.im_width = im_width
        self.im_height = im_height
        self.classes = classes
        self.ap_mode = ap_mode
        self.model_eval = model_eval
        self.every_k = every_k if every_k > 0 else 5
        self.batch_size = batch_size if batch_size > 0 else 32
        self.confidence_thresh = confidence_thresh if confidence_thresh > 0 else 0.05
        self.clustering_iou = clustering_iou if clustering_iou > 0 else 0.5
        self.top_k = top_k if top_k > 0 else 200
        self.nms_max_output_size = nms_max_output_size if nms_max_output_size else 1000
        self.matching_iou = matching_iou if matching_iou > 0 else 0.5
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs):
        '''evaluates on epoch end.'''
        if (epoch + 1) % self.every_k != 0:
            for i in self.classes:
                logs['AP_' + i] = np.nan
            logs['mAP'] = np.nan
            return
        K.set_learning_phase(0)
        eval_model = eval_builder.build(self.model_eval,
                                        self.confidence_thresh,
                                        self.clustering_iou,
                                        self.top_k,
                                        self.nms_max_output_size)

        evaluator = Evaluator(keras_model=eval_model,
                              n_classes=len(self.classes),
                              batch_size=self.batch_size,
                              infer_process_fn=(lambda inf, y: y))

        results = evaluator(data_generator=self.val_dataset,
                            matching_iou_threshold=self.matching_iou,
                            num_recall_points=11,
                            average_precision_mode=ap_mode_dict[self.ap_mode],
                            verbose=self.verbose)

        mean_average_precision, average_precisions = results
        K.set_learning_phase(1)
        if self.verbose:
            print("*******************************")
        for i in range(len(average_precisions)):
            logs['AP_'+self.classes[i]] = average_precisions[i]
            if self.verbose:
                print("{:<14}{:<6}{}".format(self.classes[i], 'AP',
                                             round(average_precisions[i], 3)))
        if self.verbose:
            print("{:<14}{:<6}{}".format('', 'mAP', round(mean_average_precision, 3)))
            print("*******************************")
        logs['mAP'] = mean_average_precision
