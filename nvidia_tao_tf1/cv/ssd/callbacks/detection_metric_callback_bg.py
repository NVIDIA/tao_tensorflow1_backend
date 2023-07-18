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

"""Unified eval and mAP callback."""

from multiprocessing import cpu_count
import sys

from keras import backend as K
from keras.utils.data_utils import OrderedEnqueuer
import numpy as np
from tqdm import trange

from nvidia_tao_tf1.cv.common.callbacks.detection_metric_callback import DetectionMetricCallback
import nvidia_tao_tf1.cv.common.logging.logging as status_logging


class DetectionMetricCallbackBG(DetectionMetricCallback):
    '''
    Callback function to calculate model mAP / validation loss per k epoch.

    Args:
        ap_evaluator: object of class APEvaluator.
        built_eval_model: eval model built with additional layers for encoded output AND bbox
            output (model requires two outputs!!!)
        eval_sequence: Eval data sequence (based on keras sequence) that gives images, labels.
            labels is list (batch_size) of tuples (encoded_label, raw_label)
        loss_ops: three element tuple or list. [gt_placeholder, pred_placeholder, loss]
        eval_model: the training graph part of built_eval_model. Note, this model must share
            TF nodes with built_eval_model
        metric_interval: calculate model mAP per k epoch
        verbose: True if you want print ap message.
    '''

    def __init__(self, ap_evaluator, built_eval_model, eval_sequence, loss_ops, *args, **kwargs):
        """Init function."""

        super().__init__(ap_evaluator=ap_evaluator,
                         built_eval_model=built_eval_model,
                         eval_sequence=eval_sequence,
                         loss_ops=loss_ops,
                         *args, **kwargs)

        self.ap_evaluator = ap_evaluator
        self.built_eval_model = built_eval_model
        self.classes = eval_sequence.classes
        self.enqueuer = OrderedEnqueuer(eval_sequence, use_multiprocessing=False)
        self.n_batches = len(eval_sequence)
        self.loss_ops = loss_ops
        self.output_height = eval_sequence.output_height
        self.output_width = eval_sequence.output_width

    def _skip_metric(self, logs):
        for i in self.classes:
            logs['AP_' + i] = np.float64(np.nan)
        logs['mAP'] = np.float64(np.nan)
        logs['validation_loss'] = np.float64(np.nan)

    def _calc_metric(self, logs):
        total_loss = 0.0
        gt_labels = []
        pred_labels = []

        if self.verbose:
            tr = trange(self.n_batches, file=sys.stdout)
            tr.set_description('Producing predictions')
        else:
            tr = range(self.n_batches)

        self.enqueuer.start(workers=max(cpu_count() - 1, 1), max_queue_size=20)
        output_generator = self.enqueuer.get()

        # Loop over all batches.
        for _ in tr:
            # Generate batch.
            batch_X, batch_labs = next(output_generator)
            encoded_lab, gt_lab = zip(*batch_labs)

            # Predict.
            y_pred_encoded, y_pred = self.built_eval_model.predict(batch_X)

            batch_loss = K.get_session().run(self.loss_ops[2],
                                             feed_dict={self.loss_ops[0]: np.array(encoded_lab),
                                                        self.loss_ops[1]: y_pred_encoded})
            total_loss += np.sum(batch_loss) * len(gt_lab)

            gt_labels.extend(gt_lab)

            for i in range(len(y_pred)):
                y_pred_valid = y_pred[i][y_pred[i][:, 1] > self.ap_evaluator.conf_thres]
                y_pred_valid[..., 2] = np.clip(y_pred_valid[..., 2].round(), 0.0,
                                               self.output_width)
                y_pred_valid[..., 3] = np.clip(y_pred_valid[..., 3].round(), 0.0,
                                               self.output_height)
                y_pred_valid[..., 4] = np.clip(y_pred_valid[..., 4].round(), 0.0,
                                               self.output_width)
                y_pred_valid[..., 5] = np.clip(y_pred_valid[..., 5].round(), 0.0,
                                               self.output_height)
                pred_labels.append(y_pred_valid)

        self.enqueuer.stop()

        logs['validation_loss'] = total_loss / len(gt_labels)

        m_ap, ap = self.ap_evaluator(gt_labels, pred_labels, verbose=self.verbose)
        m_ap = np.mean(ap[1:])
        if self.verbose:
            print("*******************************")
        for i in range(len(self.classes)):
            logs['AP_' + self.classes[i]] = np.float64(ap[i+1])
            if self.verbose:
                print("{:<14}{:<6}{}".format(self.classes[i], 'AP', round(ap[i+1], 5)))
        if self.verbose:
            print("{:<14}{:<6}{}".format('', 'mAP', round(m_ap, 5)))
            print("*******************************")
            print("Validation loss:", logs['validation_loss'])

        logs['mAP'] = m_ap

        graphical_data = {
            "validation loss": round(logs['validation_loss'], 8),
            "mean average precision": round(logs['mAP'], 5)
        }
        s_logger = status_logging.get_status_logger()
        if isinstance(s_logger, status_logging.StatusLogger):
            s_logger.graphical = graphical_data
            s_logger.write(
                status_level=status_logging.Status.RUNNING,
                message="Evaluation metrics generated."
            )
