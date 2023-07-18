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

"""A class to evaluate a DetectNet V2 model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging
import sys
from timeit import default_timer
import numpy as np
from six.moves import range
from tabulate import tabulate
import tensorflow as tf

import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.detectnet_v2.evaluation.compute_metrics import ComputeMetrics
from nvidia_tao_tf1.cv.detectnet_v2.evaluation.ground_truth import process_batch_ground_truth
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.postprocessing import PostProcessor

logger = logging.getLogger(__name__)


class Evaluator(object):
    '''
    Computes detection metrics for gridbox model.

    Class to compute metrics suite: mAP, average precision (VOC 2009)
    default IOU = 0.5
    '''

    def __init__(self,
                 postprocessing_config,
                 evaluation_config,
                 gridbox_model,
                 images,
                 ground_truth_labels,
                 steps,
                 confidence_models=None,
                 target_class_mapping=None,
                 sqlite_paths=None):
        '''
        Init function.

        Arguments:
            postprocessing_config: default object for postprocessing
            evaluation_config: default object for evaluation
            gridbox_model: gridbox model object
            images: image tensor on which evaluation will run
            ground_truth_labels: ground truth tensor
            setps: When to run evaluation
            confidence_models: If MLP or any confidence regressor is used
            target_class_mapping: how classes are mapped
        '''
        self._postprocessing_config = postprocessing_config
        self._evaluation_config = evaluation_config
        self._images = images
        self._ground_truth_labels = ground_truth_labels
        self._steps = steps
        self._target_class_mapping = target_class_mapping
        self._target_class_names = gridbox_model.get_target_class_names()
        self._confidence_models = None
        self.confidence_models = {}
        self.gridbox = gridbox_model
        self._postprocessor = PostProcessor(
            postprocessing_config=self._postprocessing_config,
            confidence_models=self._confidence_models,
            image_size=(self.gridbox.input_width, self.gridbox.input_height))
        self._sqlite_paths = None

    @property
    def keras_models(self):
        """Return list of Keras models to be loaded in current session."""
        self._keras_models = [self.gridbox.keras_model]
        self._keras_models.extend([conf_model.keras_model for conf_model
                                   in self.confidence_models.values()])
        return self._keras_models

    @property
    def ground_truth_labels(self):
        """Wrap labels to be used for validation.

        Child classes may override this if needed. This base class does nothing.
        """
        return self._ground_truth_labels

    @staticmethod
    def get_session_config():
        """Return session configuration specific to this Evaluator."""
        gpu_options = tf.compat.v1.GPUOptions(
            allow_growth=True
        )
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        return config

    def _get_validation_iterator(self, session, dataset_percentage):
        """Generator that yields batch predictions, labels, and cost.

        Args:
            session (tf.Session): Session to be used for evaluation.
            dataset_percentage (float): % of the dataset to evaluate.

        Returns:
            batch_predictions: Raw predictions for current batch.
            batch_ground_truth_labels: List of ground truth labels dicts for current batch.
            batch_mean_validation_cost (float): Mean validation cost for current batch.
            inference_time (float): Inference time for one image.
        """
        num_steps = int(self._steps * dataset_percentage / 100.0)
        prev_start = default_timer()
        log_steps = 10
        for step in range(num_steps):
            start = default_timer()

            if (step % log_steps) == 0:
                logger.info("step %d / %d, %.2fs/step" %
                            (step, num_steps, (start-prev_start)/log_steps))
                prev_start = start
            batch_predictions, batch_validation_cost, batch_ground_truth_labels = \
                session.run(self.gridbox.get_validation_tensors() +
                            [self.ground_truth_labels])
            end = default_timer()

            batch_size = len(list(batch_predictions.values())[0]['bbox'])
            inference_time = (end - start) / batch_size

            batch_mean_validation_cost = batch_validation_cost / num_steps

            yield batch_predictions, batch_ground_truth_labels, batch_mean_validation_cost,\
                inference_time

    def evaluate(self, session, dataset_percentage=100.0):
        """Evaluate a DetectNet V2 model.

        Make predictions using the model and convert prediction and ground truth arrays to
        Detection and GroundTruth objects, respectively. Also, compile frame metadata for
        metrics computation, if available.

        Arguments:
            session (tf.Session): Session to be used for evaluation.
            dataset_percentage (float): % of the dataset to evaluate.

        Return:
            metrics_results: DetectionResults object from the metrics library.
            metrics_results_with_confidence: DetectionResults with confidence models applied.
            validation_cost: Validation cost value.
            inference_time: Median inference time for one image in seconds.
        """
        target_class_names = self._target_class_names
        # The ordering of these elements is used to determine the order in which to print them
        #  during print_metrics.
        clustered_detections = collections.OrderedDict(
            (target_class, []) for target_class in target_class_names)

        ground_truths = []
        frame_metadata = []
        inference_times = []
        validation_cost = 0.
        batch_count = 0
        for batch_predictions, batch_ground_truth_labels, batch_mean_validation_cost, \
            inference_time in self._get_validation_iterator(
                session, dataset_percentage):
            validation_cost += batch_mean_validation_cost
            inference_times.append(inference_time)
            # Append detections, ground truths and metadata from this batch to the data structures
            # passed to metrics computation.
            batch_detections = \
                self._postprocessor.cluster_predictions(
                    predictions=batch_predictions)

            for target_class in target_class_names:
                clustered_detections[target_class] += batch_detections[target_class]
            # Process groundtruth tensors to get GroundTruth objects and frame_metadata needed by
            # metrics.
            batch_ground_truth_objects, batch_metadata = \
                process_batch_ground_truth(
                    batch_ground_truth_labels, len(frame_metadata))
            ground_truths += batch_ground_truth_objects
            frame_metadata += batch_metadata
            batch_count += 1
        cdm = ComputeMetrics(clustered_detections, ground_truths,
                             self.gridbox.input_width,
                             self.gridbox.input_height,
                             self._target_class_names,
                             self._evaluation_config)
        metrics_results = cdm(num_recall_points=11, ignore_neutral_boxes=False)
        # Use median instead of average to be more robust against outliers.
        inference_time = np.median(inference_times)

        return metrics_results, validation_cost, inference_time

    @staticmethod
    def print_metrics(metrics_results, validation_cost, median_inference_time):
        """"Print a consolidated metrics table along with validation cost.

        Args:
            metrics_results (dict): a DetectionResults dict.
            validation_cost (float): calculated validation cost.
            median_inference_time (float): Median inference time.
        """
        print()
        print('Validation cost: %f' % validation_cost)
        print('Mean average_precision (in %): {:0.4f}'.format(metrics_results['mAP'] * 100.))
        print()
        headers = ['class name', 'average precision (in %)']
        # flip the code and name and sort
        data = sorted([(k, v * 100)
                       for k, v in list(metrics_results['average_precisions'].items())])
        print(tabulate(data, headers=headers, tablefmt="pretty"))
        # Flush to make the output look sequential.
        print('\nMedian Inference Time: %f' % median_inference_time)
        kpi_data = {
            "validation cost": round(validation_cost, 8),
            "mean average precision": round(metrics_results['mAP'] * 100, 4)
        }
        categorical_data = {
            "average_precision": {
                k: round(v * 100, 4) for k, v in list(
                    metrics_results['average_precisions'].items()
                    )
            }
        }
        s_logger = status_logging.get_status_logger()
        if isinstance(s_logger, status_logging.StatusLogger):
            s_logger.categorical = categorical_data
            s_logger.kpi = kpi_data
            s_logger.write(
                status_level=status_logging.Status.RUNNING,
                message="Evaluation metrics generated."
            )
        sys.stdout.flush()
