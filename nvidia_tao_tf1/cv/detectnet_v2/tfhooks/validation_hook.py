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

"""A base class for a hook to compute model validation during training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from nvidia_tao_tf1.core.utils import summary_from_value
from nvidia_tao_tf1.cv.detectnet_v2.visualization.visualizer import \
    DetectNetTBVisualizer as Visualizer


class ValidationHook(tf.estimator.SessionRunHook):
    """ValidationHook to run evaluation for DetectNet V2 Model."""

    def __init__(self, evaluator, validation_period, last_epoch,
                 steps_per_epoch, results_dir, first_validation_epoch=0):
        """Create a hook object for validating a gridbox model during training.

        Args:
            evaluator: Evaluator object for running evaluation on a trained model.
            validation_period: How often (in epochs) the model is validated during training.
            last_epoch: Last epoch of training.
            steps_per_epoch: Number of steps per epoch.
            results_dir: Directory for logging the validation results.
            first_validation_epoch: The first validation epoch. Validation happens on epochs
                first_validation_epoch + i * validation_period, i=0, ...
        """
        self.evaluator = evaluator
        self.validation_period = validation_period
        self.last_epoch = last_epoch
        self.steps_per_epoch = steps_per_epoch
        self.steps_counter = 0
        self.epoch_counter = 0
        self.first_validation_epoch = first_validation_epoch
        self._global_step_tensor = tf.compat.v1.train.get_or_create_global_step()
        # Use an existing FileWriter.
        events_dir = os.path.join(
            results_dir, "events"
        )
        self.writer = tf.summary.FileWriterCache.get(events_dir)

    def before_run(self, run_context):
        """Request the value of global step.

        Args:
            run_context: A `SessionRunContext` object.
        Returns:
            A `SessionRunArgs` object.
        """
        return tf.estimator.SessionRunArgs(self._global_step_tensor)

    def _step(self, global_step_value):
        """Process one training step.

        Returns:
            Boolean indicating whether it's time to run validation.
        """
        # Global step is zero after the first step, but self.steps_counter
        # needs to be one for backward compatibility.
        self.steps_counter = global_step_value + 1

        # Validate only at the end of the epoch and not in between epochs.
        if self.steps_counter % self.steps_per_epoch != 0:
            return False

        # Calculate the current epoch.
        self.epoch_counter = int(self.steps_counter // self.steps_per_epoch)

        # Validate at every self.first_validation_epoch + i * self.validation_period epoch
        # and at the last epoch.
        is_validation_epoch = (self.epoch_counter >= self.first_validation_epoch) and \
            ((self.epoch_counter - self.first_validation_epoch) % self.validation_period == 0)
        return is_validation_epoch or self.epoch_counter == self.last_epoch

    def after_run(self, run_context, run_values):
        """Called after each call to run()."""
        run_validate = self._step(run_values.results)
        if run_validate is True:
            self.validate(run_context)

    def validate(self, run_context):
        """Called at the end of each epoch to validate the model."""
        # TODO(jrasanen) Optionally print metrics_results_with_confidence?
        metrics_result, validation_cost, median_inference_time = \
            self.evaluator.evaluate(run_context.session)

        print("Epoch %d/%d" % (self.epoch_counter, self.last_epoch))
        print('=========================')
        self.evaluator.print_metrics(metrics_result, validation_cost, median_inference_time)

        if Visualizer.enabled:
            self._add_to_tensorboard(metrics_result, validation_cost)

    def _add_to_tensorboard(self, metrics_result, validation_cost, bucket='mdrt'):
        """Add metrics to tensorboard."""
        summary = summary_from_value('validation_cost', validation_cost)
        self.writer.add_summary(summary, self.steps_counter)
        summary = summary_from_value(
            'mean average precision (mAP) (in %)',
            metrics_result['mAP']
        )
        self.writer.add_summary(summary, self.steps_counter)
        classwise_ap = metrics_result["average_precisions"]
        for class_name, ap in classwise_ap.items():
            tensor_name = f'{class_name}_AP (in %)'
            summary = summary_from_value(tensor_name, ap)
            self.writer.add_summary(summary, self.steps_counter)
