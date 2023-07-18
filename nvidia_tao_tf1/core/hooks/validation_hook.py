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

import logging

import os

import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunArgs

logger = logging.getLogger(__name__)


class ValidationHook(tf.estimator.SessionRunHook):
    """Hook to perform validation during training.

    Given an Evaluator and validation_every_n_steps, This hook performs validation after
    every n steps.
    """

    def __init__(self, evaluator, validation_every_n_steps):
        """Initialize the hook.

        Args:
            evaluator (Evaluator or list): Object or list of objects that performs evaluation
                and returns metrics.
            validation_every_n_steps (int): Perform validation every n steps.
        """
        if not isinstance(evaluator, list):
            evaluator = [evaluator]
        for evaluator_object in evaluator:
            evaluate_func = getattr(evaluator_object, "evaluate", None)
            if not callable(evaluate_func):
                raise ValueError(
                    "Evaluator {} does not have callable evaluate function!".format(
                        evaluator_object
                    )
                )
        self.n_evaluators = len(evaluator)
        self._evaluators = evaluator
        self._validation_every_n_steps = validation_every_n_steps
        self._global_step_tensor = tf.compat.v1.train.get_or_create_global_step()

    def before_run(self, run_context):
        """Called before each call to run().

        Run the ops each run.

        Args:
            run_context: A `SessionRunContext` object.
        Returns:
            A `SessionRunArgs` object.
        """
        return SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        """Called after each call to run().

        Run Validation after each call to  run.

        Args:
            run_context: A `SessionRunContext` object.
            run_values: A `SessionRunValues` object.
        """
        self.global_step = run_values.results
        if self.global_step % self._validation_every_n_steps == 0:
            for evaluator in self._evaluators:
                self.validation_metrics = evaluator.evaluate(
                    sess=self._raw_session, global_step=self.global_step
                )
                # print metrics only if something valid is returned
                if self.validation_metrics:
                    logger.info(
                        "Validation #{}: {}".format(
                            self.global_step, self.validation_metrics
                        )
                    )

    def after_create_session(self, session, coord):
        """Called when new TensorFlow session is created.

        Get raw session for this hook.

        Args:
            session: A TensorFlow Session that has been created.
            coord: A Coordinator object which keeps track of all threads.
        """
        self._raw_session = session