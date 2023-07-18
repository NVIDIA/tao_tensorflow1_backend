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

"""An early stopping hook that watches validation cost."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf

from nvidia_tao_tf1.core import distribution
from nvidia_tao_tf1.core.distribution.distribution import hvd
from nvidia_tao_tf1.core.utils import summary_from_value
from nvidia_tao_tf1.cv.detectnet_v2.tfhooks.validation_hook import ValidationHook


class LRAnnealingEarlyStoppingHook(ValidationHook):
    """Watch DetectNetv2 validation loss during training to stop training early.

    This integrates with the soft-start annealing learning rate schedule as follows.
    The learning rate is ramped up for num_soft_start_epochs from min_learning rate
    to max_learning rate. Then, validation loss is computed every validation_period epochs.
    If no improvement in loss is observable for num_patience_steps, the learning rate
    is annealed back to min_learning_rate over num_annealing_epochs. Then, the validation
    loss is monitored again, and after no improvement for num_patience_steps is observed,
    training is stopped.
    """

    def __init__(
        self,
        validation_period,
        last_epoch,
        steps_per_epoch,
        results_dir,
        first_validation_epoch,
        num_validation_steps,
        num_patience_steps,
        max_learning_rate,
        min_learning_rate,
        num_soft_start_epochs,
        num_annealing_epochs,
        validation_cost=None,
    ):
        """Create a hook object for validating DetectNetv2 during training.

        Args:
            validation_period: How often (in epochs) the model is validated during training.
            last_epoch: Last epoch of training.
            steps_per_epoch: Number of steps per epoch.
            results_dir: Directory for logging the validation results.
            first_validation_epoch: The first validation epoch. Validation happens on epochs
                first_validation_epoch + i * validation_period, i=0, ...
            num_validation_steps: Number of steps for a single validation run.
            num_patience_steps: Number of epochs we tolerate w/o validation loss improvement.
            max_learning_rate: Maximum learning rate in the soft-start-annealing learning rate
                schedule.
            max_learning_rate: Minimum learning rate in the soft-start-annealing learning rate
                schedule.
            num_soft_start_epochs: Number of epochs over which we soft-start the learning rate.
            num_annealing_epochs: Number of epochs over which we anneal the learning rate.
            validation_cost (Tensor): Validation cost tensor.
        """
        super(LRAnnealingEarlyStoppingHook, self).__init__(
            None,
            validation_period,
            last_epoch,
            steps_per_epoch,
            results_dir,
            first_validation_epoch,
        )
        if validation_period < 1:
            raise ValueError("Early stopping hook requires validation_period >= 1")
        if validation_period > num_patience_steps:
            raise ValueError(
                f"Validation period {validation_period} should be <= "
                f"Number of patience steps {num_patience_steps}"
            )
        if first_validation_epoch < 0:
            raise ValueError("Early stopping hook requires first_validation_epoch >= 0")
        if min_learning_rate <= 0.0:
            raise ValueError(
                "Early stopping min_learning_rate must be > 0"
            )
        if max_learning_rate <= 0.0:
            raise ValueError(
                "Early stopping max_learning_rate must be > 0"
            )
        if num_soft_start_epochs < 0.0:
            raise ValueError("Early stopping num_soft_start_epochs must be >= 0")
        if num_annealing_epochs < 0.0:
            raise ValueError(
                "Early stopping num_annealing_epochs must be >= 0"
            )
        if num_patience_steps > last_epoch:
            raise ValueError(
                f"Number of patience steps {num_patience_steps} "
                f"> last_epoch {last_epoch}"
            )
        self.num_validation_steps = num_validation_steps
        self.num_patience_steps = num_patience_steps
        self.validation_cost = validation_cost
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.soft_start_steps = int(num_soft_start_epochs * steps_per_epoch)
        self.annealing_steps = int(num_annealing_epochs * steps_per_epoch)
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self._session = None
        # Learning rate variable.
        self.learning_rate = None
        # Smallest cost so far.
        self._min_cost = None
        # Epoch when we observed smallest cost.
        self._min_cost_epoch = None
        # Kill-flag to request stop training.
        self._should_continue = None
        # Starting step for current phase (soft-start, or anneal).
        self._lr_phase_start_step = None
        # Op to set the lr_phase_start_step to the current step.
        self._set_lr_phase_start_step_op = None
        # Step inside current phase.
        self._lr_phase_step = None
        # Flag indicating whether we are inside the annealing phase.
        self._in_annealing_phase = None
        # Op to broadcast state to workers.
        self._broadcast_state_op = None
        # Op to set flag for stopping training.
        self._set_request_stop_op = None
        # Op to set start_annealing.
        self._set_start_annealing_op = None
        # Initialize the variables above.
        self._make_control_variables()
        logging.info(
            (
                "Early stopping: first val. epoch {} , {} validation steps, {} patience steps, "
                "{} soft-start steps, {} annealing steps, {} max_learning_rate, "
                "{} min_learning_rate"
            ).format(
                first_validation_epoch,
                self.num_validation_steps,
                self.num_patience_steps,
                self.soft_start_steps,
                self.annealing_steps,
                self.max_learning_rate,
                self.min_learning_rate,
            )
        )

    def _make_control_variables(self):
        """Initialize internal TF control variables."""
        with tf.compat.v1.name_scope("EarlyStopping"):
            self._should_continue = tf.Variable(True, name="should_continue")
            self._lr_phase_start_step = tf.Variable(0, dtype=tf.int64, name="lr_phase_start_step")
            self._lr_phase_step = tf.cast(
                tf.compat.v1.train.get_or_create_global_step() - self._lr_phase_start_step,
                tf.float32
            )
            self._in_annealing_phase = tf.Variable(False, name="in_annealing_phase")
            self._broadcast_state_op = tf.group(
                self._should_continue.assign(
                    hvd().broadcast(
                        self._should_continue,
                        distribution.get_distributor()._master_rank,
                    )
                ),
                self._in_annealing_phase.assign(
                    hvd().broadcast(
                        self._in_annealing_phase,
                        distribution.get_distributor()._master_rank,
                    )
                ),
                self._lr_phase_start_step.assign(
                    hvd().broadcast(
                        self._lr_phase_start_step,
                        distribution.get_distributor()._master_rank,
                    )
                ),
            )
            self._set_request_stop_op = self._should_continue.assign(False)
            self.learning_rate = get_variable_softstart_annealing_learning_rate(
                self._lr_phase_step,
                self.soft_start_steps,
                self.annealing_steps,
                self._in_annealing_phase,
                self.max_learning_rate,
                self.min_learning_rate,
            )
            self._set_lr_phase_start_step_op = self._lr_phase_start_step.assign(
                tf.compat.v1.train.get_or_create_global_step()
            )
            self._set_start_annealing_op = self._in_annealing_phase.assign(True)

    def _start_annealing(self):
        """Helper function to initiate annealing phase."""
        self._session.run([self._set_lr_phase_start_step_op, self._set_start_annealing_op])

    def after_create_session(self, session, coord):
        """Store session for later use."""
        self._session = session

    def broadcast_state(self):
        """Broadcast current state."""
        self._session.run(self._broadcast_state_op)

    def _compute_validation_cost(self):
        """Compute total validation cost using current session."""
        total_cost = 0
        for _ in range(self.num_validation_steps):
            total_cost += self._session.run(self.validation_cost)
        return total_cost / self.num_validation_steps

    def _validate_master(self, run_context):
        """Run validation on master."""
        current_epoch = self.epoch_counter
        logging.info(
            "Validation at epoch {}/{}".format(self.epoch_counter, self.last_epoch)
        )
        logging.info(
            "Running {} steps to compute validation cost".format(
                self.num_validation_steps
            )
        )
        validation_cost = self._compute_validation_cost()
        logging.info(
            "Validation cost {} at epoch {}".format(validation_cost, current_epoch)
        )
        # Loss decreased.
        if self._min_cost is None or self._min_cost > validation_cost:
            self._min_cost = validation_cost
            self._min_cost_epoch = current_epoch
            logging.info(
                "New best validation cost {} at epoch {}".format(
                    validation_cost, current_epoch
                )
            )
        # Loss did not decrease and we exceeded patience.
        elif current_epoch - self._min_cost_epoch >= self.num_patience_steps:
            logging.info(
                "Validation cost did not improve for {} epochs, which is >= "
                "num_patience_steps {}.".format(
                    current_epoch - self._min_cost_epoch,
                    self.num_patience_steps
                )
            )
            logging.info(
                "Best cost {} at epoch {}. Current epoch {}".format(
                    self._min_cost, self._min_cost_epoch, current_epoch
                )
            )
            annealing_started = self._session.run(self._in_annealing_phase)
            annealing_finished = (
                annealing_started
                and self._session.run(self._lr_phase_step) > self.annealing_steps
            )
            # If we are after annealing phase, stop training.
            if annealing_started and annealing_finished:
                logging.info("Requesting to stop training.")
                self._session.run(self._set_request_stop_op)
            # If we are before annealing phase, start annealing.
            elif not annealing_started:
                logging.info(
                    "Starting to anneal learning rate. Setting new best validation cost to current."
                )
                self._start_annealing()
                self._min_cost = validation_cost
                self._min_cost_epoch = current_epoch
        else:
            logging.info(
                "Last best validation cost {} at epoch {}".format(
                    self._min_cost, self._min_cost_epoch
                )
            )
        summary = summary_from_value("validation_cost", validation_cost)
        self.writer.add_summary(summary, current_epoch)

    def validate(self, run_context):
        """Called at the end of each epoch to validate the model."""
        if distribution.get_distributor().is_master():
            self._validate_master(run_context)
        # Broadcast new state.
        self.broadcast_state()
        if not self._session.run(self._should_continue):
            logging.info("Requested to stop training.")
            run_context.request_stop()


def get_variable_softstart_annealing_learning_rate(
    lr_step, soft_start_steps, annealing_steps, start_annealing, base_lr, min_lr
):
    """Return learning rate at current epoch progress.

    When start_annealing is False, ramp up learning rate from min_lr to base_lr on a logarithmic
    scale. After soft_start_steps learning rate will reach base_lr and be kept there until
    start_annealing becomes True. Then, learning rate is decreased from base_lr to min_lr,
    again on a logarithmic scale until it reaches min_lr, where it is kept for the rest
    of training.

    Note: start_annealing should not be set to True before soft_star_steps of warming up to
        base_lr, since the annealing phase will always start at base_lr.


    Args:
        lr_step (tf.Variable): Step number inside the current phase (soft-start, or annealing).
        soft_start_steps (int): Number of soft-start steps.
        annealing_steps (int): Number of annealing steps.
        start_annealing (tf.Variable): Boolean variable indicating whether we are in
            soft-start phase (False) or annealing phase (True).
        base_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.

    Returns:
        lr: A tensor (scalar float) indicating the learning rate.
    """
    # Need this as float32.
    lr_step = tf.cast(lr_step, tf.float32)

    # Ratio in soft-start phase, going from 0 to 1.
    if soft_start_steps > 0:
        t_softstart = lr_step / soft_start_steps
    else:  # Learning rate starts from base_lr.
        t_softstart = tf.constant(1.0, dtype=tf.float32)

    if annealing_steps > 0:
        # Ratio in annealing phase, going from 1 to 0.
        t_annealing = 1.0 - lr_step / annealing_steps
    else:  # Learning rate is never annealed.
        t_annealing = tf.constant(1.0, dtype=tf.float32)

    # Ratio is at least 0, even if we do more thatn annealing_steps.
    t_annealing = tf.compat.v1.where(
        t_annealing < 0.0, tf.constant(0.0, dtype=tf.float32), t_annealing
    )

    # Select appropriate schedule.
    t = tf.compat.v1.where(start_annealing, t_annealing, t_softstart)

    # Limit ratio to max 1.0.
    t = tf.compat.v1.where(t > 1.0, tf.constant(1.0, dtype=tf.float32), t)

    # Adapt learning rate linearly on log scale between min_lr and base_lr.
    lr = tf.exp(tf.math.log(min_lr) + t * (tf.math.log(base_lr) - tf.math.log(min_lr)))
    return tf.cast(lr, tf.float32)


def build_early_stopping_hook(
    evaluation_config,
    steps_per_epoch,
    results_dir,
    num_validation_steps,
    experiment_spec,
    validation_cost
):
    """Builder function to create early stopping hook.

    Args:
        evaluation_config (nvidia_tao_tf1.cv.detectnet_v2.evaluation.EvaluationConfig):
            Configuration for evaluation.
        steps_per_epoch (int): Total number of training steps per epoch.
        results_dir (str): Where to store results and write TensorBoard summaries.
        num_validation_steps (int): Number of steps needed for validation.
        experiment_spec (nvidia_tao_tf1.cv.detectnet_v2.proto.experiment_pb2):
            Experiment spec message.
        validation_cost (Tensor): Validation cost tensor. Can be
            None for workers, since validation cost is only computed on master.

    Returns:
        learning_rate: Learning rate schedule created.
    """
    learning_rate_config = experiment_spec.training_config.learning_rate
    if not learning_rate_config.HasField("early_stopping_annealing_schedule"):
        raise ValueError("Early stopping hook is missing "
                         "learning_rate_config.early_stopping_annealing_schedule")
    params = learning_rate_config.early_stopping_annealing_schedule
    num_epochs = experiment_spec.training_config.num_epochs
    return LRAnnealingEarlyStoppingHook(
        validation_period=evaluation_config.validation_period_during_training,
        last_epoch=num_epochs,
        steps_per_epoch=steps_per_epoch,
        results_dir=results_dir,
        first_validation_epoch=evaluation_config.first_validation_epoch,
        num_validation_steps=num_validation_steps,
        num_patience_steps=params.patience_steps,
        max_learning_rate=params.max_learning_rate,
        min_learning_rate=params.min_learning_rate,
        num_soft_start_epochs=params.soft_start_epochs,
        num_annealing_epochs=params.annealing_epochs,
        validation_cost=validation_cost
    )
