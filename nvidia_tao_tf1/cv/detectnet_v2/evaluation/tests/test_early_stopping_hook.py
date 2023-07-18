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

"""Test the early stoping hook."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pytest
import tensorflow as tf

from nvidia_tao_tf1.core.distribution.distribution import Distributor, hvd
from nvidia_tao_tf1.cv.detectnet_v2.tfhooks.early_stopping_hook import (
    get_variable_softstart_annealing_learning_rate,
    LRAnnealingEarlyStoppingHook,
)


@pytest.fixture(autouse=True)
def reset_graph():
    tf.reset_default_graph()


class TestLRAnnealingEarlyStoppingHook:
    """Tests for LRAnnealingEarlyStoppingHook."""

    @pytest.fixture(scope="class", autouse=True)
    def set_up(self):
        """Need to initialize horovod once."""
        hvd().init()

    def get_stopping_hook(self, validation_period=1, first_validation_epoch=1, last_epoch=123,
                          steps_per_epoch=1, results_dir="results", num_validation_steps=2,
                          num_patience_steps=1, max_learning_rate=5e-4, min_learning_rate=5e-6,
                          num_soft_start_epochs=1, num_annealing_epochs=1):
        """Create an early stopping hook instance."""
        # Reset default graph to start fresh.
        validation_cost = tf.Variable(0.42)
        return LRAnnealingEarlyStoppingHook(
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
            validation_cost,
        )

    @pytest.mark.skipif(os.getenv("RUN_ON_CI", "0") == "1", reason="Cannot be run on CI")
    def test_validate_master(self, mocker):
        """Test that validate calls Evaluator.evaluate and tensorboard update."""

        stopping_hook = self.get_stopping_hook(validation_period=1, first_validation_epoch=1,
                                               steps_per_epoch=1, num_patience_steps=2,
                                               num_soft_start_epochs=1, num_annealing_epochs=1)

        # Make sure we are master.
        rank_mock = mocker.patch.object(Distributor, "is_master")
        rank_mock.return_value = True
        broadcast_op = mocker.patch.object(LRAnnealingEarlyStoppingHook, "broadcast_state")

        # Initialize session and hook.
        inc_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
        session = tf.train.SingularMonitoredSession(hooks=[stopping_hook])
        # Note: we use raw sessions and call the hook's validate method directly to be more
        # flexible. BaseValidationHook only calls the hook when
        # global_step_value + 1 % steps_per_epoch = 0, which also makes things confusing.
        raw_session = session.raw_session()
        run_context = tf.train.SessionRunContext(None, session.raw_session())

        # Step 0, check initialization.
        assert raw_session.run(stopping_hook._should_continue)
        assert not raw_session.run(stopping_hook._in_annealing_phase)

        # Check the broadcast op is being called and validation loss is computed.
        stopping_hook.validate(run_context)
        assert np.isclose(stopping_hook._min_cost, 0.42)
        broadcast_op.assert_called()

        # Step 1 + 2, check we are in soft start.
        # Adjusting annealing learning rate depends on steps,
        # hence step increase is needed in addition to epoch counter addition
        raw_session.run(inc_step)
        stopping_hook.epoch_counter += 1
        stopping_hook.validate(run_context)
        assert raw_session.run(stopping_hook._should_continue)
        assert not raw_session.run(stopping_hook._in_annealing_phase)
        raw_session.run(inc_step)
        stopping_hook.epoch_counter += 1
        stopping_hook.validate(run_context)
        assert raw_session.run(stopping_hook._should_continue)
        assert not raw_session.run(stopping_hook._in_annealing_phase)

        # Step 3, we initiate annealing.
        raw_session.run(inc_step)
        stopping_hook.epoch_counter += 1
        stopping_hook.validate(run_context)
        assert raw_session.run(stopping_hook._should_continue)
        assert raw_session.run(stopping_hook._in_annealing_phase)

        # Step 4 + 5, check we should continue and are in annealing.
        raw_session.run(inc_step)
        stopping_hook.epoch_counter += 1
        stopping_hook.validate(run_context)
        assert raw_session.run(stopping_hook._should_continue)
        assert raw_session.run(stopping_hook._in_annealing_phase)
        raw_session.run(inc_step)
        stopping_hook.epoch_counter += 1
        stopping_hook.validate(run_context)
        assert raw_session.run(stopping_hook._should_continue)
        assert raw_session.run(stopping_hook._in_annealing_phase)

        # Step 6, we're done.
        raw_session.run(inc_step)
        stopping_hook.epoch_counter += 1
        stopping_hook.validate(run_context)
        assert not raw_session.run(stopping_hook._should_continue)
        assert raw_session.run(stopping_hook._in_annealing_phase)
        assert run_context._stop_requested

    @pytest.mark.skipif(os.getenv("RUN_ON_CI", "0") == "1", reason="Cannot be run on CI")
    def test_hook_master(self, mocker):
        """Test that validate calls Evaluator.evaluate and tensorboard update."""

        stopping_hook = self.get_stopping_hook(validation_period=1, first_validation_epoch=1,
                                               steps_per_epoch=1, num_patience_steps=2,
                                               num_soft_start_epochs=1, num_annealing_epochs=1)

        # Make sure we are master.
        rank_mock = mocker.patch.object(Distributor, "is_master")
        rank_mock.return_value = True
        broadcast_op = mocker.patch.object(LRAnnealingEarlyStoppingHook, "broadcast_state")

        # Initialize session and hook.
        inc_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
        session = tf.train.SingularMonitoredSession(hooks=[stopping_hook])
        raw_session = session.raw_session()
        run_context = tf.train.SessionRunContext(None, session.raw_session())

        # Step 0, check initialization.
        assert raw_session.run(stopping_hook._should_continue)
        assert not raw_session.run(stopping_hook._in_annealing_phase)

        # Check the broadcast op is being called and validation loss is computed.
        session.run(inc_step)
        stopping_hook.validate(run_context)
        assert np.isclose(stopping_hook._min_cost, 0.42)
        broadcast_op.assert_called()

        # Step 1 + 2, check we are in soft start.
        # Adjusting annealing learning rate depends on steps,
        # hence step increase is needed in addition to epoch counter addition
        raw_session.run(inc_step)
        stopping_hook.epoch_counter += 1
        stopping_hook.validate(run_context)
        assert raw_session.run(stopping_hook._should_continue)
        assert not raw_session.run(stopping_hook._in_annealing_phase)
        raw_session.run(inc_step)
        stopping_hook.epoch_counter += 1
        stopping_hook.validate(run_context)
        assert raw_session.run(stopping_hook._should_continue)
        assert not raw_session.run(stopping_hook._in_annealing_phase)

        # Step 3, we initiate annealing.
        raw_session.run(inc_step)
        stopping_hook.epoch_counter += 1
        stopping_hook.validate(run_context)
        assert raw_session.run(stopping_hook._should_continue)
        assert raw_session.run(stopping_hook._in_annealing_phase)

        # Step 4 + 5, check we should continue and are in annealing.
        raw_session.run(inc_step)
        stopping_hook.epoch_counter += 1
        stopping_hook.validate(run_context)
        assert raw_session.run(stopping_hook._should_continue)
        assert raw_session.run(stopping_hook._in_annealing_phase)
        raw_session.run(inc_step)
        stopping_hook.epoch_counter += 1
        stopping_hook.validate(run_context)
        assert raw_session.run(stopping_hook._should_continue)
        assert raw_session.run(stopping_hook._in_annealing_phase)

        # Step 6, we're done.
        raw_session.run(inc_step)
        stopping_hook.epoch_counter += 1
        stopping_hook.validate(run_context)
        assert not raw_session.run(stopping_hook._should_continue)
        assert raw_session.run(stopping_hook._in_annealing_phase)
        assert run_context._stop_requested

    def test_validate_worker(self, mocker):
        """Test that validate calls Evaluator.evaluate and tensorboard update."""

        stopping_hook = self.get_stopping_hook()

        # Make sure we aren't master.
        rank_mock = mocker.patch.object(Distributor, "is_master")
        rank_mock.return_value = False
        broadcast_op = mocker.patch.object(LRAnnealingEarlyStoppingHook, "broadcast_state")
        validation_compute = mocker.patch.object(
            LRAnnealingEarlyStoppingHook, "_compute_validation_cost"
        )

        # Initialize session and hook.
        session = tf.train.SingularMonitoredSession(hooks=[stopping_hook])
        run_context = tf.train.SessionRunContext(None, session.raw_session())

        # Check initialization.
        assert session.run(stopping_hook._should_continue)
        assert not session.run(stopping_hook._in_annealing_phase)

        # Check only broadcast op is being called.
        stopping_hook.validate(run_context)
        broadcast_op.assert_called()
        validation_compute.assert_not_called()


@pytest.mark.parametrize(
    "soft_start_steps, plateau_steps, annealing_steps",
    [(10, 10, 10), (100, 100, 100), (0, 1, 1), (1, 0, 0), (0, 1, 0),
     (1, 1, 1), (40000, 10000, 40000)]
)
def test_variable_softstart_annealing_learning_rate(
    soft_start_steps, plateau_steps, annealing_steps, base_lr=0.1, min_lr=0.001
):
    """Test learning rates with different soft_start and annealing_steps values."""

    def expected_lr(step, soft_start_steps, annealing_steps, is_annealing):
        if is_annealing:
            if annealing_steps > 0:
                progress = 1 - float(step) / annealing_steps
            else:
                progress = 1.0
            progress = max(0.0, progress)
        else:
            if soft_start_steps > 0:
                progress = float(step) / soft_start_steps
            else:
                progress = 1.0
            progress = min(1.0, progress)
        lr = np.exp(np.log(min_lr) + progress * (np.log(base_lr) - np.log(min_lr)))
        return lr

    def computed_lr(step, soft_start_steps, annealing_steps, is_annealing):
        return get_variable_softstart_annealing_learning_rate(
            lr_step=step,
            soft_start_steps=soft_start_steps,
            annealing_steps=annealing_steps,
            start_annealing=is_annealing,
            base_lr=base_lr,
            min_lr=min_lr,
        )

    # Check across various phases and steps that computed is ~ expected lr.
    with tf.Session() as session:
        # Spread equally across all steps + some slack at the end.
        total_steps = soft_start_steps + plateau_steps + annealing_steps + 10
        steps = np.linspace(0, total_steps, 50, dtype=np.uint32)
        is_annealing = [step > soft_start_steps + plateau_steps for step in steps]
        tf_steps = [tf.constant(step) for step in steps]
        lrs = [
            computed_lr(step, soft_start_steps, annealing_steps, anneal)
            for step, anneal in zip(tf_steps, is_annealing)
        ]
        lrs = session.run(lrs)
        expected_lrs = [
            expected_lr(step, soft_start_steps, annealing_steps, anneal)
            for step, anneal in zip(steps, is_annealing)
        ]
        # Default relative tolerance of 1e-07 seems to be too small.
        np.testing.assert_allclose(lrs, expected_lrs, rtol=1e-06)
