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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nvidia_tao_tf1.blocks.learning_rate_schedules.softstart_annealing_schedule import (
    SoftstartAnnealingLearningRateSchedule,
)


class TestSoftstartAnnealingSchedule(tf.test.TestCase):
    @tf.contrib.eager.run_test_in_graph_and_eager_modes
    def test_softstart_annealing_schedule(self):
        """
        Test SoftstartAnnealingLearningRateSchedule class.
        """
        base_learning_rate = 1.0
        min_learning_rate = 0.1
        last_step = 1000
        soft_start = 0.05
        annealing = 0.5
        expected_steps = [1, 200, 400, 600, 800, 999]
        expected_values = [0.10471285, 1.0, 1.0, 0.63095725, 0.25118864, 0.100461565]
        schedule = SoftstartAnnealingLearningRateSchedule(
            base_learning_rate,
            min_learning_rate,
            soft_start,
            annealing,
            last_step=last_step,
        )
        global_step_tensor = schedule.global_step
        if not tf.executing_eagerly():
            increment_global_step_op = tf.compat.v1.assign(
                global_step_tensor, global_step_tensor + 1
            )
            initializer = tf.compat.v1.global_variables_initializer()

        lr_tensor = schedule.learning_rate_tensor
        if not tf.executing_eagerly():
            self.evaluate(initializer)
        result_values = []
        for _ in range(last_step):
            if not tf.executing_eagerly():
                global_step = self.evaluate(increment_global_step_op)
            else:
                global_step = self.evaluate(global_step_tensor.assign_add(1))
            lr = self.evaluate(lr_tensor)
            if global_step in expected_steps:
                result_values.append(lr)
        self.assertAllClose(result_values, expected_values)
