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

'''build model for evaluation.'''

import tensorflow as tf


def build(model):
    '''Build model for evaluation.'''

    prob = model.layers[-1].output

    sequence_id = tf.keras.backend.argmax(prob, axis=-1)

    sequence_prob = tf.keras.backend.max(prob, axis=-1)

    eval_model = tf.keras.Model(inputs=[model.input],
                                outputs=[sequence_id, sequence_prob])

    return eval_model
