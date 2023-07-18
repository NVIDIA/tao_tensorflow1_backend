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

'''Utility functions used by unit tests for FasterRCNN.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def _take_first_k(arr, k, *args, **kargs):
    '''Take the first k elements.'''
    return np.arange(k)


def _fake_uniform(shape, maxval, dtype, seed):
    '''A function that monkey patches tf.random.uniform to make it deterministic in test.'''
    enough = (maxval >= shape[0])
    ret_enough = tf.where(tf.ones(shape))[:, 0]
    r = tf.cast(tf.ceil(shape[0] / maxval), tf.int32)
    ret_segment = tf.where(tf.ones([maxval]))[:, 0]
    ret_not_enough = tf.tile(ret_segment, [r])[:tf.cast(shape[0], tf.int32)]
    return tf.cond(enough,
                   true_fn=lambda: ret_enough,
                   false_fn=lambda: ret_not_enough)


def _fake_choice(arr, num, replace, *args):
    '''A function that monkey patches np.random.choice.'''
    if arr.size >= num:
        return arr[:num]
    r = (num + arr.size - 1) // arr.size
    a = np.tile(arr, r)
    return a[:num]


def _fake_exp(t):
    '''A fake exp function to replace tf.exp for monkey patch.'''
    def _exp_np(x):
        return np.exp(x)
    return tf.py_func(_exp_np, [t], tf.float32)
