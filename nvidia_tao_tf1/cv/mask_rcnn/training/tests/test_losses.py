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
"""Test preprocess ops."""
import numpy as np
import tensorflow as tf
from nvidia_tao_tf1.cv.mask_rcnn.training import losses


def test_rpn_score_loss():
    """Test RPN score losses."""
    y_pd = [[[0.1, 0.2],
             [0.3, 0.4],
             [0.5, 0.6],
             [0.7, 0.8],
             [0.9, 0.1]]]
    y_gt = [[[1, 0],
             [0, 1],
             [0, 0],
             [1, 1],
             [-1, -1]]]
    # (1, 5, 2, 1)
    y_pd = tf.expand_dims(tf.constant(y_pd), axis=-1)
    y_gt = tf.expand_dims(tf.constant(y_gt), axis=-1)
    loss = losses._rpn_score_loss(y_pd, y_gt, 1.0)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        result = sess.run(loss)
        assert np.allclose([result], [5.5957575])


def test_rpn_box_loss():
    """Test RPN box loss."""
    y_pd = [[[0.1, 0.2],
             [0.3, 0.4],
             [0.5, 0.6],
             [0.7, 0.8],
             [0.9, 0.1]]]
    y_gt = [[[1, 0],
             [0, 1],
             [0, 0],
             [1, 1],
             [-1, -1]]]
    # (1, 5, 2, 1)
    y_pd = tf.expand_dims(tf.constant(y_pd, dtype=tf.float32), axis=-1)
    y_pd = tf.stack([y_pd, y_pd, y_pd, y_pd], axis=3)[..., 0]
    y_gt = tf.expand_dims(tf.constant(y_gt, dtype=tf.float32), axis=-1)
    y_gt = tf.stack([y_gt, y_gt, y_gt, y_gt], axis=3)[..., 0]
    loss = losses._rpn_box_loss(y_pd, y_gt, 1.0)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        result = sess.run(loss)
        assert np.allclose([result], [0.086419754])


def test_frcnn_class_loss():
    """Test FRCNN class loss."""
    x = np.random.random((1, 10))
    y = np.array(x == np.max(x)).astype(np.uint8)
    X = tf.constant(x, dtype=tf.float32)
    Y = tf.constant(y, dtype=tf.float32)
    ce_loss = losses._fast_rcnn_class_loss(X, Y)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        result = sess.run(ce_loss)
        print(result)
    expected = cross_entropy_loss(softmax(x), y)
    assert np.isclose(expected, result, atol=1e-5)


def test_frcnn_box_loss():
    """Test FRCNN box loss."""
    y_pd = [[[0.1, 0.2, 0.3, 0.4],
             [0.3, 0.4, 0.5, 0.6],
             [0.5, 0.6, 0.6, 0.7],
             [0.7, 0.8, 0.8, 0.9],
             [0.9, 0.1, 1.0, 0.2]]]
    y_gt = [[[0.1, 0.2, 0.35, 0.45],
             [0.3, 0.4, 0.55, 0.65],
             [0.5, 0.65, 0.65, 0.7],
             [0.7, 0.8, 0.85, 0.9],
             [0.95, 0.1, 1.0, 0.25]]]
    bbox_pred = tf.constant(y_pd, dtype=tf.float32)
    bbox_target = tf.constant(y_gt, dtype=tf.float32)
    bbox_class = tf.constant(np.ones((1, 5)), dtype=tf.float32)
    output = losses._fast_rcnn_box_loss(bbox_pred, bbox_target, bbox_class)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        result = sess.run(output)
        print(result)
    assert np.isclose(result, 0.0005624996)


def test_mrcnn_loss():
    """Test mask loss."""
    params = {}
    params['mrcnn_weight_loss_mask'] = 1
    mask_pd = np.array(
        [[[[0.43243138, 0.10941903, 0.5394564, 0.03027718, 0.09385015],
           [0.46510113, 0.31793583, 0.12436913, 0.85575253, 0.21725659],
           [0.84771655, 0.26786283, 0.19578418, 0.47323207, 0.17927243],
           [0.14441685, 0.55496574, 0.46789852, 0.33832675, 0.87777560],
           [0.09468317, 0.51682021, 0.46277403, 0.46175286, 0.97316929]],
          [[0.10863443, 0.27351846, 0.77452679, 0.47604643, 0.33915814],
           [0.51387640, 0.68092501, 0.46150238, 0.18415834, 0.53534979],
           [0.31629868, 0.64107154, 0.68567363, 0.72082573, 0.14127229],
           [0.52808330, 0.92486021, 0.13708679, 0.57605909, 0.52032435],
           [0.13153844, 0.86583202, 0.82361283, 0.17344127, 0.86495139]],
          [[0.60915031, 0.18700866, 0.80593272, 0.38373346, 0.36316737],
           [0.42034724, 0.90357802, 0.60602034, 0.13499481, 0.58061098],
           [0.71707536, 0.29609169, 0.88630556, 0.07664849, 0.40421899],
           [0.87128055, 0.74032759, 0.11390369, 0.48023603, 0.69994274],
           [0.99873194, 0.86009772, 0.03589585, 0.34378647, 0.89354507]],
          [[0.10663731, 0.94849647, 0.73884596, 0.88823689, 0.14141050],
           [0.42600678, 0.06402352, 0.36755309, 0.77146811, 0.67770853],
           [0.30417758, 0.57815378, 0.97985004, 0.30534067, 0.61657323],
           [0.06604396, 0.13499227, 0.78873070, 0.15410758, 0.57401998],
           [0.11121709, 0.35525001, 0.65603656, 0.54746670, 0.48069364]]]])
    mask_gt = np.ones((1, 4, 5, 5))
    mask_select = np.ones((1, 4))
    mask_pd = tf.constant(mask_pd, dtype=tf.float32)
    mask_gt = tf.constant(mask_gt, dtype=tf.float32)
    mask_select = tf.constant(mask_select, dtype=tf.float32)
    output = losses.mask_rcnn_loss(mask_pd, mask_gt, mask_select, params)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        result = sess.run(output)
    assert np.isclose(result, 0.4921867)


def cross_entropy_loss(pred, gt, epsilon=1e-12):
    """"Helper func for cross entropy loss."""
    pred = np.clip(pred, epsilon, 1 - epsilon)
    return -np.sum(gt * np.log(pred))


def softmax(x):
    """"Helper func for softmax."""
    return np.exp(x) / np.sum(np.exp(x))
