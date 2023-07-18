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
"""Test model builder components."""
import os
from google.protobuf.json_format import MessageToDict
import numpy as np
import pytest

import tensorflow as tf
from nvidia_tao_tf1.cv.mask_rcnn.models import fpn
from nvidia_tao_tf1.cv.mask_rcnn.models import heads
from nvidia_tao_tf1.cv.mask_rcnn.models import mask_rcnn_model
from nvidia_tao_tf1.cv.mask_rcnn.utils.spec_loader import load_experiment_spec


@pytest.mark.parametrize("size, min_level, max_level, filters",
                         [(512, 3, 8, 128),
                          (256, 3, 7, 256),
                          (256, 2, 6, 16)])
def test_fpn(size, min_level, max_level, filters):
    """Test FPN."""
    backbone_feats = {}
    bs = 4
    ww = hh = size
    for i in range(2, 6):
        random_feat = tf.constant(
            np.random.random((bs, 3, ww//2**(i-2), hh//2**(i-2))),
            dtype=tf.float32)
        backbone_feats[i] = random_feat

    fpn_network = fpn.FPNNetwork(min_level=min_level, max_level=max_level, filters=filters)
    output = fpn_network(backbone_feats)
    init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        output = sess.run(output)
        for j in sorted(output):
            assert output[j].shape[-1] == size // 2**(j-2)
            assert output[j].shape[1] == filters
    tf.keras.backend.clear_session()


@pytest.mark.parametrize("size, num_anchors, num_classes, mlp_dim, masksize",
                         [(128, 1000, 3, 1024, 7)])
def test_heads(size, num_anchors, num_classes, mlp_dim, masksize):
    """Test MRCNN heads."""
    backbone_feats = {}
    bs = 1
    filters = 16
    ww = hh = size
    num_rois = 123
    min_level = 3
    for i in range(2, 6):
        random_feat = tf.constant(
            np.random.random((bs, 3, ww//2**(i-2), hh//2**(i-2))),
            dtype=tf.float32)
        backbone_feats[i] = random_feat

    roi_feature = tf.constant(
        np.random.random((bs, num_rois, filters, ww, hh))
    )
    mask_feature = tf.constant(
        np.random.random((bs, num_rois, filters, masksize, masksize)),
        dtype=tf.float32
    )
    class_target = tf.constant(
        np.random.random((bs, num_rois)),
        dtype=tf.float32
    )
    fpn_network = fpn.FPNNetwork(min_level=min_level, max_level=7, filters=filters)
    rpn_head = heads.RPN_Head_Model(name="rpn_head",
                                    num_anchors=num_anchors,
                                    trainable=True,
                                    data_format='channels_first')
    box_head = heads.Box_Head_Model(
            num_classes=num_classes,
            mlp_head_dim=mlp_dim,
            trainable=True,
            name='box_head'
        )
    mask_head = heads.Mask_Head_Model(
            num_classes=num_classes,
            mrcnn_resolution=masksize,
            is_gpu_inference=True,
            data_format='channels_first',
            trainable=True,
            name="mask_head"
        )
    # forward pass
    output = fpn_network(backbone_feats)
    rpn_out = rpn_head(output[min_level])
    box_out = box_head(roi_feature)
    mask_out = mask_head(mask_feature, class_target)
    # init
    init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        scores, boxes = sess.run(rpn_out)
        assert scores.shape[-1] == num_anchors
        assert boxes.shape[-1] == num_anchors * 4
        assert scores.shape[1] == size // 2

        scores, boxes, _ = sess.run(box_out)
        assert boxes.shape[-1] == num_classes * 4
        assert scores.shape[-1] == num_classes
        assert scores.shape[1] == boxes.shape[1] == num_rois

        masks = sess.run(mask_out)
        assert masks.shape[1] == num_rois
        assert masks.shape[-1] == masksize
    tf.keras.backend.clear_session()


@pytest.mark.parametrize("size, bs, nlayers",
                         [(512, 8, 10),
                          (256, 4, 34),
                          (768, 2, 18),
                          (512, 2, 50),
                          (768, 1, 101)])
def test_build_model(size, bs, nlayers):
    """Test entire model."""
    # prepare data
    ww = hh = size
    features = {}
    features['images'] = tf.constant(
        np.random.random((bs, ww, hh, 3)),
        dtype=tf.float32
    )
    fake_info = size * np.ones((1, 5))
    fake_info[0, 2] = 1

    features['image_info'] = tf.constant(
        np.tile(fake_info, (bs, 1)),
        dtype=tf.float32
    )

    labels = {}
    file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_spec_path = os.path.join(file_path, '../experiment_specs/default.txt')
    experiment_spec = load_experiment_spec(default_spec_path, merge_from_default=False)
    default_params = MessageToDict(experiment_spec,
                                   preserving_proto_field_name=True,
                                   including_default_value_fields=True)
    try:
        data_config = default_params['data_config']
        maskrcnn_config = default_params['maskrcnn_config']
    except ValueError:
        print("Make sure data_config and maskrcnn_config are configured properly.")
    finally:
        del default_params['data_config']
        del default_params['maskrcnn_config']
    default_params.update(data_config)
    default_params.update(maskrcnn_config)
    default_params['aspect_ratios'] = eval(default_params['aspect_ratios'])
    default_params['freeze_blocks'] = eval(default_params['freeze_blocks'])
    default_params['bbox_reg_weights'] = eval(default_params['bbox_reg_weights'])
    default_params['image_size'] = (hh, ww)
    default_params['use_batched_nms'] = True
    default_params['nlayers'] = nlayers
    # build model
    output = mask_rcnn_model.build_model_graph(features, labels, False, default_params)
    init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        output = sess.run(output)
        assert len(output.keys()) == 5
    tf.keras.backend.clear_session()
