# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Model definition for the Mask-RCNN Model.

Defines model_fn of Mask-RCNN for TF Estimator. The model_fn includes Mask-RCNN
model architecture, loss function, learning rate schedule, and evaluation
procedure.

"""
import itertools
import os
import tensorflow as tf
from tensorflow.python.util.tf_export import keras_export

from nvidia_tao_tf1.core.templates.resnet_tf import ResNet
from nvidia_tao_tf1.cv.mask_rcnn.layers.anchor_layer import AnchorLayer
from nvidia_tao_tf1.cv.mask_rcnn.layers.box_input_layer import BoxInput
from nvidia_tao_tf1.cv.mask_rcnn.layers.box_target_encoder import BoxTargetEncoder
from nvidia_tao_tf1.cv.mask_rcnn.layers.class_input_layer import ClassInput
from nvidia_tao_tf1.cv.mask_rcnn.layers.foreground_selector_for_mask import \
    ForegroundSelectorForMask
from nvidia_tao_tf1.cv.mask_rcnn.layers.gpu_detection_layer import GPUDetections
from nvidia_tao_tf1.cv.mask_rcnn.layers.image_input_layer import ImageInput
from nvidia_tao_tf1.cv.mask_rcnn.layers.info_input_layer import InfoInput
from nvidia_tao_tf1.cv.mask_rcnn.layers.mask_input_layer import MaskInput
from nvidia_tao_tf1.cv.mask_rcnn.layers.mask_targets_layer import MaskTargetsLayer
from nvidia_tao_tf1.cv.mask_rcnn.layers.multilevel_crop_resize_layer import MultilevelCropResize
from nvidia_tao_tf1.cv.mask_rcnn.layers.multilevel_proposal_layer import MultilevelProposal
from nvidia_tao_tf1.cv.mask_rcnn.layers.proposal_assignment_layer import ProposalAssignment

from nvidia_tao_tf1.cv.mask_rcnn.models import fpn
from nvidia_tao_tf1.cv.mask_rcnn.models import heads
from nvidia_tao_tf1.cv.mask_rcnn.training import learning_rates, losses

from nvidia_tao_tf1.cv.mask_rcnn.utils import model_loader
from nvidia_tao_tf1.cv.mask_rcnn.utils.distributed_utils import MPI_is_distributed
from nvidia_tao_tf1.cv.mask_rcnn.utils.distributed_utils import MPI_local_rank
from nvidia_tao_tf1.cv.mask_rcnn.utils.lazy_imports import LazyImport
from nvidia_tao_tf1.cv.mask_rcnn.utils.logging_formatter import logging

hvd = LazyImport("horovod.tensorflow")

MODELS = dict()
iva_bb_dict = {
    'resnet10': ('block_1a_relu', 'block_2a_relu', 'block_3a_relu', 'block_4a_relu'),
    'resnet18': ('block_1b_relu', 'block_2b_relu', 'block_3b_relu', 'block_4b_relu'),
    'resnet34': ('block_1c_relu', 'block_2d_relu', 'block_3f_relu', 'block_4c_relu'),
    'resnet50': ('block_1c_relu', 'block_2d_relu', 'block_3f_relu', 'block_4c_relu'),
    'resnet101': ('block_1c_relu', 'block_2d_relu', 'block_3f_relu', 'block_4c_relu')}


@keras_export('keras.Input', 'keras.layers.Input')
def Input(shape=None,  # pylint: disable=invalid-name
          batch_size=None,
          name=None,
          dtype=None,
          sparse=False,
          tensor=None,
          ragged=False,
          layer_class=tf.keras.layers.InputLayer,
          **kwargs):
    """Patch to instantiate a Keras tensor.

    A Keras tensor is a TensorFlow symbolic tensor object,
    which we augment with certain attributes that allow us to build a Keras model
    just by knowing the inputs and outputs of the model.
    """
    if sparse and ragged:
        raise ValueError(
            'Cannot set both sparse and ragged to True in a Keras input.')

    input_layer_config = {'name': name, 'dtype': dtype, 'sparse': sparse,
                          'ragged': ragged, 'input_tensor': tensor}

    batch_input_shape = kwargs.pop('batch_input_shape',
                                   kwargs.pop('batch_shape', None))
    if shape and batch_input_shape:
        raise ValueError('Only provide the `shape` OR `batch_input_shape` argument '
                         'to Input, not both at the same time.')
    if batch_input_shape:
        shape = batch_input_shape[1:]
        input_layer_config.update({'batch_input_shape': batch_input_shape})
    else:
        input_layer_config.update(
            {'batch_size': batch_size, 'input_shape': shape})

    if kwargs:
        raise ValueError('Unrecognized keyword arguments:', kwargs.keys())

    if shape is None and tensor is None:
        raise ValueError('Please provide to Input either a `shape`'
                         ' or a `tensor` argument. Note that '
                         '`shape` does not include the batch '
                         'dimension.')

    input_layer = layer_class(**input_layer_config)

    # Return tensor including `_keras_history`.
    # Note that in this case train_output and test_output are the same pointer.
    outputs = input_layer._inbound_nodes[0].output_tensors
    if len(outputs) == 1:
        return outputs[0]
    return outputs


def create_optimizer(learning_rate, params):
    """Creates optimized based on the specified flags."""

    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum=params['momentum'])

    if MPI_is_distributed():
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            name=None,
            device_dense='/gpu:0',
            device_sparse='',
            # compression=hvd.Compression.fp16,
            compression=hvd.Compression.none,
            sparse_as_dense=False
        )

    if params["use_amp"]:
        loss_scale = tf.train.experimental.DynamicLossScale(
            initial_loss_scale=(2 ** 15),
            increment_period=2000,
            multiplier=2.0
        )
        optimizer = tf.compat.v1.train.experimental.MixedPrecisionLossScaleOptimizer(
            optimizer,
            loss_scale=loss_scale)

    return optimizer


def compute_model_statistics(batch_size, is_training=True):
    """Compute number of parameters and FLOPS."""
    options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    options['output'] = 'none'

    from tensorflow.python.keras.backend import get_graph
    flops = tf.compat.v1.profiler.profile(get_graph(), options=options).total_float_ops
    flops_per_image = flops / batch_size

    logging.info('[%s Compute Statistics] %.1f GFLOPS/image' % (
        "Training" if is_training else "Inference",
        flops_per_image/1e9
    ))


def build_model_graph(features, labels, is_training, params, data_format='channels_first'):
    """Builds the forward model graph."""

    image_info_shape = features['image_info'].get_shape().as_list()
    if params.get('pruned_model_path', ''):
        logging.info("***********************")
        logging.info("Loading model graph...")
        logging.info("***********************")
        features['images'] = tf.transpose(features['images'], [0, 3, 1, 2],
                                          name="input_image_transpose")
        batch_size, _, image_height, image_width = features['images'].get_shape().as_list()

        if is_training:
            gt_boxes_shape = labels['gt_boxes'].get_shape().as_list()
            gt_classes_shape = labels['gt_classes'].get_shape().as_list()
            gt_masks_shape = labels['cropped_gt_masks'].get_shape().as_list()

            pruned_model = model_loader.get_model_with_input(
                os.path.join(params['pruned_model_path'], 'train_graph.json'),
                ImageInput(input_tensor=features['images'],
                           input_shape=(3, image_height, image_width)),
                InfoInput(input_tensor=features['image_info'], input_shape=image_info_shape[1:]),
                BoxInput(input_tensor=labels['gt_boxes'], input_shape=gt_boxes_shape[1:]),
                ClassInput(input_tensor=labels['gt_classes'], input_shape=gt_classes_shape[1:]),
                MaskInput(input_tensor=labels['cropped_gt_masks'], input_shape=gt_masks_shape[1:])
                )
            # dump model json
            if MPI_local_rank() == 0 and params.get('model_dir', ''):
                model_loader.dump_json(
                    pruned_model,
                    os.path.join(params['model_dir'], "train_graph.json"))
            # format model outputs
            model_outputs = pruned_model.outputs
            mo_formatted = []
            mo_formatted.append(dict(zip(list(range(2, 7)), model_outputs[0:5])))
            mo_formatted.append(dict(zip(list(range(2, 7)), model_outputs[5:10])))
            mo_formatted.append(model_outputs[10])
            mo_formatted.append(model_outputs[11])
            mo_formatted.append(model_outputs[12])
            mo_formatted.append(model_outputs[13])
            mo_formatted.append(model_outputs[14])
            mo_formatted.append(model_outputs[15])
            mo_formatted.append(model_outputs[16])
            mo_formatted.append(model_outputs[17])
            model_outputs_keys = ['rpn_score_outputs', 'rpn_box_outputs',
                                  'class_outputs', 'box_outputs', 'class_targets',
                                  'box_targets', 'box_rois', 'mask_outputs',
                                  'mask_targets', 'selected_class_targets']
            model_outputs = dict(zip(model_outputs_keys, mo_formatted))
        else:
            # patch exporter for pruned model
            def compose_call(prev_call_method):
                def call(self, inputs, training=False):
                    return prev_call_method(self, inputs, training)
                return call

            prev_batchnorm_call = tf.keras.layers.BatchNormalization.call
            tf.keras.layers.BatchNormalization.call = compose_call(
                prev_batchnorm_call
            )
            pruned_model = model_loader.get_model_with_input(
                os.path.join(params['pruned_model_path'], 'eval_graph.json'),
                ImageInput(input_tensor=features['images'],
                           input_shape=(3, image_height, image_width)),
                InfoInput(input_tensor=features['image_info'], input_shape=image_info_shape[1:]),
                None, None, None
                )
            if MPI_local_rank() == 0 and params.get('model_dir', ''):
                model_loader.dump_json(
                    pruned_model,
                    os.path.join(params['model_dir'], "eval_graph.json"))
            model_outputs = pruned_model.outputs
            mo_formatted = []
            mo_formatted.append(model_outputs[0])
            mo_formatted.append(model_outputs[1])
            mo_formatted.append(model_outputs[2])
            mo_formatted.append(model_outputs[3])
            mo_formatted.append(model_outputs[-1])
            model_outputs_keys = ['num_detections',
                                  'detection_boxes',
                                  'detection_classes',
                                  'detection_scores',
                                  'detection_masks']
            model_outputs = dict(zip(model_outputs_keys, mo_formatted))

        logging.debug(pruned_model.summary())
        return model_outputs

    logging.info("***********************")
    logging.info("Building model graph...")
    logging.info("***********************")

    if is_training:
        gt_boxes_shape = labels['gt_boxes'].get_shape().as_list()
        gt_classes_shape = labels['gt_classes'].get_shape().as_list()
        gt_masks_shape = labels['cropped_gt_masks'].get_shape().as_list()
    model_outputs = {}
    is_gpu_inference = not is_training  # and params['use_batched_nms']

    if data_format == "channels_last":
        batch_size, image_height, image_width, _ = features['images'].get_shape().as_list()
    elif data_format == "channels_first":
        features['images'] = tf.transpose(features['images'], [0, 3, 1, 2],
                                          name="input_image_transpose")
        batch_size, _, image_height, image_width = features['images'].get_shape().as_list()
    else:
        raise ValueError("data format not recognized: %s" % data_format)

    if 'source_ids' not in features:
        features['source_ids'] = -1 * tf.ones([batch_size], dtype=tf.float32)

    inputs = Input(tensor=features['images'],
                   shape=(3, image_height, image_width),
                   layer_class=ImageInput)
    an_layer = AnchorLayer(params['min_level'], params['max_level'],
                           params['num_scales'], params['aspect_ratios'],
                           params['anchor_scale'])
    all_anchors = an_layer(inputs)
    nlayers = params['nlayers']
    arch_name = params['arch']
    model_name = arch_name + str(nlayers)
    backbone_model = ResNet(nlayers, input_tensor=inputs, data_format=data_format,
                            freeze_blocks=params['freeze_blocks'], use_pooling=True,
                            all_projections=False, use_bias=False,
                            use_batch_norm=True,
                            freeze_bn=params['freeze_bn'])
    MODELS["backbone"] = backbone_model
    P1, P2, P3, P4 = iva_bb_dict[model_name]
    C2 = backbone_model.get_layer(P1).output
    C3 = backbone_model.get_layer(P2).output
    C4 = backbone_model.get_layer(P3).output
    C5 = backbone_model.get_layer(P4).output

    backbone_feats = {2: C2, 3: C3, 4: C4, 5: C5}

    MODELS["FPN"] = fpn.FPNNetwork(params['min_level'], params['max_level'], trainable=is_training)
    fpn_feats = MODELS["FPN"](backbone_feats)

    def rpn_head_fn(features, min_level=2, max_level=6, num_anchors=3):
        """Region Proposal Network (RPN) for Mask-RCNN."""
        scores_outputs = dict()
        box_outputs = dict()

        MODELS["RPN_Heads"] = heads.RPN_Head_Model(name="rpn_head",
                                                   num_anchors=num_anchors,
                                                   data_format=data_format,
                                                   trainable=is_training)

        for level in range(min_level, max_level + 1):
            scores_outputs[level], box_outputs[level] = MODELS["RPN_Heads"](features[level])
        return scores_outputs, box_outputs

    rpn_score_outputs, rpn_box_outputs = rpn_head_fn(
        features=fpn_feats,
        min_level=params['min_level'],
        max_level=params['max_level'],
        num_anchors=len(params['aspect_ratios'] * params['num_scales'])
    )

    if is_training:
        rpn_pre_nms_topn = params['train_rpn_pre_nms_topn']
        rpn_post_nms_topn = params['train_rpn_post_nms_topn']
        rpn_nms_threshold = params['train_rpn_nms_threshold']

    else:
        rpn_pre_nms_topn = params['test_rpn_pre_nms_topn']
        rpn_post_nms_topn = params['test_rpn_post_nms_topn']
        rpn_nms_threshold = params['test_rpn_nms_thresh']

    features_image_info = Input(tensor=features['image_info'],
                                shape=image_info_shape[1:],
                                layer_class=InfoInput)
    mp_layer = MultilevelProposal(
        rpn_pre_nms_topn=rpn_pre_nms_topn,
        rpn_post_nms_topn=rpn_post_nms_topn,
        rpn_nms_threshold=rpn_nms_threshold,
        rpn_min_size=params['rpn_min_size'],
        bbox_reg_weights=None,
        use_batched_nms=params['use_batched_nms'],
        name="MLP")

    _, rpn_box_rois = mp_layer(
        inputs=[v for k, v in sorted(rpn_score_outputs.items(), reverse=False)] +
               [v for k, v in sorted(rpn_box_outputs.items(), reverse=False)] +
               [v for k, v in sorted(all_anchors.items(), reverse=False)] +
               [features_image_info])

    if is_training:

        labels_gt_boxes = Input(tensor=labels['gt_boxes'],
                                shape=gt_boxes_shape[1:],
                                layer_class=BoxInput)
        labels_gt_classes = Input(tensor=labels['gt_classes'],
                                  shape=gt_classes_shape[1:],
                                  layer_class=ClassInput)
        # Sampling
        pa_layer = ProposalAssignment(
            batch_size_per_im=params['batch_size_per_im'],
            fg_fraction=params['fg_fraction'],
            fg_thresh=params['fg_thresh'],
            bg_thresh_hi=params['bg_thresh_hi'],
            bg_thresh_lo=params['bg_thresh_lo'])
        box_targets, class_targets, rpn_box_rois, proposal_to_label_map = \
            pa_layer((rpn_box_rois, labels_gt_boxes, labels_gt_classes))

    assert params['mrcnn_resolution'] % 4 == 0, "mrcnn_resolution must be a multiple of 4."
    # Performs multi-level RoIAlign.
    box_mcr_layer = MultilevelCropResize(
        output_size=params['mrcnn_resolution'] // 4,
        is_gpu_inference=is_gpu_inference)

    box_roi_features = box_mcr_layer(
        [v for k, v in sorted(fpn_feats.items(), reverse=False)] +
        [rpn_box_rois])

    MODELS["Box_Head"] = heads.Box_Head_Model(
        num_classes=params['num_classes'],
        mlp_head_dim=params['fast_rcnn_mlp_head_dim'],
        trainable=is_training,
        name='box_head'
    )

    class_outputs, box_outputs, _ = MODELS["Box_Head"](inputs=box_roi_features)

    if not is_training:
        gpu_det_layer = GPUDetections(
            pre_nms_num_detections=params['test_rpn_post_nms_topn'],
            post_nms_num_detections=params['test_detections_per_image'],
            nms_threshold=params['test_nms'],
            bbox_reg_weights=params['bbox_reg_weights'])
        detections = gpu_det_layer(
            (class_outputs, box_outputs, rpn_box_rois, features_image_info)
        )
        model_outputs.update({
            'num_detections': detections[0],
            'detection_boxes': detections[1],
            'detection_classes': detections[2],
            'detection_scores': detections[3],
            # 'box_rois': rpn_box_rois # For debugging MultilevelProposal layer
        })

    else:  # is training
        bt_encoder = BoxTargetEncoder(bbox_reg_weights=params['bbox_reg_weights'])
        encoded_box_targets = bt_encoder(
            (rpn_box_rois, box_targets, class_targets)
        )

        model_outputs.update({
            'rpn_score_outputs': rpn_score_outputs,
            'rpn_box_outputs': rpn_box_outputs,
            'class_outputs': class_outputs,
            'box_outputs': box_outputs,
            'class_targets': class_targets,
            'box_targets': encoded_box_targets,
            'box_rois': rpn_box_rois,
        })

    # Faster-RCNN mode.
    if not params['include_mask']:
        return model_outputs

    # Mask sampling
    if not is_training:
        selected_box_rois = model_outputs['detection_boxes']
        selected_class_targets = model_outputs['detection_classes']
    else:
        fg_selector = ForegroundSelectorForMask(
            max_num_fg=int(params['batch_size_per_im'] * params['fg_fraction']))
        selected_class_targets, selected_box_targets, \
            selected_box_rois, proposal_to_label_map = fg_selector(
                (class_targets, box_targets, rpn_box_rois, proposal_to_label_map)
            )

    mask_mcr_layer = MultilevelCropResize(
        output_size=params['mrcnn_resolution'] // 2,
        is_gpu_inference=is_gpu_inference)
    mask_roi_features = mask_mcr_layer(
        [v for k, v in sorted(fpn_feats.items(), reverse=False)] +
        [selected_box_rois])

    MODELS["Mask_Head"] = heads.Mask_Head_Model(
        num_classes=params['num_classes'],
        mrcnn_resolution=params['mrcnn_resolution'],
        is_gpu_inference=is_gpu_inference,
        data_format=data_format,
        trainable=is_training,
        name="mask_head"
    )
    mask_outputs = MODELS["Mask_Head"](mask_roi_features, selected_class_targets)
    if MPI_local_rank() == 0:
        # Print #FLOPs in model.
        compute_model_statistics(batch_size, is_training=is_training)

    if is_training:
        labels_cropped_gt_masks = Input(tensor=labels['cropped_gt_masks'],
                                        shape=gt_masks_shape[1:],
                                        layer_class=MaskInput)
        mt_layer = MaskTargetsLayer(mrcnn_resolution=params['mrcnn_resolution'])
        mask_targets = mt_layer(
            (selected_box_rois,
             proposal_to_label_map,
             selected_box_targets,
             labels_cropped_gt_masks))

        model_outputs.update({
            'mask_outputs': mask_outputs,
            'mask_targets': mask_targets,
            'selected_class_targets': selected_class_targets,
        })

    else:
        model_outputs.update({
            'detection_masks': tf.keras.layers.Activation(
                'sigmoid', name='mask_sigmoid')(mask_outputs),
        })
    if is_training:
        m = tf.keras.models.Model(inputs=[inputs,
                                          features_image_info,
                                          labels_gt_boxes,
                                          labels_gt_classes,
                                          labels_cropped_gt_masks],
                                  outputs=list(model_outputs.values()))
    else:
        m = tf.keras.models.Model(inputs=[inputs, features_image_info],
                                  outputs=list(model_outputs.values()))

    if MPI_local_rank() == 0 and params.get('model_dir', ''):
        if is_training:
            model_loader.dump_json(
                m,
                os.path.join(params['model_dir'], "train_graph.json"))
        else:
            model_loader.dump_json(
                m,
                os.path.join(params['model_dir'], "eval_graph.json"))
    return model_outputs


def _model_fn(features, labels, mode, params):
    """Model defination for the Mask-RCNN model based on ResNet.

    Args:
    features: the input image tensor and auxiliary information, such as
      `image_info` and `source_ids`. The image tensor has a shape of
      [batch_size, height, width, 3]. The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include score targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    Returns:
    tpu_spec: the TPUEstimatorSpec to run training, evaluation, or prediction.
    """

    # Set up training loss and learning rate.
    global_step = tf.compat.v1.train.get_or_create_global_step()
    if mode == tf.estimator.ModeKeys.PREDICT:

        if params['include_groundtruth_in_features'] and 'labels' in features:
            # In include groundtruth for eval.
            labels = features['labels']
        else:
            labels = None

        if 'features' in features:
            features = features['features']
            # Otherwise, it is in export mode, the features is past in directly.

    model_outputs = build_model_graph(features, labels,
                                      mode == tf.estimator.ModeKeys.TRAIN, params)
    model_outputs.update({
        'source_id': features['source_ids'],
        'image_info': features['image_info'],
    })
    if 'image_path' in features:
        model_outputs.update({
            'image_path': features['image_path'],
        })

    if mode == tf.estimator.ModeKeys.PREDICT and 'orig_images' in features:
        model_outputs['orig_images'] = features['orig_images']

    # First check if it is in PREDICT mode or EVAL mode to fill out predictions.
    # Predictions are used during the eval step to generate metrics.
    if mode in [tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL]:
        predictions = {}

        try:
            model_outputs['orig_images'] = features['orig_images']
        except KeyError:
            pass

        if labels and params['include_groundtruth_in_features']:
            # Labels can only be embedded in predictions. The prediction cannot output
            # dictionary as a value.
            predictions.update(labels)

        model_outputs.pop('fpn_features', None)
        predictions.update(model_outputs)

        if mode == tf.estimator.ModeKeys.PREDICT:
            # If we are doing PREDICT, we can return here.
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # score_loss and box_loss are for logging. only total_loss is optimized.
    total_rpn_loss, rpn_score_loss, rpn_box_loss = losses.rpn_loss(
        score_outputs={int(k): v for k, v in model_outputs['rpn_score_outputs'].items()},
        box_outputs={int(k): v for k, v in model_outputs['rpn_box_outputs'].items()},
        labels=labels,
        params=params
    )
    rpn_score_loss = tf.identity(rpn_score_loss, name="log_rpn_score_loss")
    rpn_box_loss = tf.identity(rpn_box_loss, name="log_rpn_box_loss")
    total_fast_rcnn_loss, fast_rcnn_class_loss, fast_rcnn_box_loss = losses.fast_rcnn_loss(
        class_outputs=model_outputs['class_outputs'],
        box_outputs=model_outputs['box_outputs'],
        class_targets=model_outputs['class_targets'],
        box_targets=model_outputs['box_targets'],
        params=params
    )
    fast_rcnn_class_loss = tf.identity(fast_rcnn_class_loss, name="log_fast_rcnn_class_loss")
    fast_rcnn_box_loss = tf.identity(fast_rcnn_box_loss, name="log_fast_rcnn_box_loss")
    # Only training has the mask loss.
    if mode == tf.estimator.ModeKeys.TRAIN and params['include_mask']:
        mask_loss = losses.mask_rcnn_loss(
            mask_outputs=model_outputs['mask_outputs'],
            mask_targets=model_outputs['mask_targets'],
            select_class_targets=model_outputs['selected_class_targets'],
            params=params
        )

    else:
        mask_loss = 0.

    trainable_variables = list(itertools.chain.from_iterable([tf.compat.v1.trainable_variables()]))
    params['l2_weight_decay'] = params.get('l2_weight_decay', 0)
    params['l1_weight_decay'] = params.get('l1_weight_decay', 0)

    l2_regularization_loss = params['l2_weight_decay'] * tf.add_n([
        tf.nn.l2_loss(v)
        for v in trainable_variables
        if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
    ])

    l1_regularization_loss = tf.contrib.layers.apply_regularization(
        tf.keras.regularizers.l1(params['l1_weight_decay'] / 2.0),
        [v for v in trainable_variables
         if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])])

    total_loss = total_rpn_loss + total_fast_rcnn_loss + mask_loss + \
        l2_regularization_loss + l1_regularization_loss
    total_loss = tf.identity(total_loss, name="log_total_loss")
    if mode == tf.estimator.ModeKeys.EVAL:
        # Predictions can only contain a dict of tensors, not a dict of dict of
        # tensors. These outputs are not used for eval purposes.
        del predictions['rpn_score_outputs']
        del predictions['rpn_box_outputs']

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=total_loss
        )

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = learning_rates.step_learning_rate_with_linear_warmup(
            global_step=global_step,
            init_learning_rate=params['init_learning_rate'],
            warmup_learning_rate=params['warmup_learning_rate'],
            warmup_steps=params['warmup_steps'],
            learning_rate_levels=params['learning_rate_levels'],
            learning_rate_steps=params['learning_rate_steps']
        )
        learning_rate = tf.identity(learning_rate, name="learning_rate")
        optimizer = create_optimizer(learning_rate, params)

        grads_and_vars = optimizer.compute_gradients(total_loss, trainable_variables,
                                                     colocate_gradients_with_ops=True)

        gradients, variables = zip(*grads_and_vars)
        grads_and_vars = []

        # Special treatment for biases (beta is named as bias in reference model)
        for grad, var in zip(gradients, variables):

            if grad is not None and any([pattern in var.name for pattern in ["bias", "beta"]]):
                grad = 2.0 * grad

            grads_and_vars.append((grad, var))

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    else:
        train_op = None
        learning_rate = None

    # replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group

    # if not isinstance(replica_id, tf.Tensor) or tf.get_static_value(replica_id) == 0:

    #     register_metric(name="L2 loss", tensor=l2_regularization_loss,
    #                     aggregator=StandardMeter())
    #     register_metric(name="L1 loss", tensor=l1_regularization_loss,
    #                     aggregator=StandardMeter())
    #     register_metric(name="Mask loss", tensor=mask_loss, aggregator=StandardMeter())
    #     register_metric(name="Total loss", tensor=total_loss, aggregator=StandardMeter())
    #     register_metric(name="RPN box loss", tensor=rpn_box_loss,
    #                     aggregator=StandardMeter())
    #     register_metric(name="RPN score loss", tensor=rpn_score_loss,
    #                     aggregator=StandardMeter())
    #     register_metric(name="RPN total loss", tensor=total_rpn_loss,
    #                     aggregator=StandardMeter())
    #     register_metric(name="FastRCNN class loss", tensor=fast_rcnn_class_loss,
    #                     aggregator=StandardMeter())
    #     register_metric(name="FastRCNN box loss", tensor=fast_rcnn_box_loss,
    #                     aggregator=StandardMeter())
    #     register_metric(name="FastRCNN total loss", tensor=total_fast_rcnn_loss,
    #                     aggregator=StandardMeter())

    #     register_metric(name="Learning rate", tensor=learning_rate, aggregator=StandardMeter())
    #     pass

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op)


def mask_rcnn_model_fn(features, labels, mode, params):
    """Mask-RCNN model."""

    return _model_fn(
        features,
        labels,
        mode,
        params)
