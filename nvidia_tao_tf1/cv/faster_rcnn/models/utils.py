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
"""Utilitity functions for FasterRCNN models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import keras
from keras.layers import Input
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils.generic_utils import CustomObjectScope
import tensorflow as tf

from nvidia_tao_tf1.cv.common.mlops.clearml import get_clearml_task
from nvidia_tao_tf1.cv.common.mlops.wandb import check_wandb_logged_in, initialize_wandb
from nvidia_tao_tf1.cv.common.utils import CUSTOM_OBJS
from nvidia_tao_tf1.cv.common.visualizer.tensorboard_visualizer import TensorBoardVisualizer
from nvidia_tao_tf1.cv.faster_rcnn.data_loader.inputs_loader import InputsLoader, RPNTargetGenerator
from nvidia_tao_tf1.cv.faster_rcnn.layers.custom_layers import (
    CropAndResize, NmsInputs, OutputParser, Proposal, ProposalTarget
)
from nvidia_tao_tf1.cv.faster_rcnn.models.darknets import DarkNet
from nvidia_tao_tf1.cv.faster_rcnn.models.efficientnet import EfficientNet
from nvidia_tao_tf1.cv.faster_rcnn.models.googlenet import GoogleNet
from nvidia_tao_tf1.cv.faster_rcnn.models.iva_vgg import IVAVGG
from nvidia_tao_tf1.cv.faster_rcnn.models.mobilenet_v1 import MobileNetV1
from nvidia_tao_tf1.cv.faster_rcnn.models.mobilenet_v2 import MobileNetV2
from nvidia_tao_tf1.cv.faster_rcnn.models.resnet101 import ResNet101
from nvidia_tao_tf1.cv.faster_rcnn.models.resnets import ResNet
from nvidia_tao_tf1.cv.faster_rcnn.models.vgg16 import VGG16


def get_optimizer(spec):
    """get the optimizer according to the spec."""
    if spec.training_config.optimizer.WhichOneof("optim") == 'adam':
        return Adam(lr=spec.training_config.optimizer.adam.lr,
                    beta_1=spec.training_config.optimizer.adam.beta_1,
                    beta_2=spec.training_config.optimizer.adam.beta_2,
                    epsilon=None,
                    decay=spec.training_config.optimizer.adam.decay,
                    amsgrad=spec.training_config.optimizer.adam.amsgrad)
    if spec.training_config.optimizer.WhichOneof("optim") == 'sgd':
        return SGD(lr=spec.training_config.optimizer.sgd.lr,
                   momentum=spec.training_config.optimizer.sgd.momentum,
                   decay=spec.training_config.optimizer.sgd.decay,
                   nesterov=spec.training_config.optimizer.sgd.nesterov)
    if spec.training_config.optimizer.WhichOneof("optim") == 'rmsprop':
        return RMSprop(lr=spec.training_config.optimizer.rmsprop.lr)
    raise ValueError('Invalid Optimizer config in spec file.')


def select_model_type(model_arch):
    '''select model type according to the config.'''
    if 'resnet:' in model_arch:
        arch = ResNet
    elif "resnet101" == model_arch:
        arch = ResNet101
    elif 'vgg:' in model_arch:
        arch = IVAVGG
    elif 'mobilenet_v1' == model_arch:
        arch = MobileNetV1
    elif 'mobilenet_v2' == model_arch:
        arch = MobileNetV2
    elif 'googlenet' == model_arch:
        arch = GoogleNet
    elif 'vgg16' == model_arch:
        arch = VGG16
    elif 'darknet' in model_arch:
        arch = DarkNet
    elif "efficientnet:" in model_arch:
        arch = EfficientNet
    else:
        raise ValueError('Unsupported model architecture: {}'.format(model_arch))
    return arch


def build_inference_model(
    model,
    config_override,
    create_session=False,
    max_box_num=100,
    regr_std_scaling=(10.0, 10.0, 5.0, 5.0),
    iou_thres=0.5,
    score_thres=0.0001,
    attach_keras_parser=True,
    eval_rois=300,
    force_batch_size=-1
):
    '''Build inference/test model from training model.'''
    def compose_call(prev_call_method):
        def call(self, inputs, training=False):
            return prev_call_method(self, inputs, training)

        return call

    def dropout_patch_call(self, inputs, training=False):
        # Just return the input tensor. Keras will map this to ``keras.backend.identity``,
        # which the TensorRT 3.0 UFF parser supports.
        return inputs

    # Patch BatchNormalization and Dropout call methods so they don't create
    # the training part of the graph.
    prev_batchnorm_call = keras.layers.normalization.BatchNormalization.call
    prev_td_call = keras.layers.wrappers.TimeDistributed.call
    prev_dropout_call = keras.layers.Dropout.call
    keras.layers.normalization.BatchNormalization.call = compose_call(
        prev_batchnorm_call
    )
    keras.layers.wrappers.TimeDistributed.call = compose_call(
        prev_td_call
    )
    keras.layers.Dropout.call = dropout_patch_call

    _explored_layers = dict()
    for l in model.layers:
        _explored_layers[l.name] = [False, None]
    input_layer = [l for l in model.layers if (type(l) == keras.layers.InputLayer)]
    layers_to_explore = input_layer
    model_outputs = {}
    # Loop until we reach the last layer.
    while layers_to_explore:
        layer = layers_to_explore.pop(0)
        # Skip layers that may be revisited in the graph to prevent duplicates.
        if not _explored_layers[layer.name][0]:
            # Check if all inbound layers explored for given layer.
            if not all([
                    _explored_layers[l.name][0]
                    for n in layer._inbound_nodes
                    for l in n.inbound_layers
                    ]):
                continue
            outputs = None
            # Visit input layer.
            if type(layer) == keras.layers.InputLayer and layer.name == 'input_image':
                # Re-use the existing InputLayer.
                outputs = layer.output
                new_layer = layer
            elif type(layer) == keras.layers.InputLayer:
                # skip the input_class_ids and input_gt_boxes
                # mark them as visited but do nothing essential
                _explored_layers[layer.name][0] = True
                _explored_layers[layer.name][1] = None
                layers_to_explore.extend([node.outbound_layer for node in layer._outbound_nodes])
                continue
            # special handling for ProposalTarget layer.
            elif type(layer) == ProposalTarget:
                # get ROIs data.
                for node in layer._inbound_nodes:
                    prev_outputs = []
                    # only use the first Input: input_rois
                    for idx, l in enumerate(node.inbound_layers[:1]):
                        keras_layer = _explored_layers[l.name][1]
                        prev_outputs.append(keras_layer.get_output_at(node.node_indices[idx]))
                    assert prev_outputs, "Expected non-input layer to have inputs."
                    # remember it
                    if len(prev_outputs) == 1:
                        prev_outputs = prev_outputs[0]
                    proposal_outputs = prev_outputs
                _explored_layers[layer.name][0] = True
                _explored_layers[layer.name][1] = None
                layers_to_explore.extend([node.outbound_layer for node in layer._outbound_nodes])
                continue
            # special handling of CropAndResize to skip the ProposalTarget layer.
            elif type(layer) == CropAndResize:
                # Create new layer.
                layer_config = layer.get_config()
                new_layer = type(layer).from_config(layer_config)
                outputs = []
                for node in layer._inbound_nodes:
                    prev_outputs = []
                    for idx, l in enumerate(node.inbound_layers):
                        # skip ProposalTarget(idx==1) because it doesn't exist
                        # in validation model. Use None as a placeholder for it
                        # will update the None later
                        if idx == 1:
                            prev_outputs.append(None)
                            continue
                        keras_layer = _explored_layers[l.name][1]
                        prev_outputs.append(keras_layer.get_output_at(node.node_indices[idx]))
                    assert prev_outputs, "Expected non-input layer to have inputs."
                    # replace None with the proposal_outputs
                    prev_outputs[1] = proposal_outputs
                    if len(prev_outputs) == 1:
                        prev_outputs = prev_outputs[0]
                    outputs.append(new_layer(prev_outputs))
                if len(outputs) == 1:
                    outputs = outputs[0]
            elif ("pre_pool_reshape" in layer.name and type(layer) == keras.layers.Reshape):
                H, W = layer._inbound_nodes[0].inbound_layers[0].output_shape[3:]
                new_layer = keras.layers.Reshape((-1, H, W), name=layer.name)
                outputs = []
                for node in layer._inbound_nodes:
                    prev_outputs = []
                    for idx, l in enumerate(node.inbound_layers):
                        keras_layer = _explored_layers[l.name][1]
                        prev_outputs.append(keras_layer.get_output_at(node.node_indices[idx]))
                    assert prev_outputs, "Expected non-input layer to have inputs."
                    if len(prev_outputs) == 1:
                        prev_outputs = prev_outputs[0]
                    outputs.append(new_layer(prev_outputs))
                if len(outputs) == 1:
                    outputs = outputs[0]
            elif ("post_pool_reshape" in layer.name and type(layer) == keras.layers.Reshape):
                new_layer = keras.layers.Reshape((eval_rois, -1, 1, 1), name=layer.name)
                outputs = []
                for node in layer._inbound_nodes:
                    prev_outputs = []
                    for idx, l in enumerate(node.inbound_layers):
                        keras_layer = _explored_layers[l.name][1]
                        prev_outputs.append(keras_layer.get_output_at(node.node_indices[idx]))
                    assert prev_outputs, "Expected non-input layer to have inputs."
                    if len(prev_outputs) == 1:
                        prev_outputs = prev_outputs[0]
                    outputs.append(new_layer(prev_outputs))
                if len(outputs) == 1:
                    outputs = outputs[0]
            else:
                # Create new layer.
                layer_config = layer.get_config()
                # override config for Proposal layer for test graph
                if type(layer) == Proposal:
                    layer_config.update(config_override)
                new_layer = type(layer).from_config(layer_config)

                # Add to model.
                outputs = []
                for node in layer._inbound_nodes:
                    prev_outputs = []
                    for idx, l in enumerate(node.inbound_layers):
                        keras_layer = _explored_layers[l.name][1]
                        prev_outputs.append(keras_layer.get_output_at(node.node_indices[idx]))
                    assert prev_outputs, "Expected non-input layer to have inputs."
                    if len(prev_outputs) == 1:
                        prev_outputs = prev_outputs[0]
                    outputs.append(new_layer(prev_outputs))
                if len(outputs) == 1:
                    outputs = outputs[0]
                weights = layer.get_weights()
                if weights is not None:
                    new_layer.set_weights(weights)
            outbound_nodes = layer._outbound_nodes
            # RPN outputs will be excluded since it has outbound nodes.
            if not outbound_nodes:
                model_outputs[layer.output.name] = outputs
            layers_to_explore.extend([node.outbound_layer for node in outbound_nodes])
            # Mark current layer as visited and assign output nodes to the layer.
            _explored_layers[layer.name][0] = True
            _explored_layers[layer.name][1] = new_layer
        else:
            continue
    # Create new keras model object from pruned specifications.
    # only use input_image as Model Input.
    output_tensors = [model_outputs[l.name] for l in model.outputs if l.name in model_outputs]
    output_tensors = [proposal_outputs] + output_tensors
    new_model = keras.models.Model(inputs=model.inputs[:1], outputs=output_tensors, name=model.name)
    # attach OutputParser layer
    if attach_keras_parser:
        parser_outputs = OutputParser(max_box_num, list(regr_std_scaling), iou_thres, score_thres)(
            new_model.outputs + new_model.inputs
        )
        new_model = keras.models.Model(
            inputs=new_model.inputs,
            outputs=parser_outputs,
            name=new_model.name
        )
    else:
        # prepare NMS input tensors for TensorRT NMSPlugin inference
        nms_inputs = NmsInputs(regr_std_scaling)(new_model.outputs)
        new_model = keras.models.Model(
            inputs=new_model.inputs,
            outputs=nms_inputs,
            name=new_model.name
        )
    # save model to file, reset the tf graph and load it to make sure the tf op names
    # not appended with _n
    os_handle, temp_file_name = tempfile.mkstemp()
    os.close(os_handle)
    input_shape = list(new_model.get_layer("input_image").output_shape)
    input_shape = tuple([force_batch_size] + input_shape[1:])
    with CustomObjectScope(CUSTOM_OBJS):
        new_model.save(temp_file_name)
    # clear old tf graph and session
    keras.backend.clear_session()
    if create_session:
        # create a new tf session and use it as Keras session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        keras.backend.set_session(tf.Session(config=config))
    keras.backend.set_learning_phase(0)

    def _get_input_layer(*args, **argskw):
        return keras.layers.InputLayer(batch_input_shape=input_shape, name="input_image")
    # Force a static batch size for the keras model in case we want to export
    # to an onnx model with static batch size
    if force_batch_size > 0:
        with CustomObjectScope({**CUSTOM_OBJS, "InputLayer": _get_input_layer}):
            new_model = keras.models.load_model(temp_file_name, compile=False)
    else:
        with CustomObjectScope({**CUSTOM_OBJS}):
            new_model = keras.models.load_model(temp_file_name, compile=False)
    os.remove(temp_file_name)
    # Unpatch Keras before return.
    keras.layers.normalization.BatchNormalization.call = prev_batchnorm_call
    keras.layers.wrappers.TimeDistributed.call = prev_td_call
    keras.layers.Dropout.call = prev_dropout_call
    return new_model


def build_or_resume_model(spec, hvd, logger, results_dir):
    '''Build a new model or resume a checkpoint model.'''
    # Define visualizer
    visualizer = TensorBoardVisualizer()
    visualizer.build_from_config(
        spec.training_config.visualizer
    )
    # Disabling visualizer from all other processes
    # other than rank 0 process.
    if not hvd.rank() == 0:
        visualizer.enabled = False

    # Setting up clearml integration
    task = None
    if hvd.rank() == 0:
        if spec.training_config.visualizer.HasField("clearml_config"):
            clearml_config = spec.training_config.visualizer.clearml_config
            task = get_clearml_task(clearml_config, "faster_rcnn")
        if spec.training_config.visualizer.HasField("wandb_config"):
            wandb_config = spec.training_config.visualizer.wandb_config
            wandb_logged_in = check_wandb_logged_in()
            if wandb_logged_in:
                wandb_name = f"{wandb_config.name}" if wandb_config.name else \
                    "faster_rcnn_train"
                initialize_wandb(
                    project=wandb_config.project if wandb_config.project else None,
                    entity=wandb_config.entity if wandb_config.entity else None,
                    notes=wandb_config.notes if wandb_config.notes else None,
                    tags=wandb_config.tags if wandb_config.tags else None,
                    sync_tensorboard=True,
                    save_code=False,
                    results_dir=results_dir,
                    wandb_logged_in=wandb_logged_in,
                    name=wandb_name
                )
    # build input tensors
    data_loader = InputsLoader(
        spec.training_dataset,
        spec.data_augmentation,
        spec.batch_size_per_gpu,
        spec.image_c,
        spec.image_mean_values,
        spec.image_scaling_factor,
        bool(spec.image_channel_order == 'bgr'),
        max_objs_per_img=spec.max_objs_per_img,
        training=True,
        enable_augmentation=spec.enable_augmentation,
        visualizer=visualizer,
        rank=hvd.rank()
    )
    img_input = Input(shape=spec.input_dims, name='input_image', tensor=data_loader.images)
    gt_cls_input = Input(shape=(None,), name='input_gt_cls', tensor=data_loader.gt_classes)
    gt_bbox_input = Input(shape=(None, 4), name='input_gt_bbox', tensor=data_loader.gt_boxes)
    # build the model
    model_type = select_model_type(spec._backbone)
    model = model_type(spec.nlayers, spec.batch_size_per_gpu,
                       spec.rpn_stride, spec.reg_type,
                       spec.weight_decay, spec.freeze_bn, spec.freeze_blocks,
                       spec.dropout_rate, spec.drop_connect_rate,
                       spec.conv_bn_share_bias, spec.all_projections,
                       spec.use_pooling, spec.anchor_sizes, spec.anchor_ratios,
                       spec.roi_pool_size, spec.roi_pool_2x, spec.num_classes,
                       spec.std_scaling, spec.rpn_pre_nms_top_N, spec.rpn_post_nms_top_N,
                       spec.rpn_nms_iou_thres, spec.gt_as_roi,
                       spec.rcnn_min_overlap, spec.rcnn_max_overlap, spec.rcnn_train_bs,
                       spec.rcnn_regr_std, spec.rpn_train_bs, spec.lambda_rpn_class,
                       spec.lambda_rpn_regr, spec.lambda_cls_class, spec.lambda_cls_regr,
                       "frcnn_"+spec._backbone.replace(":", "_"), results_dir,
                       spec.enc_key, spec.lr_scheduler,
                       spec.enable_qat,
                       activation_type=spec.activation_type,
                       early_stopping=spec.early_stopping)
    if spec.resume_from_model:
        # resume training from an existing model
        initial_epoch = model.resume_model(spec,
                                           [img_input, gt_cls_input, gt_bbox_input],
                                           hvd,
                                           logger=logger)
    else:
        # build a new model, from scratch or from pruned model
        initial_epoch = 0
        if spec.pretrained_model:
            model.build_model_from_pruned(spec.pretrained_model,
                                          img_input, gt_cls_input,
                                          gt_bbox_input, logger,
                                          spec.regularization_config)
        else:
            model.build_keras_model(img_input, gt_cls_input, gt_bbox_input)
        if (not spec.pretrained_model) and spec.pretrained_weights:
            model.load_weights(spec.pretrained_weights, spec.enc_key, logger)
    if spec.training_config.model_parallelism:
        model.parallelize(
            tuple(spec.training_config.model_parallelism),
        )
    # build target tensors
    rpn_target_generator = RPNTargetGenerator(
        # dynamic image width
        tf.shape(data_loader.images)[3],
        # dynamic image height
        tf.shape(data_loader.images)[2],
        # dynamic RPN width
        tf.shape(model.keras_model.outputs[0])[3],
        # dynamic RPN height
        tf.shape(model.keras_model.outputs[0])[2],
        spec.rpn_stride,
        spec.anchor_sizes,
        spec.anchor_ratios,
        spec.batch_size_per_gpu,
        spec.rpn_max_overlap,
        spec.rpn_min_overlap,
        spec.rpn_train_bs,
        max_objs_per_image=spec.max_objs_per_img
    )
    # Visualize model weights histogram
    if hvd.rank() == 0 and spec.training_config.visualizer.enabled:
        visualizer.keras_model_weight_histogram(model.keras_model)
    # assign to model for ease of access and testing
    model.rpn_target_generator = rpn_target_generator
    rpn_scores_tensor, rpn_deltas_tensor = data_loader.generate_rpn_targets(
        rpn_target_generator.build_rpn_target_batch
    )
    model.set_target_tensors(rpn_scores_tensor, rpn_deltas_tensor)
    # build others
    model.set_optimizer(get_optimizer(spec), hvd)
    _total_examples = (data_loader.num_samples + spec.batch_size_per_gpu - 1)
    iters_per_epoch = _total_examples // spec.batch_size_per_gpu
    iters_per_epoch = (iters_per_epoch + hvd.size() - 1) // hvd.size()
    model.build_losses()
    initial_step = initial_epoch * iters_per_epoch
    model.build_lr_scheduler(spec.epochs*iters_per_epoch, hvd.size(), initial_step)
    model.set_hvd_callbacks(hvd)
    if hvd.rank() == 0:
        if spec.checkpoint_interval > spec.epochs:
            logger.warning(
                f"Checkpoint interval: {spec.checkpoint_interval} larger than training epochs: "
                f"{spec.epochs}, disabling checkpoint."
            )
        else:
            model.build_checkpointer(spec.checkpoint_interval)
    # build validation data loader
    # firstly, check if the validation dataset is empty(None)
    # To avoid horovod hang, we have to do validation for all the processes
    # otherwise, only rank 0 will do validation and takes a few more minutes, it will
    # results in horovod deadlock.
    if spec.training_dataset.WhichOneof('dataset_split_type') in [
        'validation_fold',
        'validation_data_source',
    ]:
        logger.info("Building validation dataset...")
        val_data_loader = InputsLoader(
            spec.training_dataset,
            spec.data_augmentation,
            spec.eval_batch_size,
            spec.image_c,
            spec.image_mean_values,
            spec.image_scaling_factor,
            bool(spec.image_channel_order == 'bgr'),
            max_objs_per_img=spec.max_objs_per_img,
            training=False,
            session=keras.backend.get_session()
        )
        logger.info("Validation dataset built successfully!")
        if val_data_loader.num_samples > 0:
            model.build_validation_callback(val_data_loader, spec)
    else:
        logger.info('No validation dataset found, skip validation during training.')
    if hvd.rank() == 0 and spec.training_config.visualizer.enabled:
        logger.info('TensorBoard Visualization Enabled')
        # Can only write to tb file by 1 process
        model.build_tensorboard_callback()
    if hvd.rank() == 0:
        model.build_status_logging_callback(results_dir, spec.epochs, True)
    model.build_early_stopping_callback()
    model.compile()
    return model, iters_per_epoch, initial_epoch, task
