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

"""Model function defining the model, loss and outputs.

This function provides the model function that instantiates the unet model
along with loss, model hyper-parameters and predictions.

"""

import logging
import os
import random
import numpy as np
import tensorflow as tf
from nvidia_tao_tf1.cv.unet.distribution import distribution
from nvidia_tao_tf1.cv.unet.proto.regularizer_config_pb2 import RegularizerConfig

INFREQUENT_SUMMARY_KEY = "infrequent_summary"
FREQUENT_SUMMARY_KEY = "frequent_summary"

logger = logging.getLogger(__name__)


# Class Dice coefficient averaged over batch
def dice_coef(predict, target, axis=1, eps=0):
    """helper function to compute the dice coefficient."""

    intersection = tf.reduce_sum(predict * target, axis=axis)
    mask_sum = tf.reduce_sum(predict * predict + target * target, axis=axis)
    dice = (2. * intersection + eps) / (mask_sum + eps)
    dice_coef = tf.reduce_mean(dice, axis=0)

    return dice_coef  # average over batch


def regularization_l2loss(weight_decay):
    """helper function to compute regularization loss."""

    def loss_filter_fn(name):
        """we don't need to compute L2 loss for BN."""

        return all([
                    tensor_name not in name.lower()
                    for tensor_name in ["batchnorm", "batch_norm", "batch_normalization"]
                    ])

    filtered_params = [tf.cast(v, tf.float32) for v in tf.trainable_variables()
                       if loss_filter_fn(v.name)]

    if len(filtered_params) != 0:

        l2_loss_per_vars = [tf.nn.l2_loss(v) for v in filtered_params]
        l2_loss = tf.multiply(tf.add_n(l2_loss_per_vars), weight_decay)

    else:
        l2_loss = tf.zeros(shape=(), dtype=tf.float32)

    return l2_loss


def is_using_hvd():
    """Function to determine if the hvd is used."""

    env_vars = ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]
    if all([var in os.environ for var in env_vars]):
        return True
    return False


def get_learning_rate(lr_init, params, global_step):
    """Function to determine the learning rate based on scheduler."""

    if params.lr_scheduler:
        if params.lr_scheduler.WhichOneof("lr_scheduler") == 'cosine_decay':
            return tf.compat.v1.train.cosine_decay(
                lr_init, global_step, params.lr_scheduler.cosine_decay.decay_steps,
                alpha=params.lr_scheduler.cosine_decay.alpha, name=None)
        if params.lr_scheduler.WhichOneof("lr_scheduler") == 'exponential_decay':
            return tf.compat.v1.train.exponential_decay(
                lr_init, global_step, params.lr_scheduler.exponential_decay.decay_steps,
                decay_rate=params.lr_scheduler.exponential_decay.decay_rate,
                staircase=True, name=None)

        raise NotImplementedError('The provided learning rate scheduler is not supported.')
    # Return constant learning rate
    return lr_init


def tensorboard_visualize(tensor, tensor_name, visualize):
    """Helper function to visualize the tensors on Tensorboard."""

    frequent_collections = [FREQUENT_SUMMARY_KEY]
    if visualize:
        tf.identity(tensor, name=tensor_name)
        tf.summary.scalar(tensor_name, tensor, collections=frequent_collections)


def get_logits(output_map_activation, params):
    """Function to compute logits."""
    # Return the predictions which is class integer map
    if params["activation"] == "sigmoid":
        cond = tf.less(output_map_activation, 0.5 * tf.ones(tf.shape(output_map_activation)))
        logits = tf.where(cond, tf.zeros(tf.shape(output_map_activation)),
                          tf.ones(tf.shape(output_map_activation)))
    else:
        logits = tf.compat.v1.argmax(output_map_activation, axis=1)

    return logits


def get_color_id(num_classes):
    """Function to return a list of color values for each class."""

    colors = []
    for idx in range(num_classes):
        random.seed(idx)
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    colors_2 = [(100, 100, 100), (255, 0, 0)]
    colors = colors_2 + list(set(colors) - set(colors_2))
    return colors


def convert_class_id_to_color(tensor, palette):
    """Function to convert class ID map to color img."""

    H, W = tf.shape(tensor)[1], tf.shape(tensor)[2]
    palette = tf.constant(palette, dtype=tf.uint8)
    flat_logits = tf.reshape(tensor,
                             [tf.shape(tensor)[0], -1])
    color_image = tf.gather(palette, flat_logits)
    color_image = tf.reshape(color_image, [-1, H, W, 3])

    return color_image


def visualize_image_color(tensor, x_orig, num_classes, labels_gt):
    """Fnction to visualize the prediction on the input image during trainin on TB."""

    image_collections = [INFREQUENT_SUMMARY_KEY]
    colors = get_color_id(num_classes)
    tf.summary.image("input_image", x_orig, collections=image_collections)
    palette = np.array(colors, np.uint8)
    color_image = convert_class_id_to_color(tensor, palette)
    color_image_gt = convert_class_id_to_color(labels_gt, palette)
    tf.summary.image("predicted_image", color_image, collections=image_collections)
    color_image_vis = color_image/2
    x_orig_vis = x_orig/2
    overlay_img = x_orig_vis + color_image_vis
    overlay_img_gt = color_image_gt/2 + x_orig_vis
    tf.summary.image("predicted_overlay", overlay_img, collections=image_collections)
    tf.summary.image("groundtruth_overlay", overlay_img_gt, collections=image_collections)


def unet_fn(features, labels, mode, params):
    """Model function for tf.Estimator.

    Controls how the training is performed by specifying how the
    total_loss is computed and applied in the backward pass.

    Args:
        features (tf.Tensor): Tensor samples
        labels (tf.Tensor): Tensor labels
        mode (tf.estimator.ModeKeys): Indicates if we train, evaluate or predict
        params (dict): Additional parameters supplied to the estimator

    Returns:
        Appropriate tf.estimator.EstimatorSpec for the current mode

    """

    dtype = tf.float32
    logger.info(params)
    device = '/gpu:0'
    global_step = tf.compat.v1.train.get_global_step()

    if mode == tf.estimator.ModeKeys.TRAIN:
        lr = get_learning_rate(params.learning_rate, params, global_step)

    if isinstance(features, dict):
        features_new = features['features'], features['labels'], features['x_orig']
        features, labels_pred, x_orig = features_new

    with tf.device(device):
        # Model function is instantitated on One GPU.

        features = tf.cast(features, dtype)
        unet_model = params.unet_model
        # Constructing the unet model
        img_height, img_width, img_channels = \
            params.experiment_spec.model_config.model_input_height, \
            params.experiment_spec.model_config.model_input_width, \
            params.experiment_spec.model_config.model_input_channels

        unet_model.construct_model(input_shape=(img_channels, img_height, img_width),
                                   pretrained_weights_file=params.pretrained_weights_file,
                                   enc_key=params.key, model_json=params.model_json,
                                   custom_objs=params.custom_objs)

        unet_model.keras_model.summary()
        output_map = unet_model.keras_model(features)
        if params["activation"] == "sigmoid":
            output_map_activation = tf.math.sigmoid(output_map)
        else:
            output_map_activation = tf.nn.softmax(output_map, axis=1)

        if params.visualize and params.phase == "train":
            # For GT  vis
            labels_gt = labels
            if params["activation"] == "softmax":
                labels_gt = tf.compat.v1.argmax(labels_gt, axis=1)
            labels_gt = tf.cast(labels_gt, tf.int64)
            logits_img = get_logits(output_map_activation, params)
            logits_img = tf.expand_dims(logits_img, axis=-1)
            logits_img = tf.cast(logits_img, tf.int64)
            visualize_image_color(logits_img, x_orig, params.num_classes, labels_gt)

        if mode == tf.estimator.ModeKeys.PREDICT:

            logits = get_logits(output_map_activation, params)
            if params.phase == "test":
                predictions = {"logits": logits}
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            # For evaluation
            if params["activation"] == "softmax":
                labels_pred = tf.compat.v1.argmax(labels_pred, axis=1)
            flat_logits = tf.reshape(tf.cast(logits, tf.float32),
                                     [tf.shape(logits)[0], -1])
            flat_labels = tf.reshape(labels_pred,
                                     [tf.shape(labels_pred)[0], -1])
            elems = (flat_labels, flat_logits)
            conf_matrix = tf.map_fn(lambda x: tf.math.confusion_matrix(x[0], x[1],
                                    num_classes=params.num_conf_mat_classes,
                                    dtype=tf.float32), elems, dtype=tf.float32)
            predictions = {'conf_matrix': conf_matrix}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Transpose the output map
        trainable_variables = tf.compat.v1.trainable_variables()
        if params.experiment_spec.training_config.regularizer.type == RegularizerConfig.L2:
            regularization_loss = params.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in trainable_variables if not
                 any([pattern in v.name for pattern in
                     ["batch_normalization", "bias", "beta"]])])
        elif params.experiment_spec.training_config.regularizer.type == RegularizerConfig.L1:
            l1_regularizer = tf.contrib.layers.l1_regularizer(scale=params.weight_decay, scope=None)
            regularization_loss = tf.contrib.layers.apply_regularization(
                l1_regularizer, [v for v in trainable_variables if not
                                 any([pattern in v.name for pattern in
                                     ["batch_normalization", "bias", "beta"]])])
        else:
            # Setting reg to 0 when no regularization is provided
            l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.0, scope=None)
            regularization_loss = tf.contrib.layers.apply_regularization(
                l1_regularizer, trainable_variables)

        # Debug the tensors for NaN
        if params.visualize and params.weights_monitor:
            # Visualize the weights and gradients
            histogram_collections = [INFREQUENT_SUMMARY_KEY]
            for tr_v in trainable_variables:
                tf.debugging.check_numerics(tr_v, message='Output map had NaN/ \
                                            Infinity values.')
                tf.compat.v1.verify_tensor_all_finite(tr_v, msg="Nan")
                tf.summary.histogram(tr_v.name, tr_v, collections=histogram_collections)

        if params.activation == "sigmoid":
            crossentropy_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=output_map,
                                                        labels=labels),
                name='cross_loss_ref')
        else:
            crossentropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_map,
                                                           labels=labels,
                                                           axis=1),
                name='cross_loss_ref')

        if params.loss in ["cross_dice_sum", "dice"]:
            dice_loss = tf.reduce_mean(1 - dice_coef(output_map_activation, labels),
                                       name='dice_loss')
            tensorboard_visualize(dice_loss, 'dice_loss', params.visualize)
        if params.loss == "cross_dice_sum":
            total_loss = tf.add(crossentropy_loss, dice_loss)
            tensorboard_visualize(total_loss, 'cross_dice_loss', params.visualize)
            total_loss = tf.add(total_loss, regularization_loss, name="total_loss_ref")
        elif params.loss == "cross_entropy":
            tensorboard_visualize(crossentropy_loss, 'crossentropy_loss', params.visualize)
            total_loss = tf.add(crossentropy_loss, regularization_loss, name="total_loss_ref")
        elif params.loss == "dice":
            total_loss = tf.add(dice_loss, regularization_loss, name="total_loss_ref")

        tensorboard_visualize(total_loss, 'total_loss', params.visualize)
        tensorboard_visualize(regularization_loss, 'regularization_loss', params.visualize)
        hooks = []
        if params.visualize:
            events_dir = os.path.join(params.model_dir, "events")
            save_steps_frequent = params.save_summary_steps
            save_steps_infrequent = params.infrequent_save_summary_steps
            infrequent_summary_hook = tf.train.SummarySaverHook(
                save_steps=save_steps_infrequent,
                output_dir=events_dir,
                scaffold=tf.train.Scaffold(
                    summary_op=tf.summary.merge_all(key=INFREQUENT_SUMMARY_KEY)))
            frequent_summary_hook = tf.train.SummarySaverHook(
                save_steps=save_steps_frequent, output_dir=events_dir,
                scaffold=tf.train.Scaffold(
                    summary_op=tf.summary.merge_all(key=FREQUENT_SUMMARY_KEY)))
            hooks += [infrequent_summary_hook, frequent_summary_hook]
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {"eval_ce_loss": tf.compat.v1.metrics.mean(crossentropy_loss),
                               "eval_dice_loss": tf.compat.v1.metrics.mean(dice_loss),
                               "eval_total_loss": tf.compat.v1.metrics.mean(total_loss),
                               "eval_dice_score": tf.compat.v1.metrics.mean(1.0 - dice_loss)
                               }
            return tf.estimator.EstimatorSpec(mode=mode, loss=dice_loss,
                                              eval_metric_ops=eval_metric_ops)

        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        if distribution.get_distributor().is_distributed():
            opt = distribution.get_distributor().distribute_optimizer(opt)

        with tf.control_dependencies(
                tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            deterministic = True
            gate_gradients = (
                tf.compat.v1.train.Optimizer.GATE_OP
                if deterministic
                else tf.compat.v1.train.Optimizer.GATE_NONE)

            train_op = opt.minimize(total_loss, gate_gradients=gate_gradients,
                                    global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op,
                                      training_hooks=hooks,
                                      eval_metric_ops={})
