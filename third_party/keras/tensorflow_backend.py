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

"""Modulus Keras-specific Extensions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import keras
from keras.backend import image_data_format
from keras.backend.tensorflow_backend import _preprocess_padding
import tensorflow as tf
from tensorflow.python.training import moving_averages


"""Logger for Keras tensorflow backend."""
logger = logging.getLogger(__name__)

DATA_FORMAT_MAP = {"channels_first": "NCHW", "channels_last": "NHWC"}


def _has_nchw_support():
    """Check whether the current scope supports NCHW ops.

    Tensorflow does not support NCHW on CPU. Therefore we check if we are not explicitly put on
    CPU, and have GPUs available. In this case there will be soft-placing on the GPU device.

    Returns:
        bool: if the current scope device placement would support nchw.
    """
    # TODO:@subha This will be removed in the future when UNET completely moves to
    # Tf.keras. Since unet uses estimator.train it internally converts the mdoel
    # to tf.keras though model was built with pure keras. The _is_current_explicit_device
    # has a function `_TfDeviceCaptureOp` that does not have attribute `_set_device_from_string`
    # This is an error of keras backend: https://github.com/tensorflow/tensorflow/issues/30728
    # Hence I catch the error and import from tensorflow.python.keras.backend
    # that has the implementation for `_set_device_from_string`.
    
    try:
        from keras.backend.tensorflow_backend import _is_current_explicit_device
        explicitly_on_cpu = _is_current_explicit_device("CPU")
    except AttributeError:
        # If tf.keras is used
        from tensorflow.python.keras.backend import _is_current_explicit_device
        explicitly_on_cpu = _is_current_explicit_device("CPU")
    gpus_available = True  # We always assume there is a GPU available.
    return not explicitly_on_cpu and gpus_available


def conv2d(
    x, kernel, strides=(1, 1), padding="valid", data_format=None, dilation_rate=(1, 1)
):
    """2D convolution.

    Arguments:
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        dilation_rate: tuple of 2 integers.

    Returns:
        A tensor, result of 2D convolution.

    Raises:
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.
    """
    if data_format is None:
        data_format = image_data_format()
    if data_format not in DATA_FORMAT_MAP:
        raise ValueError("Unknown data_format " + str(data_format))

    tf_data_format = DATA_FORMAT_MAP[data_format]
    # Avoid Tensorflow's implicit assymetric padding by explicit symmetric padding
    # See https://stackoverflow.com/questions/42924324/tensorflows-asymmetric-padding-assumptions
    if padding == "same":
        filter_shape = kernel.get_shape()
        width_padding = ((filter_shape[0].value - 1) * dilation_rate[0] + 1) // 2
        height_padding = ((filter_shape[1].value - 1) * dilation_rate[1] + 1) // 2
        if tf_data_format == "NCHW":
            padding_pattern = [
                [0, 0],
                [0, 0],
                [width_padding, width_padding],
                [height_padding, height_padding],
            ]
        else:  # 'NHWC'
            padding_pattern = [
                [0, 0],
                [width_padding, width_padding],
                [height_padding, height_padding],
                [0, 0],
            ]
        x = tf.pad(x, padding_pattern, mode="CONSTANT")
        padding = "valid"

    nhwc_roundtrip = not _has_nchw_support() and tf_data_format == "NCHW"

    if nhwc_roundtrip:
        tf_data_format = "NHWC"
        x = tf.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC

    padding = _preprocess_padding(padding)

    x = tf.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=dilation_rate,
        strides=strides,
        padding=padding,
        data_format=tf_data_format,
    )

    if nhwc_roundtrip:
        x = tf.transpose(x, (0, 3, 1, 2))  # NCHW -> NHWC

    return x


def pool2d(
    x, pool_size, strides=(1, 1), padding="valid", data_format=None, pool_mode="max"
):
    """2D Pooling.

    # Arguments
        x: Tensor or variable.
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        pool_mode: string, `"max"` or `"avg"`.
    # Returns
        A tensor, result of 2D pooling.
    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
        ValueError: if `pool_mode` is neither `"max"` or `"avg"`.
    """
    if data_format is None:
        data_format = image_data_format()
    if data_format not in DATA_FORMAT_MAP:
        raise ValueError("Unknown data_format " + str(data_format))

    tf_data_format = DATA_FORMAT_MAP[data_format]
    # Avoid Tensorflow's implicit assymetric padding by explicit symmetric padding
    if padding == "same":
        width_padding = ((pool_size[0] - 1)) // 2
        height_padding = ((pool_size[1] - 1)) // 2
        if tf_data_format == "NCHW":
            padding_pattern = [
                [0, 0],
                [0, 0],
                [width_padding, width_padding],
                [height_padding, height_padding],
            ]
        else:  # 'NHWC'
            padding_pattern = [
                [0, 0],
                [width_padding, width_padding],
                [height_padding, height_padding],
                [0, 0],
            ]
        x = tf.pad(x, padding_pattern, mode="CONSTANT")
        padding = "valid"

    nhwc_roundtrip = not _has_nchw_support() and tf_data_format == "NCHW"

    if nhwc_roundtrip:
        tf_data_format = "NHWC"
        x = tf.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC

    if nhwc_roundtrip or tf_data_format == "NHWC":
        strides = (1,) + strides + (1,)
        pool_size = (1,) + pool_size + (1,)
    else:
        strides = (1, 1) + strides
        pool_size = (1, 1) + pool_size

    padding = _preprocess_padding(padding)

    if pool_mode == "max":
        x = tf.nn.max_pool(
            x, pool_size, strides, padding=padding, data_format=tf_data_format
        )
    elif pool_mode == "avg":
        x = tf.nn.avg_pool(
            x, pool_size, strides, padding=padding, data_format=tf_data_format
        )
    else:
        raise ValueError("Invalid pooling mode:", pool_mode)

    if nhwc_roundtrip:
        x = tf.transpose(x, (0, 3, 1, 2))  # NCHW -> NHWC

    return x


def moving_average_update(x, value, momentum):
    """Compute the moving average of a variable.

    # Arguments
        x: A `Variable`.
        value: A tensor with the same shape as `x`.
        momentum: The moving average momentum.

    # Returns
        An operation to update the variable.
    """
    # See: https://github.com/keras-team/keras/commit/3ce40705a7235cabe81cfaa2ab9b9d56f225af52
    return moving_averages.assign_moving_average(
        x, value, momentum, zero_debias=False
    )  # A zero_debias==True creates unwanted tf variables.


def batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=1e-3):
    """Applies batch normalization on x given mean, var, beta and gamma.

    I.e. returns:
    `output = (x - mean) / sqrt(var + epsilon) * gamma + beta`

    # Arguments
        x: Input tensor or variable.
        mean: Mean of batch.
        var: Variance of batch.
        beta: Tensor with which to center the input.
        gamma: Tensor by which to scale the input.
        axis: Integer, the axis that should be normalized.
            (typically the features axis).
        epsilon: Fuzz factor.

    # Returns
        A tensor.

    TODO(xiangbok): Fixes a bug in keras v2.2.4, this function is adapted from #1df4052.
    """
    # if ndim(x) == 4:
    #     # The CPU implementation of FusedBatchNorm only support NHWC
    #     if axis == 1 or axis == -3:
    #         tf_data_format = 'NCHW'
    #     elif axis == 3 or axis == -1:
    #         tf_data_format = 'NHWC'
    #     else:
    #         tf_data_format = None

    #     if (x.dtype != tf.float16 and  # fused bn doesn't support fp16.
    #             (tf_data_format == 'NHWC' or (tf_data_format == 'NCHW' and _has_nchw_support()))):
    #         # The mean / var / beta / gamma may be processed by broadcast
    #         # so it may have extra axes with 1,
    #         # it is not needed and should be removed
    #         if ndim(mean) > 1:
    #             mean = tf.reshape(mean, [-1])
    #         if ndim(var) > 1:
    #             var = tf.reshape(var, [-1])
    #         if beta is None:
    #             beta = zeros_like(mean)
    #         elif ndim(beta) > 1:
    #             beta = tf.reshape(beta, [-1])
    #         if gamma is None:
    #             gamma = ones_like(mean)
    #         elif ndim(gamma) > 1:
    #             gamma = tf.reshape(gamma, [-1])
    #         y, _, _ = tf.nn.fused_batch_norm(
    #             x,
    #             gamma,
    #             beta,
    #             epsilon=epsilon,
    #             mean=mean,
    #             variance=var,
    #             data_format=tf_data_format,
    #             is_training=False
    #         )
    #         return y
    # default
    return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)


def softmax(x, axis=-1):
    """Softmax activation function.

    Patched to allow use of the backend's `softmax` regardless of the
    cardinality of the input dimensions.

    # Arguments
        x: Input tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = keras.backend.ndim(x)
    if ndim == 4 and axis == 1:
        # Nvbug 2356150: in the "channels_first" case tf.nn.softmax adds a channel swap
        # roundtrip to perform the softmax in "channels_last" order. The channel swap is done
        # through tensor shape manipulations, which TensorRT cannot handle (TensorRT needs
        # the permutation vector to be a constant). Below is a workaround for the NCHW softmax.
        # Transpose to "channels_last" order.
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        # Do the softmax in "channels_last" order (do not necessitate transpose).
        x = tf.nn.softmax(x, axis=-1)
        # Tranpose back to "channels_first".
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        return x
    if ndim >= 2:
        return tf.nn.softmax(x, axis=axis)
    raise ValueError(
        "Cannot apply softmax to a tensor that is 1D. " "Received input: %s" % x
    )


def flatten_call(self, inputs):
    """call method of Flatten layer."""
    # Overrides the suboptimal change added to keras that makes Flatten layers' channels_first
    # to be export incompatible (reverts https://github.com/keras-team/keras/pull/9696).
    return keras.backend.batch_flatten(inputs)


def _patch_backend_function(f):
    """Patch keras backend functionality.

    The patch is applied to both the general keras backend and the framework specific backend.

    Args:
        f (func): a function with the same name as exists in the keras backend.
    """
    name = f.__name__
    logger.debug("Patching %s" % name)
    keras.backend.__setattr__(name, f)
    keras.backend.tensorflow_backend.__setattr__(name, f)


def _patch_dataset_map():
    """
    Patches `tf.data.Dataset.map` function which properly sets the random seeds.

    Patches with a wrapped version of the original method which properly sets the random seeds in
    in the context of the subgraph created by the map operation.

    This patch addresses the problem that the random seed is not set in the graph used by the
    augmentations and other functions which are applied via the map operation. This issue was seen
    in TF v13.1.

    """
    # See https://github.com/tensorflow/tensorflow/issues/29101
    old_map = tf.data.Dataset.map

    def new_map(self, map_func, num_parallel_calls=None):
        seed = tf.get_default_graph().seed

        def _map_func_set_random_wrapper(*args, **kwargs):
            tf.set_random_seed(seed)
            return map_func(*args, **kwargs)

        return old_map(
            self, _map_func_set_random_wrapper, num_parallel_calls=num_parallel_calls
        )

    tf.data.Dataset.map = new_map


def patch():
    """Apply the patches to the module."""
    _patch_backend_function(conv2d)
    _patch_backend_function(pool2d)
    _patch_backend_function(moving_average_update)
    _patch_backend_function(batch_normalization)
    _patch_backend_function(_has_nchw_support)
    _patch_dataset_map()
    keras.layers.activations.__setattr__("softmax", softmax)
    keras.layers.Flatten.call = flatten_call
    keras.backend.set_image_data_format("channels_first")


def limit_tensorflow_GPU_mem(gpu_fraction=0.33):
    """Limit TensorFlow memory usage.

    Configure TensorFlow to grow its memory pool up to specified limit instead of
    greedily allocating all available GPU memory.

    Args:
        gpu_fraction (float): maximum fraction of GPU memory in TensorFlow pool
    """

    def get_session(gpu_fraction):
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True
        )
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    keras.backend.set_session(get_session(gpu_fraction=gpu_fraction))
