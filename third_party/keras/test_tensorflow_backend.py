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

"""Test the Keras backend changes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras import backend as K
import numpy as np
from numpy.testing import assert_allclose
import pytest
import tensorflow as tf

import third_party.keras.tensorflow_backend

FILTERS = 5


conv2d_tests = [
    # channels_last, valid and same, and with different striding.
    ((1, 4, 4, 3), (1, 4, 4, FILTERS), (3, 3), (1, 1), "same", "channels_last"),
    ((1, 4, 4, 3), (1, 2, 2, FILTERS), (3, 3), (2, 2), "same", "channels_last"),
    ((1, 4, 4, 3), (1, 2, 2, FILTERS), (3, 3), (1, 1), "valid", "channels_last"),
    ((1, 4, 4, 3), (1, 1, 1, FILTERS), (3, 3), (2, 2), "valid", "channels_last"),
    # channels_first, valid and same, and with different striding.
    ((1, 3, 4, 4), (1, FILTERS, 4, 4), (3, 3), (1, 1), "same", "channels_first"),
    ((1, 3, 4, 4), (1, FILTERS, 2, 2), (3, 3), (2, 2), "same", "channels_first"),
    ((1, 3, 4, 4), (1, FILTERS, 2, 2), (3, 3), (1, 1), "valid", "channels_first"),
    ((1, 3, 4, 4), (1, FILTERS, 1, 1), (3, 3), (2, 2), "valid", "channels_first"),
]


@pytest.mark.parametrize(
    "shape,expected_shape,kernel,strides,padding,data_format", conv2d_tests
)
def test_conv2d(shape, expected_shape, kernel, strides, padding, data_format):
    """Test the conv2d padding and data-format compatibility and shapes.

    These checks are needed, as we have patched it to use explicit symmetric padding.
    """
    inputs = tf.placeholder(shape=shape, dtype=tf.float32)
    x = keras.layers.Conv2D(
        filters=FILTERS,
        kernel_size=kernel,
        strides=strides,
        padding=padding,
        data_format=data_format,
    )(inputs)
    sess = keras.backend.get_session()
    out = sess.run(x, feed_dict={inputs: np.zeros(shape)})
    output_shape = out.shape
    assert output_shape == expected_shape


conv2d_transpose_tests = [
    # channels_last, valid and same, and with different striding.
    ((1, 4, 4, 3), (1, 4, 4, FILTERS), (3, 3), (1, 1), "same", "channels_last"),
    ((1, 4, 4, 3), (1, 8, 8, FILTERS), (3, 3), (2, 2), "same", "channels_last"),
    ((1, 4, 4, 3), (1, 6, 6, FILTERS), (3, 3), (1, 1), "valid", "channels_last"),
    ((1, 4, 4, 3), (1, 9, 9, FILTERS), (3, 3), (2, 2), "valid", "channels_last"),
    # channels_first, valid and same, and with different striding.
    ((1, 3, 4, 4), (1, FILTERS, 4, 4), (3, 3), (1, 1), "same", "channels_first"),
    ((1, 3, 4, 4), (1, FILTERS, 8, 8), (3, 3), (2, 2), "same", "channels_first"),
    ((1, 3, 4, 4), (1, FILTERS, 6, 6), (3, 3), (1, 1), "valid", "channels_first"),
    ((1, 3, 4, 4), (1, FILTERS, 9, 9), (3, 3), (2, 2), "valid", "channels_first"),
]


@pytest.mark.parametrize(
    "shape,expected_shape,kernel,strides,padding,data_format", conv2d_transpose_tests
)
def test_conv2d_transpose(shape, expected_shape, kernel, strides, padding, data_format):
    inputs = tf.placeholder(shape=shape, dtype=tf.float32)
    x = keras.layers.Conv2DTranspose(
        filters=FILTERS,
        kernel_size=kernel,
        strides=strides,
        padding=padding,
        data_format=data_format,
    )(inputs)
    sess = keras.backend.get_session()
    out = sess.run(x, feed_dict={inputs: np.zeros(shape)})
    output_shape = out.shape
    assert output_shape == expected_shape


bias_tests = [
    ((1, 3, 4, 4), (3), "channels_first"),
    ((1, 4, 4, 3), (3), "channels_last"),
]


@pytest.mark.parametrize("input_shape,bias_shape,data_format", bias_tests)
def test_bias(input_shape, bias_shape, data_format):
    """Test the bias_add improvements.

    These improvements allow for native usage of NCHW format without transposing or reshaping.
    The bias will never change the output size, so we only check if this function does not fail.
    """
    inputs = tf.placeholder(shape=input_shape, dtype=tf.float32)
    bias = tf.placeholder(shape=bias_shape, dtype=tf.float32)
    x = keras.backend.bias_add(inputs, bias, data_format=data_format)
    sess = keras.backend.get_session()
    sess.run(x, feed_dict={inputs: np.zeros(input_shape), bias: np.zeros(bias_shape)})


batch_norm_tests = [
    # All reduction
    ((1024, 3, 4, 4), -1),
    # NCHW reduction
    ((1024, 3, 2, 2), 1),
    # NHWC reduction
    ((1024, 2, 2, 3), 3),
]


@pytest.mark.parametrize("input_shape,axis", batch_norm_tests)
@pytest.mark.parametrize("is_training", [True])
def test_batchnorm_correctness(mocker, input_shape, axis, is_training):
    """Test batchnorm by training it on a normal distributed that is offset and scaled."""
    mocker.spy(tf.nn, "fused_batch_norm")

    inputs = keras.layers.Input(shape=input_shape[1:])
    bn_layer = keras.layers.normalization.BatchNormalization(axis=axis, momentum=0.8)
    bn = bn_layer(inputs)
    model = keras.models.Model(inputs=inputs, outputs=bn)

    model.compile(loss="mse", optimizer="sgd")

    mean, var = 5, 10
    x = np.random.normal(loc=mean, scale=var, size=input_shape)

    model.fit(x, x, epochs=50, verbose=0)

    beta = K.eval(bn_layer.beta)
    gamma = K.eval(bn_layer.gamma)

    nchannels = input_shape[axis]
    beta_expected = np.array([mean] * nchannels, dtype=np.float32)
    gamma_expected = np.array([var] * nchannels, dtype=np.float32)

    # Test if batchnorm has learned the correct mean and variance
    assert_allclose(beta, beta_expected, atol=5e-1)
    assert_allclose(gamma, gamma_expected, atol=5e-1)

    if axis == 1 and not third_party.keras.tensorflow_backend._has_nchw_support():
        pytest.skip("Fused batchnorm with NCHW only supported on GPU.")

    # Test that the fused batch norm was actually called. It's called twice: training or not.
    assert tf.nn.fused_batch_norm.call_count == 1


def test_data_format():
    assert keras.backend.image_data_format() == "channels_first"


pool2d_tests = [
    # channels_last, valid and same, and with different striding.
    ((1, 4, 4, 3), (1, 4, 4, 3), (3, 3), (1, 1), "same", "channels_last", "max"),
    ((1, 4, 4, 3), (1, 2, 2, 3), (3, 3), (2, 2), "same", "channels_last", "average"),
    ((1, 4, 4, 3), (1, 2, 2, 3), (3, 3), (1, 1), "valid", "channels_last", "max"),
    ((1, 4, 4, 3), (1, 1, 1, 3), (3, 3), (2, 2), "valid", "channels_last", "average"),
    # channels_first, valid and same, and with different striding.
    ((1, 3, 4, 4), (1, 3, 4, 4), (3, 3), (1, 1), "same", "channels_first", "max"),
    ((1, 3, 4, 4), (1, 3, 2, 2), (3, 3), (2, 2), "same", "channels_first", "average"),
    ((1, 3, 4, 4), (1, 3, 2, 2), (3, 3), (1, 1), "valid", "channels_first", "max"),
    ((1, 3, 4, 4), (1, 3, 1, 1), (3, 3), (2, 2), "valid", "channels_first", "average"),
]


@pytest.mark.parametrize(
    "shape,expected_shape,pool_size,strides,padding,data_format," "pooling_type",
    pool2d_tests,
)
def test_pool2d(
    shape, expected_shape, pool_size, strides, padding, data_format, pooling_type
):
    """Test the pooling padding and data-format compatibility and shapes.

    These checks are needed, as we have patched it to use explicit symmetric padding.
    """
    if pooling_type == "max":
        layer_class = keras.layers.MaxPooling2D
    elif pooling_type == "average":
        layer_class = keras.layers.AveragePooling2D
    else:
        raise ValueError("Unknown pooling type: %s" % pooling_type)

    inputs = tf.placeholder(shape=shape, dtype=tf.float32)
    x = layer_class(
        pool_size=pool_size, strides=strides, padding=padding, data_format=data_format
    )(inputs)
    sess = keras.backend.get_session()
    out = sess.run(x, feed_dict={inputs: np.zeros(shape)})
    output_shape = out.shape
    assert output_shape == expected_shape


def test_patch_dataset_map():
    class _RandomSeedRecorder:
        def __init__(self):
            self.random_seed = None

        def __call__(self, elt):
            self.random_seed = tf.get_default_graph().seed
            return elt

    seed = 37
    tf.set_random_seed(seed)
    recorder = _RandomSeedRecorder()
    ds = tf.data.Dataset.from_generator(lambda: (yield (0)), (tf.int64))

    # Patch is already applied at this point, but if you comment it out and uncomment below code you
    # can verify that this bug still exists and needs to be patched. If the test fails it may be
    # that TF fixed this and we can remove the patch. Talk to @ehall
    # ds.map(recorder)
    # assert recorder.random_seed is None
    #
    # third_party.keras.tensorflow_backend._patch_dataset_map()

    # Test that existing dataset got patched.
    ds.map(recorder)
    assert recorder.random_seed == seed

    # Test that new dataset also gets patched.
    recorder = _RandomSeedRecorder()
    ds = tf.data.Dataset.from_generator(lambda: (yield (0)), (tf.int64))
    ds.map(recorder)
    assert recorder.random_seed == seed


def test_reproducible_mapping():
    def test():
        tf.reset_default_graph()
        np.random.seed(42)
        tf.set_random_seed(42)
        images = np.random.rand(100, 64, 64, 3).astype(np.float32)

        def mapfn(p):
            return tf.image.random_hue(p, 0.04)

        dataset = tf.data.Dataset.from_tensor_slices(images)
        dataset = dataset.map(mapfn)
        dataset = dataset.batch(32)
        x = dataset.make_one_shot_iterator().get_next()

        with tf.Session() as sess:
            return sess.run(x)

    assert np.allclose(test(), test()), "num_parallel_calls=1 undeterministic"
