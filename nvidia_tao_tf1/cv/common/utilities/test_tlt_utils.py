import os

from keras import backend as K
from keras.layers import Concatenate, Conv2D, Dense, Flatten, Input
from keras.models import Model

from nvidia_tao_tf1.blocks.models import KerasModel

import numpy as np

from nvidia_tao_tf1.cv.common.utilities import tlt_utils


class DummyModel(KerasModel):
    """Dummy model for tests."""

    def _build_dummy_model(self, dummy_tensor):
        """Build dummy model.

        Args:
            dummy_tensor (tensor): Dummy tensor.

        Returns:
            x_3 (tensor): Model output.
        """
        x_1_1 = Conv2D(32,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       data_format='channels_first',
                       name='layer-1-1')(dummy_tensor)

        x_2_1 = Conv2D(32,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       data_format='channels_first',
                       name='layer-2-1')(dummy_tensor)

        x_2 = Concatenate(axis=1)([x_1_1, x_2_1])

        x_2_flatten = Flatten(data_format='channels_first')(x_2)

        x_3 = Dense(10)(x_2_flatten)

        return x_3

    def build(self, key, dummy_input):
        """Build dummy model.

        Args:
            key (str): Encode / decode model.
            dummy_input (tensor): Input to model.

        Returns:
            keras model.
        """
        model_name = 'DummyNet'
        dummy_tensor = Input(tensor=dummy_input, name='dummy_input')
        dummy_output = self._build_dummy_model(dummy_tensor)
        model = Model(inputs=[dummy_tensor], outputs=[dummy_output], name=model_name)

        self._keras_model = model

        return self._keras_model


def test_onnx_export(tmpdir):
    """Test onnx export."""
    dummy_model = DummyModel()

    key = "test"
    dummy_input = np.random.randn(1, 3, 72, 72)
    dummy_input = K.constant(dummy_input)

    model = dummy_model.build(key, dummy_input)

    # Test save_exported_file() using onnx as backend.
    output_file_name_onnx_backend = os.path.join(tmpdir, 'test_onnx_backend.tlt')

    tlt_utils.save_exported_file(
        model,
        output_file_name=output_file_name_onnx_backend,
        key=key,
        backend='onnx')

    assert os.path.isfile(output_file_name_onnx_backend)


def test_uff_export(tmpdir):
    """Test UFF export."""
    dummy_model = DummyModel()

    key = "test"
    dummy_input = np.random.randn(1, 3, 72, 72)
    dummy_input = K.constant(dummy_input)

    model = dummy_model.build(key, dummy_input)

    # Test save_exported_file() using uff as backend.
    output_file_name_uff_backend = os.path.join(tmpdir, 'test_uff_backend.tlt')

    tlt_utils.save_exported_file(
        model,
        output_file_name_uff_backend,
        key=key,
        backend='uff')

    assert os.path.isfile(output_file_name_uff_backend)


def test_tfonnx_export(tmpdir):
    """Test tfonnx export."""
    dummy_model = DummyModel()

    key = "test"
    dummy_input = np.random.randn(1, 3, 72, 72)
    dummy_input = K.constant(dummy_input)

    model = dummy_model.build(key, dummy_input)

    # Test save_exported_file() using tfonnx as backend.
    output_file_name_tfonnx_backend = os.path.join(tmpdir, 'test_tfonnx_backend.tlt')

    tlt_utils.save_exported_file(
        model,
        output_file_name_tfonnx_backend,
        key=key,
        backend='tfonnx')

    assert os.path.isfile(output_file_name_tfonnx_backend)
