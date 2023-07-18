import gc
import logging
import os

import keras
from keras import backend as K
from keras.layers import Concatenate, Conv2D, Dense, Flatten, Input
from keras.models import Model

from numba import cuda
import numpy as np
import pytest

from nvidia_tao_tf1.blocks.models import KerasModel
from nvidia_tao_tf1.core.export._tensorrt import Engine, ONNXEngineBuilder, UFFEngineBuilder
from nvidia_tao_tf1.cv.common.utilities.tlt_utils import encode_from_keras, load_model
from nvidia_tao_tf1.cv.core.export.base_exporter import BaseExporter

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
    level='INFO'
)


class DummyModel(KerasModel):
    """Dummy model for tests."""

    def _build_dummy_model(self, dummy_tensor):
        """Build Dummy model for testing purposes.

        Args:
            dummy_tensor (tensor): Input tensor to model
        Returns:
            x_3 (tensor): Dummy model output.
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
        """Build Dummy Model.

        Args:
            key (str): Key to decode/encode model.
            dummy_input (tensor): Dummy input for model.
        Returns
            outputs from model.
        """
        model_name = 'DummyNet'
        dummy_tensor = Input(tensor=dummy_input, name='dummy_input')
        dummy_output = self._build_dummy_model(dummy_tensor)
        model = Model(inputs=[dummy_tensor], outputs=[dummy_output], name=model_name)

        self._keras_model = model

        return self._keras_model.outputs

    def save_model(self, file_name, enc_key='test', encrypt=True):
        """Save Dummy Model.

        Args:
            file_name (str): File to save dummy model.
            enc_key (str): Key to encode model.
        """
        if encrypt:
            encode_from_keras(
                self._keras_model,
                file_name,
                bytes(enc_key, 'utf-8'))
        else:
            self._keras_model.save(file_name)


class ExporterTest(BaseExporter):
    """Exporter class for testing purposes."""

    def __init__(self,
                 model_path=None,
                 key='test',
                 data_type='int8',
                 backend='tfonnx',
                 strict_type=False,
                 data_format='channels_first'):
        """Instantiate exporter for testing.

        Args:
            model_path (str): Path to dummy model file.
            key (str): Key to decode model.
            data_type (str): Backend data-type for the optimized TensorRT engine.
            strict_type (bool): Apply TensorRT strict_type_constraints or not for INT8 mode.
            backend (str): Type of intermediate backend parser to be instantiated.
        """
        super(ExporterTest, self).__init__(model_path=model_path,
                                           key=key,
                                           data_type=data_type,
                                           backend=backend,
                                           strict_type=strict_type,
                                           data_format=data_format)
        keras.backend.set_image_data_format(data_format)

    def export_to_etlt(self, output_filename, target_opset=10):
        """Function to export model to etlt.

        Args:
            output_filename (str): Output .etlt filename
            target_opset (int): Target opset version to use for onnx conversion.
        Returns:
            output_onnx_filename (str): Temporary unencrypted file
            in_tensor_names (list): List of input tensor names
            out_tensor_names (list): List of output tensor names
        """
        keras.backend.set_learning_phase(0)
        model = load_model(self.model_path,
                           key=self.key)

        output_node_names = ['dense_1/BiasAdd']

        tmp_file_name, in_tensor_names, out_tensor_names = self.save_exported_file(
            model,
            output_filename,
            output_node_names=output_node_names,
            target_opset=target_opset,
            delete_tmp_file=False
        )
        del model
        # del model
        gc.collect()
        cuda.close()

        return tmp_file_name, in_tensor_names, out_tensor_names

    def export(self,
               input_dims,
               output_filename,
               backend,
               calibration_cache='',
               data_file_name='',
               n_batches=1,
               batch_size=1,
               verbose=True,
               target_opset=10,
               calibration_images_dir='',
               save_engine=True,
               engine_file_name='',
               max_workspace_size=1 << 30,
               max_batch_size=1,
               opt_batch_size=1,
               static_batch_size=1,
               save_unencrypted_model=False,
               validate_trt_engine=True,
               tmp_file_name='',
               in_tensor_name='',
               out_tensor_name='',
               tmp_dir=''
               ):
        """Export.

        Args:
        ETLT export
            input_dims (list): Input dims with channels_first(CHW) or channels_last (HWC)
            output_filename (str): Output .etlt filename
            backend (str): Model type to export to

        Calibration and TRT export
            calibration_cache (str): Calibration cache file to write to or read from.
            data_file_name (str): Tensorfile to run calibration for int8 optimization
            n_batches (int): Number of batches to calibrate over
            batch_size (int): Number of images per batch
            verbose (bool): Verbosity of the logger
            target_opset (int): Target opset version to use for onnx conversion.
            calibration_images_dir (str): Directory of images to run int8 calibration if
                data file is unavailable.
            save_engine (bool): If True, saves trt engine file to `engine_file_name`
            engine_file_name (str): Output trt engine file
            max_workspace_size (int): Max size of workspace to be set for trt engine builder.
            max_batch_size (int): Max batch size for trt engine builder
            opt_batch_size (int): Optimum batch size to use for model conversion.
                Default is 1.
            static_batch_size (int): Set a static batch size for exported etlt model.
                Default is -1(dynamic batch size)
            save_unencrypted_model (bool): Option on whether to save an encrypted model or not.
            validated_trt_engine (bool): Option to validate trt engine.
            tmp_file_name (str): Temporary file name to use.
            in_tensor_name (str): Input tensor name to the model.
            out_tensor_name (str): Output tensor name to the model.
            tmp_dir (str): Pytests temporally directory. Used only for int8 export.
        """
        # Get int8 calibrator.
        calibrator = None
        max_batch_size = max(batch_size, max_batch_size)
        data_format = self.data_format
        input_dims = (1, 256, 256)

        if self.backend == 'tfonnx':
            backend = 'onnx'
        preprocessing_params = {'scale': [0.5], 'means': [0.5], 'flip_channel': False}
        keras.backend.clear_session()
        if self.data_type == 'int8':
            calibration_cache = os.path.join(tmp_dir, 'calibration')
            calibration_data = os.path.join(tmp_dir, 'calibration2')
            calibrator = self.get_calibrator(
                calibration_cache=calibration_cache,
                data_file_name=calibration_data,
                n_batches=n_batches,
                batch_size=batch_size,
                input_dims=input_dims,
                calibration_images_dir='nvidia_tao_tf1/cv/core/export/images/',
                preprocessing_params=preprocessing_params
            )
        if backend == 'onnx':
            engine_builder = ONNXEngineBuilder(tmp_file_name,
                                               max_batch_size=max_batch_size,
                                               max_workspace_size=max_workspace_size,
                                               dtype=self.data_type,
                                               strict_type=self.strict_type,
                                               verbose=verbose,
                                               calibrator=calibrator,
                                               tensor_scale_dict=self.tensor_scale_dict,
                                               dynamic_batch=True,
                                               input_dims=None,
                                               opt_batch_size=opt_batch_size)
        elif backend == 'uff':
            engine_builder = UFFEngineBuilder(tmp_file_name,
                                              in_tensor_name,
                                              max_batch_size=max_batch_size,
                                              max_workspace_size=max_workspace_size,
                                              dtype=self.data_type,
                                              strict_type=self.strict_type,
                                              verbose=verbose,
                                              calibrator=calibrator,
                                              tensor_scale_dict=self.tensor_scale_dict,
                                              data_format=data_format)
        else:
            raise NotImplementedError("Invalid backend")

        trt_engine = engine_builder.get_engine()
        if save_engine:
            with open(engine_file_name, 'wb') as outf:
                outf.write(trt_engine.serialize())
        if validate_trt_engine:
            try:
                engine = Engine(trt_engine)
                dummy_input = np.ones((1,) + input_dims)
                trt_output = engine.infer(dummy_input)
                logger.info('TRT engine outputs: {}'.format(trt_output.keys()))
                for output_name in trt_output.keys():
                    out = trt_output[output_name]
                    logger.info('{}: {}'.format(output_name, out.shape))
            except Exception as error:
                logger.error('TRT engine validation error!')
                logger.error(error)
        if trt_engine:
            del trt_engine

@pytest.mark.parametrize(
    "encrypt_model",
    [False]
)
def test_export(tmpdir, encrypt_model):
    '''Function to test model exports.

    Args:
        tmpdir (str): Pytests temporary directory.

    Returns:
    '''
    key = 'test'
    if encrypt_model:
        model_filename = "model.tlt"
    else:
        model_filename = "model.hdf5"
    model_path = os.path.join(tmpdir, model_filename)
    model = DummyModel()
    dummy_input = np.random.randn(1, 1, 256, 256)
    dummy_input = K.constant(dummy_input)
    model.build(key, dummy_input)
    model.save_model(model_path, key, encrypt=encrypt_model)

    exporter = ExporterTest(model_path,
                            key=key,
                            backend='tfonnx',
                            data_type='int8')

    tmp_file_name, in_tensor_name, out_tensor_name = exporter.export_to_etlt(
        model_path, target_opset=10)

    # Test ONNX export.
    onnx_output_path = os.path.join(tmpdir, 'output_onnx')
    onnx_engine_file_name = onnx_output_path
    exporter.backend = 'onnx'
    exporter.data_type = 'fp32'

    exporter.export(input_dims=None,
                    output_filename=onnx_output_path,
                    backend='onnx',
                    target_opset=10,
                    tmp_file_name=tmp_file_name,
                    in_tensor_name=in_tensor_name,
                    out_tensor_name=out_tensor_name,
                    engine_file_name=onnx_engine_file_name)

    assert os.path.isfile(onnx_output_path)

    # Test UFF export.
    uff_output_path = os.path.join(tmpdir, 'output_uff')
    uff_engine_file_name = uff_output_path
    exporter.backend = 'uff'

    exporter.export(input_dims=None,
                    output_filename=uff_output_path,
                    backend='onnx',
                    engine_file_name=uff_engine_file_name,
                    target_opset=10,
                    tmp_file_name=tmp_file_name,
                    in_tensor_name=in_tensor_name,
                    out_tensor_name=out_tensor_name)

    assert os.path.isfile(uff_output_path)

    # Test int8 export.
    int_eight_output_path = os.path.join(tmpdir, 'int_eight_output')

    exporter.export(input_dims=None,
                    output_filename=int_eight_output_path,
                    backend='onnx',
                    target_opset=10,
                    tmp_file_name=tmp_file_name,
                    in_tensor_name=in_tensor_name,
                    out_tensor_name=out_tensor_name,
                    engine_file_name=int_eight_output_path,
                    tmp_dir=tmpdir)

    assert os.path.isfile(int_eight_output_path)
