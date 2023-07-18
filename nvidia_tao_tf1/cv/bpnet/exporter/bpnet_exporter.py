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

"""Bpnet exporter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import logging
import os
from shutil import copyfile

import keras
try:
    import tensorrt  # noqa pylint: disable=W0611 pylint: disable=W0611
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Failed to import TensorRT package, exporting TLT to a TensorRT engine "
        "will not be available."
    )
from numba import cuda
import numpy as np

try:
    from nvidia_tao_tf1.core.export._tensorrt import Engine, ONNXEngineBuilder, UFFEngineBuilder
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Cannot import TensorRT packages, exporting to TLT to a TensorRT engine "
        "will not be available."
    )
from nvidia_tao_tf1.cv.bpnet.utils import export_utils
from nvidia_tao_tf1.cv.common.utilities import path_processing as io_utils
from nvidia_tao_tf1.cv.core.export.base_exporter import BaseExporter

logger = logging.getLogger(__name__)


class BpNetExporter(BaseExporter):
    """Exporter class to export a trained BpNet model."""

    def __init__(self,
                 model_path=None,
                 key=None,
                 data_type="fp32",
                 strict_type=False,
                 experiment_spec_path="",
                 backend="onnx",
                 data_format="channels_last"):
        """Instantiate the BpNet exporter to export etlt model.

        Args:
            model_path(str): Path to the BpNet model file.
            key (str): Key to decode the model.
            data_type (str): Backend data-type for the optimized TensorRT engine.
            experiment_spec_path (str): Path to BpNet experiment spec file.
            backend (str): Type of intermediate backend parser to be instantiated.
        """
        super(BpNetExporter, self).__init__(model_path=model_path,
                                            key=key,
                                            data_type=data_type,
                                            strict_type=strict_type,
                                            backend=backend,
                                            data_format=data_format)

        self.experiment_spec_path = experiment_spec_path
        self.experiment_spec = io_utils.load_yaml_file(experiment_spec_path)

        # Set keras backend format
        keras.backend.set_image_data_format(data_format)

        # Reformat preprocessing params to use with preprocessing block in
        # BaseExporter
        self.preprocessing_params = self._reformat_data_preprocessing_parameters(
            self.experiment_spec['dataloader']['normalization_params'])

    def _reformat_data_preprocessing_parameters(self, normalization_params=None):
        """Reformat normalization params to be consumed by pre-processing block.

        Args:
            normalization_params (dict): Normalization params used for training

        Returns:
            preprocessing_params (dict): Preprocessing parameters including mean, scale
                and flip_channel keys.
        """

        if normalization_params is None:
            logger.warning("Using default normalization params!")
            means = [0.5, 0.5, 0.5]
            scale = [256.0, 256.0, 256.0]
        else:
            means = normalization_params['image_offset']
            scale = normalization_params['image_scale']

        # Reformat as expected by preprocessing funtion.
        means = np.array(means) * np.array(scale)
        scale = 1.0 / np.array(scale)
        # Network is trained in RGB format
        flip_channel = False

        preprocessing_params = {"scale": scale,
                                "means": means,
                                "flip_channel": flip_channel}
        return preprocessing_params

    def export_to_etlt(self, output_filename, sdk_compatible_model=False, upsample_ratio=4.0):
        """Function to export model to etlt.

        Args:
            output_filename (str): Output .etlt filename
            sdk_compatible_model (bool): Generate SDK (TLT CV Infer/DS/IX) compatible model
            upsample_ratio (int): [NMS][CustomLayers] Upsampling factor.

        Returns:
            tmp_file_name (str): Temporary unencrypted file
            in_tensor_names (list): List of input tensor names
            out_tensor_names (list): List of output tensor names
        """
        # Load Keras model from file.
        model = self.load_model(self.model_path)
        # model.summary() # Disable for TLT release
        model, custom_objects = export_utils.update_model(
            model, sdk_compatible_model=sdk_compatible_model, upsample_ratio=upsample_ratio)

        output_filename, in_tensor_names, out_tensor_names = self.save_exported_file(
            model,
            output_filename,
            custom_objects=custom_objects,
            delete_tmp_file=False,
            target_opset=10
        )

        # Trigger garbage collector to clear memory of the deleted loaded model
        del model
        gc.collect()
        # NOTE: sometimes cuda doesn't release the GPU memory even after graph
        # is cleared. So explicitly clear GPU memory.
        cuda.close()

        return output_filename, in_tensor_names, out_tensor_names

    def export(self,
               input_dims,
               output_filename,
               backend,
               calibration_cache="",
               data_file_name="",
               n_batches=1,
               batch_size=1,
               verbose=True,
               calibration_images_dir="",
               save_engine=False,
               engine_file_name="",
               max_workspace_size=1 << 30,
               max_batch_size=1,
               force_ptq=False,
               static_batch_size=None,
               sdk_compatible_model=False,
               upsample_ratio=4.0,
               save_unencrypted_model=False,
               validate_trt_engine=False,
               opt_batch_size=1):
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
            calibration_images_dir (str): Directory of images to run int8 calibration if
                data file is unavailable.
            save_engine (bool): If True, saves trt engine file to `engine_file_name`
            engine_file_name (str): Output trt engine file
            max_workspace_size (int): Max size of workspace to be set for trt engine builder.
            max_batch_size (int): Max batch size for trt engine builder
            force_ptq (bool): Flag to force post training quantization for QAT models
            static_batch_size (int): Set a static batch size for exported etlt model.
                Default is -1(dynamic batch size)
            opt_batch_size (int): Optimum batch size to use for model conversion.

        Deployment model
            sdk_compatible_model (bool): Generate SDK (TLT CV Infer/DS/IX) compatible model
            upsample_ratio (int): [NMS][CustomLayers] Upsampling factor.

        Debugging
            save_unencrypted_model (bool): Flag to save unencrypted model (debug purpose)
            validate_trt_engine (bool): Flag to enable trt engine execution for validation.
        """
        if force_ptq:
            print("BpNet doesn't support QAT. Post training quantization is used by default.")

        # set dynamic_batch flag
        dynamic_batch = bool(static_batch_size <= 0)

        _, in_tensor_name, out_tensor_names = self.export_to_etlt(
            output_filename,
            sdk_compatible_model=sdk_compatible_model,
            upsample_ratio=upsample_ratio
        )

        # Get int8 calibrator
        calibrator = None
        max_batch_size = max(batch_size, max_batch_size)
        data_format = self.data_format
        preprocessing_params = self.preprocessing_params
        input_dims = tuple(input_dims)
        logger.debug("Input dims: {}".format(input_dims))
        if self.backend == "tfonnx":
            backend = "onnx"

        keras.backend.clear_session()
        if self.data_type == "int8":
            # no tensor scale, take traditional INT8 calibration approach
            # use calibrator to generate calibration cache
            calibrator = self.get_calibrator(calibration_cache=calibration_cache,
                                             data_file_name=data_file_name,
                                             n_batches=n_batches,
                                             batch_size=batch_size,
                                             input_dims=input_dims,
                                             calibration_images_dir=calibration_images_dir,
                                             preprocessing_params=preprocessing_params)
            logger.info("Calibration takes time especially if number of batches is large.")

        # Assuming single input node graph for uff engine creation.
        if not isinstance(input_dims, dict):
            input_dims_dict = {in_tensor_name: input_dims}

        # Verify with engine generation / run calibration.
        if backend == "uff":
            engine_builder = UFFEngineBuilder(output_filename,
                                              in_tensor_name,
                                              input_dims_dict,
                                              out_tensor_names,
                                              max_batch_size=max_batch_size,
                                              max_workspace_size=max_workspace_size,
                                              dtype=self.data_type,
                                              strict_type=self.strict_type,
                                              verbose=verbose,
                                              calibrator=calibrator,
                                              tensor_scale_dict=self.tensor_scale_dict,
                                              data_format=data_format)
        elif backend == "onnx":
            tensor_scale_dict = None if force_ptq else self.tensor_scale_dict
            engine_builder = ONNXEngineBuilder(output_filename,
                                               max_batch_size=max_batch_size,
                                               max_workspace_size=max_workspace_size,
                                               dtype=self.data_type,
                                               strict_type=self.strict_type,
                                               verbose=verbose,
                                               calibrator=calibrator,
                                               tensor_scale_dict=tensor_scale_dict,
                                               dynamic_batch=dynamic_batch,
                                               input_dims=input_dims_dict,
                                               opt_batch_size=opt_batch_size)
        else:
            raise NotImplementedError("Invalid backend.")

        trt_engine = engine_builder.get_engine()
        if save_engine:
            with open(engine_file_name, "wb") as outf:
                outf.write(trt_engine.serialize())
        if validate_trt_engine:
            try:
                engine = Engine(trt_engine)
                dummy_input = np.ones((1,) + input_dims)
                trt_output = engine.infer(dummy_input)
                logger.info("TRT engine outputs: {}".format(trt_output.keys()))
                for output_name in trt_output.keys():
                    out = trt_output[output_name]
                    logger.info("{}: {}".format(output_name, out.shape))
            except Exception:
                logger.error("TRT engine validation error!")
        if trt_engine:
            del trt_engine
