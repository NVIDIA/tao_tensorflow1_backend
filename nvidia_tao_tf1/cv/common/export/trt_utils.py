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

"""Utilities for TensorRT related operations."""
# TODO: remove EngineBuilder related code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys
import traceback

try:
    import tensorrt as trt  # noqa pylint: disable=W0611 pylint: disable=W0611
    # Get TensorRT version number.
    [NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, _] = [
        int(item) for item
        in trt.__version__.split(".")
    ]
    trt_available = True
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Failed to import TensorRT package, exporting TLT to a TensorRT engine "
        "will not be available."
    )
    trt_available = False


# Default TensorRT parameters.
DEFAULT_MAX_WORKSPACE_SIZE = 2 * (1 << 30)
DEFAULT_MAX_BATCH_SIZE = 1


# Define logger.
logger = logging.getLogger(__name__)


def _create_tensorrt_logger(verbose=False):
    """Create a TensorRT logger.

    Args:
        verbose(bool): Flag to set logger as verbose or not.
    Return:
        tensorrt_logger(trt.infer.ConsoleLogger): TensorRT console logger object.
    """
    if str(os.getenv('SUPPRES_VERBOSE_LOGGING', '0')) == '1':
        # Do not print any warnings in TLT docker
        trt_verbosity = trt.Logger.Severity.ERROR
    elif verbose:
        trt_verbosity = trt.Logger.INFO
    else:
        trt_verbosity = trt.Logger.WARNING
    tensorrt_logger = trt.Logger(trt_verbosity)
    return tensorrt_logger


def _set_excluded_layer_precision(network, fp32_layer_names, fp16_layer_names):
    """When generating an INT8 model, it sets excluded layers' precision as fp32 or fp16.

    In detail, this function is only used when generating INT8 TensorRT models. It accepts
    two lists of layer names: (1). for the layers in fp32_layer_names, their precision will
    be set as fp32; (2). for those in fp16_layer_names, their precision will be set as fp16.

    Args:
        network: TensorRT network object.
        fp32_layer_names (list): List of layer names. These layers use fp32.
        fp16_layer_names (list): List of layer names. These layers use fp16.
    """
    is_mixed_precision = False
    use_fp16_mode = False

    for i, layer in enumerate(network):
        if any(s in layer.name for s in fp32_layer_names):
            is_mixed_precision = True
            layer.precision = trt.float32
            layer.set_output_type(0, trt.float32)
            logger.info("fp32 index: %d; name: %s", i, layer.name)
        elif any(s in layer.name for s in fp16_layer_names):
            is_mixed_precision = True
            use_fp16_mode = True
            layer.precision = trt.float16
            layer.set_output_type(0, trt.float16)
            logger.info("fp16 index: %d; name: %s", i, layer.name)
        else:
            layer.precision = trt.int8
            layer.set_output_type(0, trt.int8)

    return is_mixed_precision, use_fp16_mode


class EngineBuilder(object):
    """Create a TensorRT engine.

    Args:
        filename (list): List of filenames to load model from.
        max_batch_size (int): Maximum batch size.
        vmax_workspace_size (int): Maximum workspace size.
        dtype (str): data type ('fp32', 'fp16' or 'int8').
        calibrator (:any:`Calibrator`): Calibrator to use for INT8 optimization.
        fp32_layer_names (list): List of layer names. These layers use fp32.
        fp16_layer_names (list): List of layer names. These layers use fp16.
        verbose (bool): Whether to turn on verbose mode.
        tensor_scale_dict (dict): Dictionary mapping names to tensor scaling factors.
        strict_type(bool): Whether or not to apply strict_type_constraints for INT8 mode.
    """

    def __init__(
        self,
        filenames,
        max_batch_size=DEFAULT_MAX_BATCH_SIZE,
        max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
        dtype="fp32",
        calibrator=None,
        fp32_layer_names=None,
        fp16_layer_names=None,
        verbose=False,
        tensor_scale_dict=None,
        strict_type=False,
    ):
        """Initialization routine."""
        if dtype == "int8":
            self._dtype = trt.DataType.INT8
        elif dtype == "fp16":
            self._dtype = trt.DataType.HALF
        elif dtype == "fp32":
            self._dtype = trt.DataType.FLOAT
        else:
            raise ValueError("Unsupported data type: %s" % dtype)
        self._strict_type = strict_type
        if fp32_layer_names is None:
            fp32_layer_names = []
        elif dtype != "int8":
            raise ValueError(
                "FP32 layer precision could be set only when dtype is INT8"
            )

        if fp16_layer_names is None:
            fp16_layer_names = []
        elif dtype != "int8":
            raise ValueError(
                "FP16 layer precision could be set only when dtype is INT8"
            )

        self._fp32_layer_names = fp32_layer_names
        self._fp16_layer_names = fp16_layer_names

        self._tensorrt_logger = _create_tensorrt_logger(verbose)
        builder = trt.Builder(self._tensorrt_logger)
        config = builder.create_builder_config()
        trt.init_libnvinfer_plugins(self._tensorrt_logger, "")
        if self._dtype == trt.DataType.HALF and not builder.platform_has_fast_fp16:
            logger.error("Specified FP16 but not supported on platform.")
            raise AttributeError(
                "Specified FP16 but not supported on platform.")
            return

        if self._dtype == trt.DataType.INT8 and not builder.platform_has_fast_int8:
            logger.error("Specified INT8 but not supported on platform.")
            raise AttributeError(
                "Specified INT8 but not supported on platform.")
            return

        if self._dtype == trt.DataType.INT8:
            if tensor_scale_dict is None and calibrator is None:
                logger.error("Specified INT8 but neither calibrator "
                             "nor tensor_scale_dict is provided.")
                raise AttributeError("Specified INT8 but no calibrator "
                                     "or tensor_scale_dict is provided.")

        network = builder.create_network()

        self._load_from_files(filenames, network)

        builder.max_batch_size = max_batch_size
        config.max_workspace_size = max_workspace_size

        if self._dtype == trt.DataType.HALF:
            config.set_flag(trt.BuilderFlag.FP16)

        if self._dtype == trt.DataType.INT8:
            config.set_flag(trt.BuilderFlag.INT8)
            if tensor_scale_dict is None:
                config.int8_calibrator = calibrator
                # When use mixed precision, for TensorRT builder:
                # strict_type_constraints needs to be True;
                # fp16_mode needs to be True if any layer uses fp16 precision.
                set_strict_types, set_fp16_mode = \
                    _set_excluded_layer_precision(
                        network=network,
                        fp32_layer_names=self._fp32_layer_names,
                        fp16_layer_names=self._fp16_layer_names,
                    )
                if set_strict_types:
                    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
                if set_fp16_mode:
                    config.set_flag(trt.BuilderFlag.FP16)
            else:
                # Discrete Volta GPUs don't have int8 tensor cores. So TensorRT might
                # not pick int8 implementation over fp16 or even fp32 for V100
                # GPUs found on data centers (e.g., AVDC). This will be a discrepancy
                # compared to Turing GPUs including d-GPU of DDPX and also Xavier i-GPU
                # both of which have int8 accelerators. We set the builder to strict
                # mode to avoid picking higher precision implementation even if they are
                # faster.
                if self._strict_type:
                    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
                else:
                    config.set_flag(trt.BuilderFlag.FP16)
                self._set_tensor_dynamic_ranges(
                    network=network, tensor_scale_dict=tensor_scale_dict
                )

        engine = builder.build_engine(network, config)

        try:
            assert engine
        except AssertionError:
            logger.error("Failed to create engine")
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)  # Fixed format
            tb_info = traceback.extract_tb(tb)
            _, line, _, text = tb_info[-1]
            raise AssertionError(
                "Parsing failed on line {} in statement {}".format(
                    line, text)
            )

        self._engine = engine

    def _load_from_files(self, filenames, network):
        """Load an engine from files."""
        raise NotImplementedError()

    @staticmethod
    def _set_tensor_dynamic_ranges(network, tensor_scale_dict):
        """Set the scaling factors obtained from quantization-aware training.

        Args:
            network: TensorRT network object.
            tensor_scale_dict (dict): Dictionary mapping names to tensor scaling factors.
        """
        tensors_found = []
        for idx in range(network.num_inputs):
            input_tensor = network.get_input(idx)
            if input_tensor.name in tensor_scale_dict:
                tensors_found.append(input_tensor.name)
                cal_scale = tensor_scale_dict[input_tensor.name]
                input_tensor.dynamic_range = (-cal_scale, cal_scale)

        for layer in network:
            found_all_outputs = True
            for idx in range(layer.num_outputs):
                output_tensor = layer.get_output(idx)
                if output_tensor.name in tensor_scale_dict:
                    tensors_found.append(output_tensor.name)
                    cal_scale = tensor_scale_dict[output_tensor.name]
                    output_tensor.dynamic_range = (-cal_scale, cal_scale)
                else:
                    found_all_outputs = False
            if found_all_outputs:
                layer.precision = trt.int8
        tensors_in_dict = tensor_scale_dict.keys()
        assert set(tensors_in_dict) == set(tensors_found), (
            "Some of the tensor names specified in tensor "
            "scale dictionary was not found in the network."
        )

    def get_engine(self):
        """Return the engine that was built by the instance."""
        return self._engine


class UFFEngineBuilder(EngineBuilder):
    """Create a TensorRT engine from a UFF file.

    Args:
        filename (str): UFF file to create engine from.
        input_node_name (str): Name of the input node.
        input_dims (list): Dimensions of the input tensor.
        output_node_names (list): Names of the output nodes.
    """

    def __init__(
        self,
        filename,
        input_node_name,
        input_dims,
        output_node_names,
        *args,
        **kwargs
    ):
        """Init routine."""
        self._input_node_name = input_node_name
        if not isinstance(output_node_names, list):
            output_node_names = [output_node_names]
        self._output_node_names = output_node_names
        self._input_dims = input_dims

        super(UFFEngineBuilder, self).__init__([filename], *args, **kwargs)

    def _load_from_files(self, filenames, network):
        filename = filenames[0]
        parser = trt.UffParser()
        for key, value in self._input_dims.items():
            parser.register_input(key, value, trt.UffInputOrder(0))
        for name in self._output_node_names:
            parser.register_output(name)
        try:
            assert parser.parse(filename, network, trt.DataType.FLOAT)
        except AssertionError:
            logger.error("Failed to parse UFF File")
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)  # Fixed format
            tb_info = traceback.extract_tb(tb)
            _, line, _, text = tb_info[-1]
            raise AssertionError(
                "UFF parsing failed on line {} in statement {}".format(line, text)
            )
