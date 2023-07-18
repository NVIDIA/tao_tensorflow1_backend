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
"""Modulus INT8 calibration APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
from io import open  # Python 2/3 compatibility.  pylint: disable=W0622
import logging
import os
import sys
import tempfile
import traceback

import numpy as np

from nvidia_tao_tf1.core.decorators import override, subclass
from nvidia_tao_tf1.core.export._onnx import keras_to_onnx
from nvidia_tao_tf1.core.export._uff import keras_to_uff
from nvidia_tao_tf1.core.export.caffe import keras_to_caffe
from nvidia_tao_tf1.core.export.data import TensorFile

"""Logger for data export APIs."""
logger = logging.getLogger(__name__)
try:
    import pycuda.autoinit  # noqa pylint: disable=W0611
    import pycuda.driver as cuda
    import tensorrt as trt
except ImportError:
    # TODO(xiangbok): we should probably do this test in modulus/export/__init__.py.
    logger.warning(
        "Failed to import TRT and/or CUDA. TensorRT optimization "
        "and inference will not be available."
    )

DEFAULT_MAX_WORKSPACE_SIZE = 1 << 30
DEFAULT_MAX_BATCH_SIZE = 100
DEFAULT_MIN_BATCH_SIZE = 1
DEFAULT_OPT_BATCH_SIZE = 100

# Array of TensorRT loggers. We need to keep global references to
# the TensorRT loggers that we create to prevent them from being
# garbage collected as those are referenced from C++ code without
# Python knowing about it.
tensorrt_loggers = []

# If we were unable to load TensorRT packages because TensorRT is not installed
# then we will stub the exported API.
if "trt" not in globals():
    keras_to_tensorrt = None
    load_tensorrt_engine = None
else:
    # We were able to load TensorRT package so we are implementing the API
    def _create_tensorrt_logger(verbose=False):
        """Create a TensorRT logger.

        Args:
            verbose (bool): whether to make the logger verbose.
        """
        if str(os.getenv('SUPPRES_VERBOSE_LOGGING', '0')) == '1':
            # Do not print any warnings in TLT docker
            trt_verbosity = trt.Logger.Severity.ERROR
        elif str(os.getenv('SUPPRES_VERBOSE_LOGGING', '0')) == '0':
            trt_verbosity = trt.Logger.Severity.INFO
        elif verbose:
            trt_verbosity = trt.Logger.Severity.VERBOSE
        else:
            trt_verbosity = trt.Logger.Severity.WARNING
        tensorrt_logger = trt.Logger(trt_verbosity)
        tensorrt_loggers.append(tensorrt_logger)
        return tensorrt_logger

    class Calibrator(trt.IInt8EntropyCalibrator2):
        """Calibrator class.

        This inherits from ``trt.IInt8EntropyCalibrator2`` to implement
        the calibration interface that TensorRT needs to calibrate the
        INT8 quantization factors.

        Args:
            data_filename (str): ``TensorFile`` data file to use.
            calibration_filename (str): Name of calibration to read/write to.
            n_batches (int): Number of batches for calibrate for.
            batch_size (int): Batch size to use for calibration (this must be
                smaller or equal to the batch size of the provided data).
        """

        def __init__(
            self, data_filename, cache_filename, n_batches, batch_size, *args, **kwargs
        ):
            """Init routine."""
            super(Calibrator, self).__init__(*args, **kwargs)

            self._data_file = TensorFile(data_filename, "r")
            self._cache_filename = cache_filename

            self._batch_size = batch_size
            self._n_batches = n_batches

            self._batch_count = 0
            self._data_mem = None

        def get_batch(self, names):
            """Return one batch.

            Args:
                names (list): list of memory bindings names.
            """
            if self._batch_count < self._n_batches:
                batch = np.array(self._data_file.read())
                if batch is not None:
                    if batch.shape[0] < self._batch_size:
                        raise ValueError(
                            "Data file batch size (%d) < request batch size (%d)"
                            % (batch.shape[0], self._batch_size)
                        )
                    batch = batch[: self._batch_size]

                    if self._data_mem is None:
                        self._data_mem = cuda.mem_alloc(
                            batch.size * 4
                        )  # 4 bytes per float32.

                    self._batch_count += 1

                    # Transfer input data to device.
                    cuda.memcpy_htod(
                        self._data_mem, np.ascontiguousarray(batch, dtype=np.float32)
                    )

                    return [int(self._data_mem)]

            self._data_mem.free()
            return None

        def get_batch_size(self):
            """Return batch size."""
            return self._batch_size

        def read_calibration_cache(self):
            """Read calibration from file."""
            logger.debug("read_calibration_cache - no-op")
            if os.path.isfile(self._cache_filename):
                logger.warning(
                    "Calibration file %s exists but is being "
                    "ignored." % self._cache_filename
                )

        def write_calibration_cache(self, cache):
            """Write calibration to file.

            Args:
                cache (memoryview): buffer to read calibration data from.
            """
            logger.info(
                "Saving calibration cache (size %d) to %s",
                len(cache),
                self._cache_filename,
            )
            with open(self._cache_filename, "wb") as f:
                f.write(cache)

    class Engine(object):
        """A class to represent a TensorRT engine.

        This class provides utility functions for performing inference on
        a TensorRT engine.

        Args:
            engine: CUDA engine to wrap.
            forward_time_ema_decay (float): Decay factor for smoothing the calculation of
                                            forward time. By default, no smoothing is applied.
        """

        def __init__(self, engine, forward_time_ema_decay=0.0):
            """Initialization routine."""
            self._engine = engine
            self._context = None
            self._forward_time = None
            self._forward_time_ema_decay = forward_time_ema_decay

        @contextlib.contextmanager
        def _create_context(self):
            """Create an execution context and allocate input/output buffers."""
            try:
                with self._engine.create_execution_context() as self._context:
                    self._device_buffers = []
                    self._host_buffers = []
                    self._input_binding_ids = {}
                    max_batch_size = self._engine.max_batch_size
                    for i in range(self._engine.num_bindings):
                        shape = self._engine.get_binding_shape(i)
                        if len(shape) == 3:
                            size = trt.volume(shape)
                            elt_count = size * max_batch_size
                            output_shape = (max_batch_size, shape[0], shape[1], shape[2])
                        elif len(shape) == 4 and shape[0] not in [-1, None]:
                            # explicit batch
                            elt_count = shape[0] * shape[1] * shape[2] * shape[3]
                            output_shape = shape
                        elif len(shape) == 2:
                            elt_count = shape[0] * shape[1] * max_batch_size
                            output_shape = (max_batch_size, shape[0], shape[1])
                        elif len(shape) == 1:
                            elt_count = shape[0] * max_batch_size
                            output_shape = (max_batch_size, shape[0])
                        else:
                            raise ValueError("Unhandled shape: {}".format(str(shape)))
                        if self._engine.binding_is_input(i):
                            binding_name = self._engine.get_binding_name(i)
                            self._input_binding_ids[binding_name] = i
                            page_locked_mem = None
                        else:
                            page_locked_mem = cuda.pagelocked_empty(
                                elt_count, dtype=np.float32
                            )
                            page_locked_mem = page_locked_mem.reshape(*output_shape)
                        # Allocate pagelocked memory.
                        self._host_buffers.append(page_locked_mem)
                        self._device_buffers.append(
                            cuda.mem_alloc(elt_count * np.dtype(np.float32).itemsize)
                        )
                    if not self._input_binding_ids:
                        raise RuntimeError("No input bindings detected.")
                    # Create stream and events to measure timings.
                    self._stream = cuda.Stream()
                    self._start = cuda.Event()
                    self._end = cuda.Event()
                    yield
            finally:
                # Release context and allocated memory.
                self._release_context()

        def _do_infer(self, batch):
            bindings = [int(device_buffer) for device_buffer in self._device_buffers]

            if not isinstance(batch, dict):
                if len(self._input_binding_ids) > 1:
                    raise ValueError(
                        "Input node names must be provided in case of multiple "
                        "inputs. "
                        "Got these inputs: %s" % self._input_binding_ids.keys()
                    )
                # Single input case.
                batch = {list(self._input_binding_ids.keys())[0]: batch}

            batch_sizes = {array.shape[0] for array in batch.values()}
            if len(batch_sizes) != 1:
                raise ValueError(
                    "All arrays must have the same batch size. "
                    "Got %s." % repr(batch_sizes)
                )
            batch_size = batch_sizes.pop()

            if (
                self._engine.has_implicit_batch_dimension and
                batch_size > self._engine.max_batch_size
            ):
                raise ValueError(
                    "Batch size (%d) > max batch size (%d)"
                    % (batch_size, self._engine.max_batch_size)
                )

            # Transfer input data to device.
            for node_name, array in batch.items():
                array = array.astype("float32")
                cuda.memcpy_htod_async(
                    self._device_buffers[self._input_binding_ids[node_name]],
                    array,
                    self._stream,
                )
            # Execute model.
            self._start.record(self._stream)
            if self._engine.has_implicit_batch_dimension:
                # UFF
                self._context.execute_async(batch_size, bindings, self._stream.handle, None)
            else:
                # ONNX
                self._context.execute_async_v2(bindings, self._stream.handle, None)
            self._end.record(self._stream)
            self._end.synchronize()
            elapsed_ms_per_batch = self._end.time_since(self._start)
            elapsed_ms_per_sample = elapsed_ms_per_batch / batch_size
            logger.debug(
                "Elapsed time: %.3fms, %.4fms/sample.",
                elapsed_ms_per_batch,
                elapsed_ms_per_sample,
            )
            # CUDA time_since returns durations in milliseconds.
            elapsed_time_per_sample = 1e-3 * elapsed_ms_per_sample
            if self._forward_time is None:
                self._forward_time = elapsed_time_per_sample
            else:
                a = self._forward_time_ema_decay
                self._forward_time = (
                    1 - a
                ) * elapsed_time_per_sample + a * self._forward_time

            # Transfer predictions back.
            outputs = {}
            for i in range(self._engine.num_bindings):
                if not self._engine.binding_is_input(i):
                    # Using a synchronous memcpy here to ensure outputs are ready
                    # for consumption by caller upon returning from this call.
                    cuda.memcpy_dtoh(self._host_buffers[i], self._device_buffers[i])
                    out = self._host_buffers[i][:batch_size]
                    name = self._engine.get_binding_name(i)
                    outputs[name] = out
            return outputs

        def _release_context(self):
            """Release context and allocated memory."""
            for device_buffer in self._device_buffers:
                device_buffer.free()
                del (device_buffer)

            for host_buffer in self._host_buffers:
                del (host_buffer)

            del (self._start)
            del (self._end)
            del (self._stream)

        def get_forward_time(self):
            """Return the inference duration.

            The duration is calculated at the CUDA level and excludes
            data loading and post-processing.

            If a decay factor is specified in the constructor,
            the returned value is smoothed with an exponential moving
            average.

            The returned value is expressed in seconds.
            """
            return self._forward_time

        def infer(self, batch):
            """Perform inference on a Numpy array.

            Args:
                batch (ndarray): array to perform inference on.
            Returns:
                A dictionary of outputs where keys are output names
                and values are output tensors.
            """
            with self._create_context():
                outputs = self._do_infer(batch)
            return outputs

        def infer_iterator(self, iterator):
            """Perform inference on an iterator of Numpy arrays.

            This method should be preferred to ``infer`` when performing
            inference on multiple Numpy arrays since this will re-use
            the allocated execution and memory.

            Args:
                iterator: an iterator that yields Numpy arrays.
            Yields:
                A dictionary of outputs where keys are output names
                and values are output tensors, for each array returned
                by the iterator.
            Returns:
                None.
            """
            with self._create_context():
                for batch in iterator:
                    outputs = self._do_infer(batch)
                    yield outputs

        def save(self, filename):
            """Save serialized engine into specified file.

            Args:
                filename (str): name of file to save engine to.
            """
            with open(filename, "wb") as outf:
                outf.write(self._engine.serialize())

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
                pass
                # # To ensure int8 optimization is not done for shape layer
                # if (not layer.get_output(0).is_shape_tensor):
                #     layer.precision = trt.int8
                #     layer.set_output_type(0, trt.int8)

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
                raise AttributeError("Specified FP16 but not supported on platform.")
                return

            if self._dtype == trt.DataType.INT8 and not builder.platform_has_fast_int8:
                logger.error("Specified INT8 but not supported on platform.")
                raise AttributeError("Specified INT8 but not supported on platform.")
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
                    "Parsing failed on line {} in statement {}".format(line, text)
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

            tensors_in_network = []
            for layer in network:
                found_all_outputs = True
                for idx in range(layer.num_outputs):
                    output_tensor = layer.get_output(idx)
                    tensors_in_network.append(output_tensor.name)
                    if output_tensor.name in tensor_scale_dict:
                        tensors_found.append(output_tensor.name)
                        cal_scale = tensor_scale_dict[output_tensor.name]
                        output_tensor.dynamic_range = (-cal_scale, cal_scale)
                    else:
                        found_all_outputs = False
                if found_all_outputs:
                    layer.precision = trt.int8
            tensors_in_dict = tensor_scale_dict.keys()
            if set(tensors_in_dict) != set(tensors_found):
                logger.debug(
                    "Tensors in scale dictionary but not in network: {}".format(
                        set(tensors_in_dict) - set(tensors_found)
                    )
                )
                logger.debug(
                    "Tensors in the network but not in scale dictionary: {}".format(
                        set(tensors_in_network) - set(tensors_found)
                    )
                )

        def get_engine(self):
            """Return the engine that was built by the instance."""
            return self._engine

    @subclass
    class CaffeEngineBuilder(EngineBuilder):
        """Create a TensorRT engine from Caffe proto and model files.

        Args:
            prototxt_filename (str): Caffe model definition.
            caffemodel_filename (str): Caffe model snapshot.
            input_node_name (str): Name of the input node.
            input_dims (list): Dimensions of the input tensor.
            output_node_names (list): Names of the output nodes.
        """

        def __init__(
            self,
            prototxt_filename,
            caffemodel_filename,
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

            super(CaffeEngineBuilder, self).__init__(
                [prototxt_filename, caffemodel_filename], *args, **kwargs
            )

        @override
        def _load_from_files(self, filenames, network):
            """Parse a Caffe model."""
            parser = trt.CaffeParser()
            prototxt_filename, caffemodel_filename = filenames
            blob_name_to_tensor = parser.parse(
                prototxt_filename, caffemodel_filename, network, trt.DataType.FLOAT
            )

            try:
                assert blob_name_to_tensor
            except AssertionError:
                logger.error("Failed to parse caffe model")
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)
                tb_info = traceback.extract_tb(tb)
                _, line, _, text = tb_info[-1]
                raise AssertionError(
                    "Caffe parsing failed on line {} in statement {}".format(line, text)
                )

            # Mark the outputs.
            for l in self._output_node_names:
                logger.info("Marking " + l + " as output layer")
                t = blob_name_to_tensor.find(str(l))
                try:
                    assert t
                except AssertionError:
                    logger.error("Failed to find output layer {}".format(l))
                    _, _, tb = sys.exc_info()
                    traceback.print_tb(tb)
                    tb_info = traceback.extract_tb(tb)
                    _, line, _, text = tb_info[-1]
                    raise AssertionError(
                        "Caffe parsing failed on line {} in statement {}".format(
                            line, text
                        )
                    )
                network.mark_output(t)

    @subclass
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
            data_format="channels_first",
            **kwargs
        ):
            """Init routine."""
            self._input_node_name = input_node_name
            if not isinstance(output_node_names, list):
                output_node_names = [output_node_names]
            self._output_node_names = output_node_names
            self._input_dims = input_dims
            self._data_format = data_format

            super(UFFEngineBuilder, self).__init__([filename], *args, **kwargs)

        @override
        def _load_from_files(self, filenames, network):
            filename = filenames[0]
            parser = trt.UffParser()
            for key, value in self._input_dims.items():
                if self._data_format == "channels_first":
                    parser.register_input(key, value, trt.UffInputOrder(0))
                else:
                    parser.register_input(key, value, trt.UffInputOrder(1))
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

    @subclass
    class ONNXEngineBuilder(EngineBuilder):
        """Create a TensorRT engine from an ONNX file.

        Args:
            filename (str): ONNX file to create engine from.
            input_node_name (str): Name of the input node.
            input_dims (list): Dimensions of the input tensor.
            output_node_names (list): Names of the output nodes.
        """

        @override
        def __init__(
            self,
            filenames,
            max_batch_size=DEFAULT_MAX_BATCH_SIZE,
            min_batch_size=DEFAULT_MIN_BATCH_SIZE,
            max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
            opt_batch_size=DEFAULT_OPT_BATCH_SIZE,
            dtype="fp32",
            calibrator=None,
            fp32_layer_names=None,
            fp16_layer_names=None,
            verbose=False,
            tensor_scale_dict=None,
            dynamic_batch=False,
            strict_type=False,
            input_dims=None,
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
            self._strict_type = strict_type
            self._tensorrt_logger = _create_tensorrt_logger(verbose)
            builder = trt.Builder(self._tensorrt_logger)

            if self._dtype == trt.DataType.HALF and not builder.platform_has_fast_fp16:
                logger.error("Specified FP16 but not supported on platform.")
                raise AttributeError("Specified FP16 but not supported on platform.")
                return

            if self._dtype == trt.DataType.INT8 and not builder.platform_has_fast_int8:
                logger.error("Specified INT8 but not supported on platform.")
                raise AttributeError("Specified INT8 but not supported on platform.")
                return

            if self._dtype == trt.DataType.INT8:
                if tensor_scale_dict is None and calibrator is None:
                    logger.error("Specified INT8 but neither calibrator "
                                 "nor tensor_scale_dict is provided.")
                    raise AttributeError("Specified INT8 but no calibrator "
                                         "or tensor_scale_dict is provided.")

            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

            self._load_from_files([filenames], network)

            config = builder.create_builder_config()
            if dynamic_batch:
                opt_profile = builder.create_optimization_profile()
                model_input = network.get_input(0)
                input_shape = model_input.shape
                input_name = model_input.name
                # If input_dims is provided, use this shape instead of model shape
                # NOTE: This is to handle fully convolutional models with -1
                # for height and width.
                if input_dims is not None:
                    if input_name in input_dims.keys():
                        input_shape[1] = input_dims[input_name][0]
                        input_shape[2] = input_dims[input_name][1]
                        input_shape[3] = input_dims[input_name][2]
                    else:
                        raise ValueError("Input name not present in"
                                         "the provided input_dims!")
                real_shape_min = (min_batch_size, input_shape[1],
                                  input_shape[2], input_shape[3])
                real_shape_opt = (opt_batch_size, input_shape[1],
                                  input_shape[2], input_shape[3])
                real_shape_max = (max_batch_size, input_shape[1],
                                  input_shape[2], input_shape[3])
                opt_profile.set_shape(input=input_name,
                                      min=real_shape_min,
                                      opt=real_shape_opt,
                                      max=real_shape_max)
                config.add_optimization_profile(opt_profile)
            config.max_workspace_size = max_workspace_size

            if self._dtype == trt.DataType.HALF:
                config.flags |= 1 << int(trt.BuilderFlag.FP16)

            if self._dtype == trt.DataType.INT8:
                config.flags |= 1 << int(trt.BuilderFlag.INT8)
                if tensor_scale_dict is None:
                    config.int8_calibrator = calibrator
                    # When use mixed precision, for TensorRT builder:
                    # strict_type_constraints needs to be True;
                    # fp16_mode needs to be True if any layer uses fp16 precision.
                    strict_type_constraints, fp16_mode = \
                        _set_excluded_layer_precision(
                            network=network,
                            fp32_layer_names=self._fp32_layer_names,
                            fp16_layer_names=self._fp16_layer_names,
                        )
                    if strict_type_constraints:
                        config.flags |= 1 << int(trt.BuilderFlag.STRICT_TYPES)
                    if fp16_mode:
                        config.flags |= 1 << int(trt.BuilderFlag.FP16)
                else:
                    # Discrete Volta GPUs don't have int8 tensor cores. So TensorRT might
                    # not pick int8 implementation over fp16 or even fp32 for V100
                    # GPUs found on data centers (e.g., AVDC). This will be a discrepancy
                    # compared to Turing GPUs including d-GPU of DDPX and also Xavier i-GPU
                    # both of which have int8 accelerators. We set the builder to strict
                    # mode to avoid picking higher precision implementation even if they are
                    # faster.
                    if self._strict_type:
                        config.flags |= 1 << int(trt.BuilderFlag.STRICT_TYPES)
                    else:
                        config.flags |= 1 << int(trt.BuilderFlag.FP16)
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
                    "Parsing failed on line {} in statement {}".format(line, text)
                )

            self._engine = engine

        @override
        def _load_from_files(self, filenames, network):
            filename = filenames[0]
            parser = trt.OnnxParser(network, self._tensorrt_logger)
            with open(filename, "rb") as model_file:
                ret = parser.parse(model_file.read())
            for index in range(parser.num_errors):
                print(parser.get_error(index))
            assert ret, 'ONNX parser failed to parse the model.'

            # Note: there might be an issue when running inference on TRT:
            # [TensorRT] ERROR: Network must have at least one output.
            # See https://github.com/NVIDIA/TensorRT/issues/183.
            # Just keep a note in case we have this issue again.

    def keras_to_tensorrt(
        model,
        input_dims,
        output_node_names=None,
        dtype="fp32",
        max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
        max_batch_size=DEFAULT_MAX_BATCH_SIZE,
        calibration_data_filename=None,
        calibration_cache_filename=None,
        calibration_n_batches=16,
        calibration_batch_size=16,
        fp32_layer_names=None,
        fp16_layer_names=None,
        parser="uff",
        verbose=False,
        custom_objects=None,
        tensor_scale_dict=None,
    ):
        """Create a TensorRT engine out of a Keras model.

        NOTE: the current Keras session is cleared in this function.
        Do not use this function during training.

        Args:
            model (Model): Keras model to export.
            output_filename (str): File to write exported model to.
            in_dims (list or dict): List of input dimensions, or a dictionary of
                input_node_name:input_dims pairs in the case of multiple inputs.
            output_node_names (list of str): List of model output node names as
                returned by model.layers[some_idx].get_output_at(0).name.split(':')[0].
                If not provided, then the last layer is assumed to be the output node.
            max_workspace_size (int): Maximum TensorRT workspace size.
            max_batch_size (int): Maximum TensorRT batch size.
            calibration_data_filename (str): Calibratio data file to use.
            calibration_cache_filename (str): Calibration cache file to write to.
            calibration_n_batches (int): Number of calibration batches.
            calibration_batch_size (int): Calibration batch size.
            fp32_layer_names (list): Fp32 layers names. It is useful only when dtype is int8.
            fp16_layer_names (list): Fp16 layers names. It is useful only when dtype is int8.
            parser='uff' (str): Parser ('uff' or 'caffe') to use for intermediate representation.
            verbose (bool): Whether to turn ON verbose messages.
            custom_objects (dict): Dictionary mapping names (strings) to custom
                classes or functions to be considered during deserialization for export.
            tensor_scale_dict (dict): Dictionary mapping names to tensor scaling factors.
        Returns:
            The names of the input and output nodes. These must be
            passed to the TensorRT optimization tool to identify
            input and output blobs. If multiple output nodes are specified,
            then a list of output node names is returned.
        """
        if dtype == "int8":
            if calibration_data_filename is None:
                raise ValueError(
                    "A calibration data file must be provided for INT8 export."
                )
            calibrator = Calibrator(
                data_filename=calibration_data_filename,
                cache_filename=calibration_cache_filename,
                n_batches=calibration_n_batches,
                batch_size=calibration_batch_size,
            )
        else:
            calibrator = None

        # Custom keras objects are only supported with UFF parser.
        if custom_objects is not None:
            assert (
                parser == "uff"
            ), "Custom keras objects are only supported with UFF parser."

        if parser == "uff":
            # First, convert model to UFF.
            os_handle, tmp_uff_filename = tempfile.mkstemp(suffix=".uff")
            os.close(os_handle)

            input_node_name, output_node_names, _ = keras_to_uff(
                model,
                tmp_uff_filename,
                output_node_names,
                custom_objects=custom_objects,
            )

            if not isinstance(input_dims, dict):
                input_dims = {input_node_name: input_dims}

            logger.info("Model output names: %s", str(output_node_names))

            builder = UFFEngineBuilder(
                tmp_uff_filename,
                input_node_name,
                input_dims,
                output_node_names,
                max_batch_size=max_batch_size,
                max_workspace_size=max_workspace_size,
                dtype=dtype,
                fp32_layer_names=fp32_layer_names,
                fp16_layer_names=fp16_layer_names,
                verbose=verbose,
                calibrator=calibrator,
                tensor_scale_dict=tensor_scale_dict,
            )
            # Delete temp file.
            os.remove(tmp_uff_filename)
        elif parser == "caffe":
            # First, convert to Caffe.
            os_handle, tmp_proto_filename = tempfile.mkstemp(suffix=".prototxt")
            os.close(os_handle)
            os_handle, tmp_caffemodel_filename = tempfile.mkstemp(suffix=".caffemodel")
            os.close(os_handle)

            input_node_name, output_node_names = keras_to_caffe(
                model, tmp_proto_filename, tmp_caffemodel_filename, output_node_names
            )

            builder = CaffeEngineBuilder(
                tmp_proto_filename,
                tmp_caffemodel_filename,
                input_node_name,
                input_dims,
                output_node_names,
                max_batch_size=max_batch_size,
                max_workspace_size=max_workspace_size,
                dtype=dtype,
                verbose=verbose,
                calibrator=calibrator,
                tensor_scale_dict=tensor_scale_dict,
            )

            # Delete temp files.
            os.remove(tmp_proto_filename)
            os.remove(tmp_caffemodel_filename)
        elif parser == "onnx":
            # First, convert model to ONNX.
            os_handle, tmp_onnx_filename = tempfile.mkstemp(suffix=".onnx")
            os.close(os_handle)

            input_node_name, output_node_names, _ = keras_to_onnx(
                model, tmp_onnx_filename, custom_objects=custom_objects,
                target_opset=12
            )

            if not isinstance(input_dims, dict):
                input_dims = {input_node_name: input_dims}

            logger.info("Model output names: %s", str(output_node_names))

            builder = ONNXEngineBuilder(
                tmp_onnx_filename,
                max_batch_size=max_batch_size,
                max_workspace_size=max_workspace_size,
                dtype=dtype,
                fp32_layer_names=fp32_layer_names,
                fp16_layer_names=fp16_layer_names,
                verbose=verbose,
                calibrator=calibrator,
                tensor_scale_dict=tensor_scale_dict,
            )
            # Delete temp file.
            os.remove(tmp_onnx_filename)
        else:
            raise ValueError("Unknown parser: %s" % parser)

        engine = Engine(builder.get_engine())

        return input_node_name, output_node_names, engine

    def load_tensorrt_engine(filename, verbose=False):
        """Load a serialized TensorRT engine.

        Args:
            filename (str): Path to the serialized engine.
            verbose (bool): Whether to turn ON verbose mode.
        """
        tensorrt_logger = _create_tensorrt_logger(verbose)

        if not os.path.isfile(filename):
            raise ValueError("File does not exist")

        with trt.Runtime(tensorrt_logger) as runtime, open(filename, "rb") as inpf:
            tensorrt_engine = runtime.deserialize_cuda_engine(inpf.read())

        engine = Engine(tensorrt_engine)

        return engine
