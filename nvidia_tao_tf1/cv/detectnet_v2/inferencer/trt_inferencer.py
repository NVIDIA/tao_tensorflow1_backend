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

"""Simple inference handler for maglev trained DetectNet_v2 models serialized to TRT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import struct
import sys

import tempfile
import traceback


import numpy as np

import pycuda.autoinit # noqa pylint: disable=unused-import
import pycuda.driver as cuda

from six.moves import range

try:
    import tensorrt as trt  # noqa pylint: disable=W0611 pylint: disable=W0611
    from nvidia_tao_tf1.cv.detectnet_v2.inferencer.utilities import Calibrator
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Failed to import TensorRT package, exporting TLT to a TensorRT engine "
        "will not be available."
    )

from nvidia_tao_tf1.cv.detectnet_v2.inferencer.base_inferencer import Inferencer
from nvidia_tao_tf1.cv.detectnet_v2.inferencer.utilities import HostDeviceMem
from nvidia_tao_tf1.encoding import encoding

logger = logging.getLogger(__name__)

trt_loggers = []

# TensorRT default params.
DEFAULT_MAX_WORKSPACE_SIZE = 1 << 15  # amounts to 1GB of context space.


def _create_tensorrt_logger(verbose=False):
    """Create a TensorRT logger.

    Args:
        verbose(bool): Flag to set logger as verbose or not.
    Return:
        tensorrt_logger(trt.infer.ConsoleLogger): TensorRT console logger object.
    """
    if verbose:
        trt_verbosity = trt.Logger.INFO
    else:
        trt_verbosity = trt.Logger.WARNING
    tensorrt_logger = trt.Logger(trt_verbosity)
    trt_loggers.append(tensorrt_logger)
    return tensorrt_logger


def _exception_check(check_case, fail_string):
    """Simple function for exception handling and traceback print.

    Args:
        check_case: item to check exception for.
        fail_string (str): String to be printed at traceback error.
    Returns:
        No explicit returns.
    Raises:
        Prints out traceback and raises AssertionError with line number and
        error text.
    """
    try:
        assert check_case
    except AssertionError:
        logger.error("Fail string")
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)  # Fixed format
        tb_info = traceback.extract_tb(tb)
        _, line, _, text = tb_info[-1]
        raise AssertionError('Failed in {} in statement {}'.format(line,
                                                                   text))


class TRTInferencer(Inferencer):
    """Network handler for inference tool."""

    def __init__(self, target_classes=None, framework="tensorrt",
                 image_height=544, image_width=960, image_channels=3,
                 uff_model=None, caffemodel=None,
                 etlt_model=None, etlt_key=None, prototxt=None, parser="caffe",
                 calib_tensorfile=None, n_batches=None, input_nodes=None,
                 output_nodes=None, max_workspace_size=1 << 30,
                 data_type="fp32", calib_file=None, trt_engine=None, gpu_set=0, batch_size=1,
                 save_engine=False, verbose=False):
        """Setting up handler class for tensorrt exported DetectNet_v2 model.

        Args:
            target_classes (list): List of target classes the model will detect.
                This is in order of the network output, and therefore must be taken from
                the costfunction_config of the spec file.
            frameworks (str): The inference backend framework being used.
            image_height (int): Vertical dimension which the model will inference the image.
            image_width (int): Horizontal dimension at which the model will inference the image.
            uff_model (str): Path to the TRT Model uff file.
            caffemodel (str): Path to the caffemodel file for exported Caffe model.
            prototxt (str): Path to the prototxt file for exported Caffe model.
            parser (str): Type of TRT parser to be used.
            calib_tensorfile (str): Path to the calibration tensorfile.
            n_batches (int): No. of batches to calibrate the network when running on int8 mode.
            input_nodes (list): List of input nodes to the graph.
            output_nodes (list): List of output nodes in the graph.
            max_workspace_size (int): Max size of the TRT workspace to be set (Default: 1GB)
            data_type (int): TensorRT backend datatype.
            calib_file (str): Path to save the calibration cache file.
            trt_engine (str): Path to save the TensorRT engine file.
            gpu_set (int): Index of the GPU to be used for inference.
            batch_size (int): Number of images per batch at inference.
            save_engine (bool): Flag to save optimized TensorRT engine or not.
            verbose (bool): Whether or not to log with debug details.

        Returns:
            Initialized TRTInferencer object.
        """
        super(TRTInferencer, self).__init__(target_classes=target_classes,
                                            image_height=image_height,
                                            image_width=image_width,
                                            image_channels=image_channels,
                                            gpu_set=gpu_set,
                                            batch_size=batch_size)
        self.framework = framework
        self._uff_model = uff_model
        self._caffemodel = caffemodel
        self._prototxt = prototxt
        self._etlt_model = etlt_model
        self._etlt_key = etlt_key
        self._parser_kind = parser
        self._trt_logger = _create_tensorrt_logger(verbose)
        self._calib_tensorfile = calib_tensorfile
        self.n_batches = n_batches
        self.max_workspace_size = max_workspace_size
        self._data_type = data_type
        self._calib_file = calib_file
        self._engine_file = trt_engine
        self._save_engine = save_engine
        # Initializing variables that will be used in subsequent steps.
        self.builder = None
        self.calibrator = None
        self.network = None
        self.context = None
        self.runtime = None
        self.stream = None
        self._set_input_output_nodes()
        if self._data_type == "int8":
            # Check if the correct file combinations are present. Either a
            # tensorfile must be present, or a valid cache file.
            check_tensorfile_exists = self._calib_tensorfile is not None and \
                os.path.exists(self._calib_tensorfile)
            check_int8_cache_exists = self._calib_file is not None and \
                os.path.exists(self._calib_file)
            error_string = "Either a valid tensorfile must be present or a cache file."
            assert check_tensorfile_exists or check_int8_cache_exists, error_string

        self.input_dims = (self.num_channels,
                           self.image_height,
                           self.image_width)
        self.constructed = False

    def _set_input_output_nodes(self):
        """Set the input output nodes in the TensorRTInferencer."""
        self.input_node = ["input_1"]
        if self._parser_kind == "caffe":
            self.output_nodes = ["output_bbox", "output_cov/Sigmoid"]
        elif self._parser_kind in ["uff", "etlt"]:
            self.output_nodes = ["output_bbox/BiasAdd", "output_cov/Sigmoid"]
        else:
            raise NotImplementedError("Parser kind not supported.")

    def _platform_compatibility_check(self):
        """Check for builder compatibility.

        Return:
            None:
        Raises:
            AttributeError: Whether configuration is compatible or not.
        """
        if self._dtype == trt.DataType.HALF and not self.builder.platform_has_fast_fp16:
            logger.error("Specified FP16 but not supported on platform.")
            raise AttributeError("Specified FP16 but not supported on platform.")

        if self._dtype == trt.DataType.INT8 and not self.builder.platform_has_fast_int8:
            logger.error("Specified INT8 but not supported on platform.")
            raise AttributeError("Specified INT8 but not supported on platform.")

        if self._dtype == trt.DataType.INT8 and self.calibrator is None:
            logger.error("Specified INT8 but no calibrator provided.")
            raise AttributeError("Specified INT8 but no calibrator provided.")

    def _parse_caffe_model(self):
        """Simple function to parse a caffe model.

        Args:
            None.
        Returns
            None.
        Raises:
            Assertion error for network creation.
        """
        self.parser = trt.CaffeParser()
        assert os.path.isfile(self._caffemodel), "{} not found.".format(self._caffemodel)
        assert os.path.isfile(self._prototxt), "{} not found.".format(self._prototxt)
        self.blob_name_to_tensor = self.parser.parse(self._prototxt,
                                                     self._caffemodel,
                                                     self.network,
                                                     trt.float32)
        _exception_check(self.blob_name_to_tensor,
                         "Failed to parse caffe model")
        # Mark output blobs.
        for l in self.output_nodes:
            logger.info("Marking {} as output layer".format(l))
            t = self.blob_name_to_tensor.find(str(l))
            _exception_check(t, "Failed to find output layer")
            self.network.mark_output(t)

    def _parse_uff_model(self):
        """Simple function to parse a uff model.

        Args:
            None.
        Returns
            None.
        Raises:
            Assertion error for network creation.
        """
        self.parser = trt.UffParser()
        assert os.path.isfile(self._uff_model), "{} not found.".format(self._uff_model)
        # Register input blob
        for blob in self.input_node:
            self.parser.register_input(blob.encode(), self.input_dims)
        # Register the output blobs
        for blob in self.output_nodes:
            self.parser.register_output(blob.encode())

        _exception_check(self.parser.parse(self._uff_model,
                                           self.network,
                                           trt.float32),
                         "Failed to parse UFF model")

    def _parse_etlt_model(self):
        """Simple function to parse an etlt model.

        Args:
            None.
        Returns
            None.
        Raises:
            Assertion error for network creation.
        """
        if not os.path.exists(self._etlt_model):
            raise ValueError("Cannot find etlt file.")
        os_handle, tmp_uff_file = tempfile.mkstemp()
        os.close(os_handle)

        # Unpack etlt file.
        with open(self._etlt_model, "rb") as efile:
            num_chars = efile.read(4)
            num_chars = struct.unpack("<i", num_chars)[0]
            input_node = str(efile.read(num_chars))
            with open(tmp_uff_file, "wb") as tfile:
                encoding.decode(efile, tfile, self._etlt_key.encode())
        self._uff_model = tmp_uff_file
        self._input_node = [input_node]

        # Parse the decoded UFF file.
        self._parse_uff_model()
        os.remove(self._uff_model)
        logger.debug("Parsed ETLT model file.")

    def _set_dtype(self):
        """Simple function to set backend datatype.

        Args:
            None.
        Returns
            None.
        Raises:
            ValueError for unsupported datatype.
        """
        if self._data_type == 'int8':
            self._dtype = trt.int8
        elif self._data_type == 'fp16':
            self._dtype = trt.float16
        elif self._data_type == 'fp32':
            self._dtype = trt.float32
        else:
            raise ValueError("Unsupported data type: %s" % self._data_type)

    def network_init(self):
        """Initializing the keras model and compiling it for inference.

        Args:
            None
        Returns:
            No explicit returns. Defines the self.mdl attribute to the intialized
            keras model.
        """

        # Creating a runtime handler.
        self.runtime = trt.Runtime(self._trt_logger)

        if not os.path.isfile(self._engine_file):
            logger.info("Engine file not found at {}".format(self._engine_file))
            logger.info("Using TensorRT to optimize model and generate an engine.")

            # Set backend tensorrt data type.
            self._set_dtype()

            # Instantiate a builder.
            self.builder = trt.Builder(self._trt_logger)

            self.calibrator = None
            # Set up calibrator
            if self._data_type == "int8":
                logger.info("Initializing int8 calibration table.")
                # TODO:<vpraveen> Update to use custom calibrator when the repo
                # moves to TRT 5.1.
                self.calibrator = Calibrator(self._calib_tensorfile,
                                             self._calib_file,
                                             self.n_batches,
                                             self.batch_size)

            # Check if platform is compatible for the configuration of TRT engine
            # that will be created.
            self._platform_compatibility_check()

            # Instantiate the network.
            self.network = self.builder.create_network()
            builder_config = self.builder.create_builder_config()

            # Parse the model using caffe / uff parser.
            if self._parser_kind == "caffe":
                self._parse_caffe_model()
            elif self._parser_kind == "uff":
                self._parse_uff_model()
            elif self._parser_kind == "etlt":
                self._parse_etlt_model()
            else:
                raise NotImplementedError("{} parser is not supported".format(self._parser_kind))

            # set context information batch size and workspace for trt backend.
            self.builder.max_batch_size = self.batch_size
            builder_config.max_workspace_size = self.max_workspace_size

            # Set fp16 or int 8 mode based on inference.
            if self._dtype == trt.float16:
                builder_config.set_flag(trt.BuilderFlag.FP16)

            # Setting the engine builder to create int8 engine and calibrate the
            # graph.
            if self._dtype == trt.int8:
                logger.debug("Setting trt calibrator")
                builder_config.set_flag(trt.BuilderFlag.INT8)
                builder_config.int8_calibrator = self.calibrator
                # Sometimes TensorRT may choose non int8 implementations of
                # layers for discrete Volta GPU setup since Volta GPU's don't
                # have tensor core. Therefore it may be best to force the build
                # restrictions, to choose int8 kernels for GPU's without
                # int8 tensor cores.
                builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

            # Build tensorrt engine.
            self.engine = self.builder.build_engine(self.network, builder_config)
            logger.debug("Number of bindings {}".format(self.engine.num_bindings))
            logger.debug("TensorRT engine built")

            # Serialize and save the tensorrt engine for future use.
            if self._save_engine:
                logger.info("Saving engine to {} for further use".format(self._engine_file))
                with open(self._engine_file, "wb") as ef:
                    ef.write(self.engine.serialize())
                ef.closed

            del self.builder
            del self.network
        else:
            # Reading from a pre serialized engine file if one exists.
            logger.info("Reading from engine file at: {}".format(self._engine_file))
            with open(self._engine_file, "rb") as ef:
                self.engine = self.runtime.deserialize_cuda_engine(ef.read())
            ef.closed

        # Create an execution context to enqueue operations to.
        self.context = self.engine.create_execution_context()
        logger.debug("Generated TRT execution context.")

        # Create pycuda execution stream.
        self.stream = cuda.Stream()
        self.allocate_buffers()
        self.constructed = True

    def allocate_buffers(self):
        """Simple function to allocate CPU-GPU buffers.

        Engine bindings are interated across and memory buffers are allocated based
        on the binding dimensions.

        Args:
            self(TRTInferencer object): all required arguments are class members.
        Returns:
            No explicit returns.
        """
        self.inputs = []
        self.outputs = []
        self.bindings = []
        for binding in range(self.engine.num_bindings):
            size = self.engine.get_binding_shape(binding)
            npshape = size
            binding_name = self.engine.get_binding_name(binding)
            logger.debug("Binding name: {}, size: {}".format(binding_name,
                                                             trt.volume(size)))
            num_elements = trt.volume(size) * self.batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(num_elements, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem, binding_name, npshape))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem, binding_name, npshape))

    def infer_batch(self, chunk):
        """Function to infer a batch of images using trained keras model.

        Args:
            chunk (array): list of images in the batch to infer.
        Returns:
            infer_out: raw_predictions from model.predict.
            resized: resized size of the batch.
        """
        if not self.constructed:
            raise ValueError("Cannot run inference. Run Inferencer.network_init() first.")

        infer_shape = (self.batch_size,) + (self.num_channels, self.image_height, self.image_width)
        infer_input = np.zeros(infer_shape)

        # Prepare image batches.
        logger.debug("Inferring images")
        for idx, image in enumerate(chunk):
            input_image, resized = self.input_preprocessing(image)
            infer_input[idx, :, :, :] = input_image

        # Infer on image batches.
        logger.debug("Number of input blobs {}".format(len(self.inputs)))

        # copy buffers to GPU.
        np.copyto(self.inputs[0].host, infer_input.ravel())
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        # Enqueue inference context.
        self.context.execute_async(stream_handle=self.stream.handle,
                                   bindings=self.bindings,
                                   batch_size=self.batch_size)

        # Copy inference back from the GPU to host.
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        # Sychronize cuda stream events.
        self.stream.synchronize()

        output = self.get_reshaped_outputs()
        infer_dict = self.predictions_to_dict(output)
        logger.debug("Inferred_outputs: {}".format(len(output)))
        infer_out = self.keras_output_map(infer_dict)
        return infer_out, resized

    def get_reshaped_outputs(self):
        """Function to collate outputs and get results in NCHW formatself.

        Args:
            self(TRTInferencer object): all required arguments are class members.

        Returns:
            output (list): list of reshaped np arrays
        """
        # Collate results.
        output = [out.host for out in self.outputs]
        logger.debug("Number of outputs: {}".format(len(output)))
        for idx, out in enumerate(output):
            logger.debug("Output shape: {}, {}".format(out.shape,
                                                       self.outputs[idx].numpy_shape))
            out_shape = (self.batch_size,) + tuple(self.outputs[idx].numpy_shape)
            output[idx] = np.reshape(output[idx], out_shape)
        logger.debug("Coverage blob shape: {}".format(output[0].shape))
        return output

    def clear_buffers(self):
        """Simple function to free input, output buffers allocated earlier.

        Args:
            No explicit arguments. Inputs and outputs are member variables.
        Returns:
            No explicit returns.
        Raises:
            ValueError if buffers not found.
        """
        # Loop through inputs and free inputs.
        logger.info("Clearing input buffers.")
        for inp in self.inputs:
            inp.device.free()

        # Loop through outputs and free them.
        logger.info("Clearing output buffers.")
        for out in self.outputs:
            out.device.free()

    def clear_trt_session(self):
        """Simple function to free destroy tensorrt handlers.

        Args:
            No explicit arguments. Destroys context, runtime and engine.
        Returns:
            No explicit returns.
        Raises:
            ValueError if buffers not found.
        """
        if self.runtime:
            logger.info("Clearing tensorrt runtime.")
            del self.runtime

        if self.context:
            logger.info("Clearing tensorrt context.")
            del self.context

        if self.engine:
            logger.info("Clearing tensorrt engine.")
            del self.engine

        del self.stream
