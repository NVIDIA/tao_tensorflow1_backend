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
"""TensorRT inference model builder for FasterRCNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
from io import open  # Python 2/3 compatibility.  pylint: disable=W0622
import logging
import os

import numpy as np
import pycuda.autoinit  # noqa pylint: disable=W0611
import pycuda.driver as cuda
import tensorrt as trt


logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')
logger = logging.getLogger(__name__)


BINDING_TO_DTYPE_UFF = {
    "input_image": np.float32, "NMS": np.float32, "NMS_1": np.int32,
}

BINDING_TO_DTYPE_ONNX = {
    "input_image": np.float32, "nms_out": np.float32, "nms_out_1": np.int32,
}


class CacheCalibrator(trt.IInt8EntropyCalibrator2):
    """Calibrator class that loads a cache file directly.

    This inherits from ``trt.IInt8EntropyCalibrator2`` to implement
    the calibration interface that TensorRT needs to calibrate the
    INT8 quantization factors.

    Args:
        calibration_filename (str): name of calibration to read/write to.
    """

    def __init__(self, cache_filename, *args, **kwargs):
        """Init routine."""
        super(CacheCalibrator, self).__init__(*args, **kwargs)
        self._cache_filename = cache_filename

    def get_batch(self, names):
        """Dummy method since we are going to use cache file directly.

        Args:
            names (list): list of memory bindings names.
        """
        return None

    def get_batch_size(self):
        """Return batch size."""
        return 8

    def read_calibration_cache(self):
        """Read calibration from file."""
        if os.path.exists(self._cache_filename):
            with open(self._cache_filename, "rb") as f:
                return f.read()
        else:
            raise ValueError('''Calibration cache file
                                not found: {}'''.format(self._cache_filename))

    def write_calibration_cache(self, cache):
        """Do nothing since we already have cache file.

        Args:
            cache (memoryview): buffer to read calibration data from.
        """
        return


class Engine(object):
    """A class to represent a TensorRT engine.

    This class provides utility functions for performing inference on
    a TensorRT engine.

    Args:
        engine: the CUDA engine to wrap.
    """

    def __init__(self, engine, batch_size, input_width, input_height):
        """Initialization routine."""
        self._engine = engine
        self._context = None
        self._batch_size = batch_size
        self._input_width = input_width
        self._input_height = input_height
        self._is_uff = self._engine.has_implicit_batch_dimension

    @contextlib.contextmanager
    def _create_context(self):
        """Create an execution context and allocate input/output buffers."""
        BINDING_TO_DTYPE = BINDING_TO_DTYPE_UFF if self._is_uff else \
            BINDING_TO_DTYPE_ONNX
        try:
            with self._engine.create_execution_context() as self._context:
                # Create stream and events to measure timings.
                self._stream = cuda.Stream()
                self._start = cuda.Event()
                self._end = cuda.Event()
                self._device_buffers = []
                self._host_buffers = []
                self._input_binding_ids = {}
                if self._is_uff:
                    # make sure the infer batch size is no more than
                    # engine.max_batch_size
                    assert self._batch_size <= self._engine.max_batch_size, (
                        f"Error: inference batch size: {self._batch_size} is larger than "
                        f"engine's max_batch_size: {self._engine.max_batch_size}"
                    )
                infer_batch_size = self._batch_size
                for i in range(self._engine.num_bindings):
                    if len(list(self._engine.get_binding_shape(i))) == 3:
                        dims = trt.Dims3(self._engine.get_binding_shape(i))
                        size = trt.volume(dims)
                        elt_count = size * infer_batch_size
                        target_shape = (infer_batch_size, dims[0], dims[1], dims[2])
                    elif len(list(self._engine.get_binding_shape(i))) == 4:
                        # with explicit batch dim
                        dims = trt.Dims4(self._engine.get_binding_shape(i))
                        elt_count = infer_batch_size * dims[1] * dims[2] * dims[3]
                        target_shape = (infer_batch_size, dims[1], dims[2], dims[3])
                    else:
                        raise ValueError('''Binding shapes can only be 3 or 4,
                                        got {}.'''.format(self._engine.get_binding_shape(i)))
                    binding_name = self._engine.get_binding_name(i)
                    dtype = BINDING_TO_DTYPE[binding_name]
                    if self._engine.binding_is_input(i):
                        self._context.set_optimization_profile_async(0, self._stream.handle)
                        self._context.set_binding_shape(i, target_shape)
                        self._input_binding_ids[binding_name] = i
                        page_locked_mem = None
                    else:
                        page_locked_mem = cuda.pagelocked_empty(elt_count, dtype=dtype)
                        page_locked_mem = page_locked_mem.reshape(*target_shape)
                    # Allocate pagelocked memory.
                    self._host_buffers.append(page_locked_mem)
                    _mem_alloced = cuda.mem_alloc(elt_count * np.dtype(dtype).itemsize)
                    self._device_buffers.append(_mem_alloced)
                if not self._input_binding_ids:
                    raise RuntimeError("No input bindings detected.")
                yield
        finally:
            # Release context and allocated memory.
            self._release_context()

    def _do_infer(self, batch):
        bindings = [int(device_buffer) for device_buffer in self._device_buffers]

        if not isinstance(batch, dict):
            if len(self._input_binding_ids) > 1:
                raise ValueError('''Input node names must be provided in case of multiple
                                  inputs.
                                 Got these inputs: %s''' % self._input_binding_ids.keys())
            # Single input case.
            batch = {list(self._input_binding_ids.keys())[0]: batch}

        batch_sizes = {array.shape[0] for array in batch.values()}
        if len(batch_sizes) != 1:
            raise ValueError('''All arrays must have the same batch size.
                              Got %s.''' % repr(batch_sizes))
        batch_size = batch_sizes.pop()
        assert batch_size == self._batch_size, (
            f"Inference data batch size: {batch_size} is not equal to batch size "
            f"of the input/output buffers: {self._batch_size}."
        )
        # Transfer input data to device.
        for node_name, array in batch.items():
            array = array.astype('float32')
            cuda.memcpy_htod_async(self._device_buffers[self._input_binding_ids[node_name]],
                                   array, self._stream)
        # Execute model.
        self._start.record(self._stream)
        if self._is_uff:
            self._context.execute_async(batch_size, bindings, self._stream.handle, None)
        else:
            self._context.execute_async_v2(bindings, self._stream.handle, None)
        self._end.record(self._stream)
        self._end.synchronize()
        # Transfer predictions back.
        outputs = dict()
        for i in range(self._engine.num_bindings):
            if not self._engine.binding_is_input(i):
                cuda.memcpy_dtoh_async(self._host_buffers[i], self._device_buffers[i],
                                       self._stream)
                out = self._host_buffers[i][:batch_size, ...]
                name = self._engine.get_binding_name(i)
                outputs[name] = out
        # outputs["nms_out"][:, 0, :, 0] is image index, not useful
        denormalize = np.array(
            [self._input_width, self._input_height,
             self._input_width, self._input_height],
            dtype=np.float32
        )
        if self._is_uff:
            nms_out_name = "NMS"
            nms_out_1_name = "NMS_1"
        else:
            nms_out_name = "nms_out"
            nms_out_1_name = "nms_out_1"
        # (x1, y1, x2, y2), shape = (N, 1, R, 4)
        nmsed_boxes = outputs[nms_out_name][:, 0, :, 3:7] * denormalize
        # convert to (y1, x1, y2, x2) to keep consistent with keras model format
        nmsed_boxes = np.take(nmsed_boxes, np.array([1, 0, 3, 2]), axis=2)
        # shape = (N, 1, R, 1)
        nmsed_scores = outputs[nms_out_name][:, 0, :, 2]
        # shape = (N, 1, R, 1)
        nmsed_classes = outputs[nms_out_name][:, 0, :, 1]
        # shape = (N, 1, 1, 1)
        num_dets = outputs[nms_out_1_name][:, 0, 0, 0]
        rois_output = None
        return [nmsed_boxes, nmsed_scores, nmsed_classes, num_dets, rois_output]

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

    def infer(self, batch):
        """Perform inference on a Numpy array.

        Args:
            batch (ndarray): array to perform inference on.
        Returns:
            A dictionary of outputs where keys are output names
            and values are output tensors.
        """
        with self._create_context():
            outputs = self._do_infer(np.ascontiguousarray(batch))
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


class TrtModel(object):
    '''A TensorRT model builder for FasterRCNN model inference based on TensorRT.

    The TensorRT model builder builds a TensorRT engine from the engine file from the
    tlt-converter and do inference in TensorRT. We use this as a way to verify the
    TensorRT inference functionality of the FasterRCNN model.
    '''

    def __init__(self,
                 trt_engine_file,
                 batch_size,
                 input_h,
                 input_w):
        '''Initialize the TensorRT model builder.'''
        self._trt_engine_file = trt_engine_file
        self._batch_size = batch_size
        self._input_w = input_w
        self._input_h = input_h
        self._trt_logger = trt.Logger(trt.Logger.Severity.WARNING)
        trt.init_libnvinfer_plugins(self._trt_logger, "")

    def load_trt_engine_file(self):
        '''load TensorRT engine file generated by tlt-converter.'''
        runtime = trt.Runtime(self._trt_logger)
        with open(self._trt_engine_file, 'rb') as f:
            _engine = f.read()
            logger.info("Loading existing TensorRT engine and "
                        "ignoring the specified batch size and data type"
                        " information in spec file.")
            self.engine = Engine(runtime.deserialize_cuda_engine(_engine),
                                 self._batch_size,
                                 self._input_w,
                                 self._input_h)

    def build_or_load_trt_engine(self):
        '''Build engine or load engine depends on whether a trt engine is available.'''
        if self._trt_engine_file is not None:
            # load engine
            logger.info('''Loading TensorRT engine file: {}
                        for inference.'''.format(self._trt_engine_file))
            self.load_trt_engine_file()
        else:
            raise ValueError('''A TensorRT engine file should
                              be provided for TensorRT based inference.''')

    def predict(self, batch):
        '''Do inference with TensorRT engine.'''
        return self.engine.infer(batch)
