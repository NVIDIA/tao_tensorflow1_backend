# Copyright 2019-2020 NVIDIA Corporation.  All rights reserved.
"""Utility class for performing TensorRT image inference."""

import numpy as np
import tensorrt as trt

from nvidia_tao_tf1.cv.common.inferencer.engine import allocate_buffers, do_inference, load_engine

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTInferencer(object):
    """Manages TensorRT objects for model inference."""

    def __init__(self, trt_engine_path, input_shape=None, batch_size=None):
        """Initializes TensorRT objects needed for model inference.

        Args:
            trt_engine_path (str): path where TensorRT engine should be stored
            input_shape (tuple): (batch, channel, height, width) for dynamic shape engine
            batch_size (int): batch size for dynamic shape engine
        """

        # We first load all custom plugins shipped with TensorRT,
        # some of them will be needed during inference
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')

        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        self.trt_engine = load_engine(self.trt_runtime, trt_engine_path)
        self.max_batch_size = self.trt_engine.max_batch_size
        self.execute_v2 = False
        # Execution context is needed for inference
        self.context = None

        # Allocate memory for multiple usage [e.g. multiple batch inference]
        self._input_shape = []
        for binding in range(self.trt_engine.num_bindings):
            if self.trt_engine.binding_is_input(binding):
                self._input_shape = self.trt_engine.get_binding_shape(binding)[-3:]
        assert len(self._input_shape) == 3, "Engine doesn't have valid input dimensions"

        # set binding_shape for dynamic input
        if (input_shape is not None) or (batch_size is not None):
            self.context = self.trt_engine.create_execution_context()
            if input_shape is not None:
                self.context.set_binding_shape(0, input_shape)
                self.max_batch_size = input_shape[0]
            else:
                self.context.set_binding_shape(0, [batch_size] + list(self._input_shape))
                self.max_batch_size = batch_size
            self.execute_v2 = True

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.trt_engine,
                                                                                 self.context)

        if self.context is None:
            self.context = self.trt_engine.create_execution_context()

        input_volume = trt.volume(self._input_shape)
        self.numpy_array = np.zeros((self.max_batch_size, input_volume))

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
        for inp in self.inputs:
            inp.device.free()

        # Loop through outputs and free them.
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
        if self.trt_runtime:
            del self.trt_runtime

        if self.context:
            del self.context

        if self.trt_engine:
            del self.trt_engine

        if self.stream:
            del self.stream

    def infer_batch(self, imgs):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """

        # Verify if the supplied batch size is not too big
        max_batch_size = self.max_batch_size
        actual_batch_size = len(imgs)
        if actual_batch_size > max_batch_size:
            raise ValueError("image_paths list bigger ({}) than \
                engine max batch size ({})".format(actual_batch_size, max_batch_size))

        self.numpy_array[:actual_batch_size] = imgs.reshape(actual_batch_size, -1)
        # ...copy them into appropriate place into memory...
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, self.numpy_array.ravel())

        # ...fetch model outputs...
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=max_batch_size,
            execute_v2=self.execute_v2)

        # ...and return results up to the actual batch size.
        return [i.reshape(max_batch_size, -1)[:actual_batch_size] for i in results]

    def __del__(self):
        """Clear things up on object deletion."""

        # Clear session and buffer
        self.clear_trt_session()
        self.clear_buffers()
