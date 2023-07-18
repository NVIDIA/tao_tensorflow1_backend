# Copyright 2019-2020 NVIDIA Corporation.  All rights reserved.
"""Helper functions for loading engine."""
import numpy as np
import pycuda.autoinit  # noqa pylint: disable=unused-import
import pycuda.driver as cuda
import tensorrt as trt


class HostDeviceMem(object):
    """Simple helper data class that's a little nice to use than a 2-tuple."""

    def __init__(self, host_mem, device_mem):
        """Init function."""
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        """___str___."""
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        """___repr___."""
        return self.__str__()


def do_inference(context, bindings, inputs,
                 outputs, stream, batch_size=1,
                 execute_v2=False):
    """Generalization for multiple inputs/outputs.

    inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    # Run inference.
    if execute_v2:
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    else:
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def allocate_buffers(engine, context=None):
    """Allocates host and device buffer for TRT engine inference.

    This function is similair to the one in common.py, but
    converts network outputs (which are np.float32) appropriately
    before writing them to Python buffer. This is needed, since
    TensorRT plugins doesn't support output type description, and
    in our particular case, we use NMS plugin as network output.

    Args:
        engine (trt.ICudaEngine): TensorRT engine
        context (trt.IExecutionContext): Context for dynamic shape engine

    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: buffer to device bindings
        stream (cuda.Stream): cuda stream for engine inference synchronization
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    # Current NMS implementation in TRT only supports DataType.FLOAT but
    # it may change in the future, which could brake this sample here
    # when using lower precision [e.g. NMS output would not be np.float32
    # anymore, even though this is assumed in binding_to_type]
    binding_to_type = {"Input": np.float32, "NMS": np.float32, "NMS_1": np.int32,
                       "BatchedNMS": np.int32, "BatchedNMS_1": np.float32,
                       "BatchedNMS_2": np.float32, "BatchedNMS_3": np.float32,
                       "generate_detections": np.float32,
                       "mask_head/mask_fcn_logits/BiasAdd": np.float32,
                       "softmax_1": np.float32,
                       "input_1": np.float32}

    for binding in engine:
        if context:
            binding_id = engine.get_binding_index(str(binding))
            size = trt.volume(context.get_binding_shape(binding_id))
        else:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        # avoid error when bind to a number (YOLO BatchedNMS)
        size = engine.max_batch_size if size == 0 else size
        if str(binding) in binding_to_type:
            dtype = binding_to_type[str(binding)]
        else:
            dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def load_engine(trt_runtime, engine_path):
    """Helper funtion to load an exported engine."""
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine
