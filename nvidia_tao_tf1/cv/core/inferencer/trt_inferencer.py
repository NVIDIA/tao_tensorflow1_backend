# Copyright 2021 NVIDIA Corporation.  All rights reserved.

"""Wrapper class for performing TensorRT inference."""

import logging

import tensorrt as trt
from nvidia_tao_tf1.core.export._tensorrt import Engine

logger = logging.getLogger(__name__)
# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTInferencer(object):
    """TensorRT model inference wrapper."""

    def __init__(self, trt_engine):
        """Initialize the TensorRT model builder.

        Args:
            trt_engine (str or trt.ICudaEngine): trt engine path or
                deserialized trt engine.
        """

        # Initialize runtime needed for loading TensorRT engine from file
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        self.trt_runtime = trt.Runtime(TRT_LOGGER)

        if isinstance(trt_engine, trt.ICudaEngine):
            # It's already an engine
            self.trt_engine_path = None
            self.trt_engine = trt_engine
        else:
            # Assume it's a filepath
            self.trt_engine_path = trt_engine
            # Deserialize the engine
            self.trt_engine = self._load_trt_engine_file(self.trt_engine_path)

        self.engine = Engine(self.trt_engine)

    def _load_trt_engine_file(self, trt_engine_path):
        """Load serialized engine file into memory.

        Args:
            trt_engine_path (str): path to the tensorrt file

        Returns:
            trt_engine (trt.ICudaEngine): deserialized engine
        """
        # Load serialized engine file into memory
        with open(trt_engine_path, "rb") as f:
            trt_engine = self.trt_runtime.deserialize_cuda_engine(f.read())
            logger.info("Loading TensorRT engine: {}".format(trt_engine_path))

        return trt_engine

    def predict(self, input_data):
        """Do inference with TensorRT engine.

        Args:
            input_data (np.ndarray): Inputs to run inference on.

        Returns:
            (dict): dictionary mapping output names to output values.
        """
        return self.engine.infer(input_data)
