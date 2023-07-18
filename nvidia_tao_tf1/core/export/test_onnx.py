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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

"""Root logger for export app."""
logger = logging.getLogger(__name__)  # noqa

# import numpy as np
# import pytest

try:
    import tensorrt as trt
except ImportError:
    logger.warning(
        "Failed to import TRT package. TRT inference testing will not be available."
    )
    trt = None

# from nvidia_tao_tf1.core.export._tensorrt import Engine, ONNXEngineBuilder

MNIST_ONNX_FILE = "./nvidia_tao_tf1/core/export/data/mnist.onnx"


class TestOnnx(object):
    """Test ONNX export to TensorRT."""

    def test_parser(self):
        """Test parsing an ONNX model."""
        trt_verbosity = trt.Logger.Severity.INFO
        tensorrt_logger = trt.Logger(trt_verbosity)
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(
            tensorrt_logger
        ) as builder, builder.create_network(explicit_batch) as network:
            with trt.OnnxParser(network, tensorrt_logger) as parser:
                with open(MNIST_ONNX_FILE, "rb") as model:
                    parser.parse(model.read())

    # comment out these tests before we can create an onnx file with explicit batch
    # walk around as these is failing with TRT 7.0 explicit batch
    # def test_engine_builder(self):
    #     """Test inference on an ONNX model."""
    #     builder = ONNXEngineBuilder(MNIST_ONNX_FILE, verbose=True)
    #     engine = Engine(builder.get_engine())
    #     output = engine.infer(np.zeros((2, 1, 28, 28)))
    #     assert output["Plus214_Output_0"].shape == (2, 10)

    # def test_engine_builder_fp16(self):
    #     """Test inference on an ONNX model in FP16 mode."""
    #     try:
    #         builder = ONNXEngineBuilder(MNIST_ONNX_FILE, verbose=True, dtype="fp16")
    #     except AttributeError as e:
    #         if "FP16 but not supported" in str(e):
    #             pytest.skip("FP16 not supported on platform.")
    #     engine = Engine(builder.get_engine())
    #     output = engine.infer(np.zeros((2, 1, 28, 28)))
    #     assert output["Plus214_Output_0"].shape == (2, 10)
