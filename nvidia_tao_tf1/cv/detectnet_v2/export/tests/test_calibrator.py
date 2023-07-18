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

"""Test detectnet exporter to generate model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import numpy as np
import pycuda.autoinit  # noqa pylint: disable=unused-import
import pycuda.driver as cuda
import pytest
import tensorrt as trt

from nvidia_tao_tf1.cv.detectnet_v2.export.detectnet_calibrator import DetectNetCalibrator
from nvidia_tao_tf1.cv.detectnet_v2.inferencer.utilities import HostDeviceMem
from nvidia_tao_tf1.cv.detectnet_v2.spec_handler.spec_loader import load_experiment_spec

detectnet_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))
training_spec = os.path.join(detectnet_root,
                             "experiment_specs/default_spec.txt")
topologies = [
    (1, 2),  # case 1.
    (10, 6)
]


class TestCalibrator(object):
    """Simple class to test the int8 calibrator."""

    def _setup_calibrator_instance(self, n_batches, batch_size):
        """Simple function to instantiate a Detectnetv2 calibrator."""
        self._experiment_spec = load_experiment_spec(
            training_spec,
            merge_from_default=False,
            validation_schema="train_val")
        os_handle, calibration_cachefile = tempfile.mkstemp(suffix=".bin")
        os.close(os_handle)
        self.calibrator = DetectNetCalibrator(
            self._experiment_spec,
            calibration_cachefile,
            n_batches,
            batch_size)
        self._batch_count = 0

    def allocate_io_memory(self, n_batches, batch_size):
        assert hasattr(self._experiment_spec, "augmentation_config"), (
            "Augmentation config is required to get the data shape."
        )
        preprocessing = self._experiment_spec.augmentation_config.preprocessing
        input_shape = (
            batch_size,
            preprocessing.output_image_height,
            preprocessing.output_image_width,
            preprocessing.output_image_channel
        )
        num_elements = input_shape[0] * input_shape[1] * \
            input_shape[2] * input_shape[3]

        # Set up array to receive tf data.
        self.tf_data = cuda.pagelocked_empty(
            num_elements, trt.nptype(trt.float32)
        )

        # Set up arrays to mimic tensorrt data in the GPU.
        host_mem = cuda.pagelocked_empty(
            num_elements, trt.nptype(trt.float32)
        )
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        self.tensorrt_data = HostDeviceMem(
            host_mem, device_mem,
            "tensorrt_input", input_shape)

    def common(self, n_batches):
        """Common function to yield batches and check tensorrt cuda transfer and back."""
        while self._batch_count < n_batches:
            self._batch_count += 1
            batch = self.calibrator.get_data_from_source()
            np.copyto(self.tf_data, batch.ravel())

            # Copy data from host to device
            cuda.memcpy_htod(self.tensorrt_data.device, self.tf_data)

            # Copy data from device to host
            cuda.memcpy_dtoh(self.tensorrt_data.host, self.tensorrt_data.device)

            data_under_test = np.reshape(self.tensorrt_data.host, batch.shape)
            assert np.array_equal(batch, data_under_test), (
                "The roundtrip from CPU to GPU and back failed."
            )

    # first test case
    @pytest.mark.skipif(os.getenv("RUN_ON_CI", "0") == "1", reason="Cannot be run on CI")
    @pytest.mark.parametrize(
        "n_batches, batch_size",
        topologies
    )
    def test_calibrator(self,
                        n_batches,
                        batch_size):
        self._setup_calibrator_instance(n_batches, batch_size)
        assert self.calibrator, (
            "Calibrator was not created."
        )
        self.allocate_io_memory(n_batches, batch_size)
        try:
            self.common(n_batches)
        finally:
            # Freeing up the GPU memory at the end of the
            # test, irrespective of whether it passes or fails.
            self.tensorrt_data.device.free()
