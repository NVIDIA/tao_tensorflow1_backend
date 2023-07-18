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

"""DetectNet_v2 calibrator class for TensorRT INT8 Calibration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import pycuda.autoinit  # noqa pylint: disable=unused-import
import pycuda.driver as cuda
import tensorflow as tf

# Simple helper class for calibration.
from nvidia_tao_tf1.cv.common.export.base_calibrator import BaseCalibrator
from nvidia_tao_tf1.cv.detectnet_v2.common.graph import get_init_ops
# Import Detectnet_v2 dataloader.
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.build_dataloader import build_dataloader

logger = logging.getLogger(__name__)


class DetectNetCalibrator(BaseCalibrator):
    """Detectnet_v2 calibrator class."""

    def __init__(self, experiment_spec, cache_filename,
                 n_batches, batch_size,
                 *args, **kwargs):
        """Init routine.

        This inherits from ``nvidia_tao_tf1.cv.common.export.base_calibrator.BaseCalibrator``
        to implement the calibration interface that TensorRT needs to
        calibrate the INT8 quantization factors. The data source here is assumed
        to be the data tensors that are yielded from the DetectNet_v2 dataloader.

        Args:
            data_filename (str): ``TensorFile`` data file to use.
            cache_filename (str): name of calibration file to read/write to.
            n_batches (int): number of batches for calibrate for.
            batch_size (int): batch size to use for calibration (this must be
                smaller or equal to the batch size of the provided data).
        """
        super(DetectNetCalibrator, self).__init__(
            cache_filename,
            n_batches, batch_size,
            *args, **kwargs
        )
        self._data_source = None
        # Instantiate the dataloader.
        self.instantiate_data_source(experiment_spec)
        # Configure tensorflow before running tensorrt.
        self.set_session()

    def set_session(self):
        """Simple function to set the tensorflow session."""
        # Setting this to minimize the default allocation at import.
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.33,
            allow_growth=True)
        # Configuring tensorflow to use CPU so that is doesn't interfere
        # with tensorrt.
        device_count = {'GPU': 0, 'CPU': 1}
        session_config = tf.compat.v1.ConfigProto(
            gpu_options=gpu_options,
            device_count=device_count
        )
        self.session = tf.compat.v1.Session(
            config=session_config,
            graph=tf.get_default_graph()
        )
        self.session.run(get_init_ops())

    def instantiate_data_source(self, experiment_spec):
        """Simple function to instantiate the data_source of the dataloader.

        Args:
            experiment_spec (nvidia_tao_tf1.cv.detectnet_v2.proto.experiment_pb2): Detectnet_v2
                experiment spec proto object.

        Returns:
            No explicit returns.
        """
        if not (hasattr(experiment_spec, 'dataset_config') or
                hasattr(experiment_spec, 'augmentation_config')):
            raise ValueError(
                "Experiment spec doesnt' have dataset_config or "
                "augmentation config. Please make sure the dataset_config "
                "and augmentation config are both present in the experiment_spec "
                "file provided.")
        dataset_config = experiment_spec.dataset_config
        augmentation_config = experiment_spec.augmentation_config
        dataloader = build_dataloader(dataset_config, augmentation_config)
        self._data_source, _, num_samples = dataloader.get_dataset_tensors(
            self._batch_size,
            training=True,
            enable_augmentation=False
        )
        logger.info("Number of samples from the dataloader: {}".format(num_samples))

    def get_data_from_source(self):
        """Simple function to get data from the defined data_source."""
        batch = self.session.run(self._data_source)
        if batch is None:
            raise ValueError(
                "Batch wasn't yielded from the data source. You may have run "
                "out of batches. Please set the num batches accordingly")
        return batch

    def get_batch(self, names):
        """Return one batch.

        Args:
            names (list): list of memory bindings names.
        """
        if self._batch_count < self._n_batches:
            batch = self.get_data_from_source()
            if batch is not None:
                if self._data_mem is None:
                    # 4 bytes per float32.
                    self._data_mem = cuda.mem_alloc(batch.size * 4)

                self._batch_count += 1

                # Transfer input data to device.
                cuda.memcpy_htod(self._data_mem, np.ascontiguousarray(
                    batch, dtype=np.float32))
                return [int(self._data_mem)]

        if self._batch_count >= self._n_batches:
            self.session.close()
            tf.reset_default_graph()

        if self._data_mem is not None:
            self._data_mem.free()
        return None
