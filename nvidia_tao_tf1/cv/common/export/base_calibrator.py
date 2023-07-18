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

"""Base calibrator class for TensorRT INT8 Calibration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

# Simple helper class for calibration.
try:
    import tensorrt as trt  # noqa pylint: disable=W0611 pylint: disable=W0611
    trt_available = True
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Failed to import TensorRT package, exporting TLT to a TensorRT engine "
        "will not be available."
    )
    trt_available = False

logger = logging.getLogger(__name__)


if trt_available:
    class BaseCalibrator(trt.IInt8EntropyCalibrator2):
        """Calibrator class."""

        def __init__(self, cache_filename,
                     n_batches,
                     batch_size,
                     *args, **kwargs):
            """Init routine.

            This inherits from ``trt.IInt8EntropyCalibrator2`` to implement
            the calibration interface that TensorRT needs to calibrate the
            INT8 quantization factors.

            Args:
                cache_filename (str): name of calibration file to read/write to.
                n_batches (int): number of batches for calibrate for.
                batch_size (int): batch size to use for calibration data.
            """
            super(BaseCalibrator, self).__init__(*args, **kwargs)

            self._data_source = None
            self._cache_filename = cache_filename

            self._batch_size = batch_size
            self._n_batches = n_batches

            self._batch_count = 0
            self._data_mem = None

        def instantiate_data_source(self, data_filename):
            """Simple function to instantiate the data_source of the dataloader.

            Args:
                data_filename (str): The path to the data file.

            Returns:
                No explicit returns.
            """
            raise NotImplementedError(
                "Base calibrator doesn't implement data source instantiation."
            )

        def get_data_from_source(self):
            """Simple function to get data from the defined data_source."""
            raise NotImplementedError(
                "Base calibrator doesn't implement yielding data from data source"
            )

        def get_batch(self, names):
            """Return one batch.

            Args:
                names (list): list of memory bindings names.
            """
            raise NotImplementedError(
                "Base calibrator doesn't implement calibrator get_batch()"
            )

        def get_batch_size(self):
            """Return batch size."""
            return self._batch_size

        def read_calibration_cache(self):
            """Read calibration from file."""
            logger.debug("read_calibration_cache - no-op")
            if os.path.isfile(self._cache_filename):
                logger.warning("Calibration file exists at {}."
                               " Reading this cache.".format(self._cache_filename))
                with open(self._cache_filename, "rb") as cal_file:
                    return cal_file.read()
            return None

        def write_calibration_cache(self, cache):
            """Write calibration to file.

            Args:
                cache (memoryview): buffer to read calibration data from.
            """
            logger.info("Saving calibration cache (size %d) to %s",
                        len(cache), self._cache_filename)
            with open(self._cache_filename, 'wb') as f:
                f.write(cache)
