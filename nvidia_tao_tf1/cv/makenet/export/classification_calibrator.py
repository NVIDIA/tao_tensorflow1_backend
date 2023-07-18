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

from functools import partial
import logging

import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pycuda.autoinit  # noqa pylint: disable=unused-import
import pycuda.driver as cuda
import tensorflow as tf

# Simple helper class for calibration.
from nvidia_tao_tf1.cv.common.export.base_calibrator import BaseCalibrator
# Building the classification dataloader.
from nvidia_tao_tf1.cv.makenet.utils import preprocess_crop  # noqa pylint: disable=unused-import
from nvidia_tao_tf1.cv.makenet.utils.preprocess_input import preprocess_input

logger = logging.getLogger(__name__)


class ClassificationCalibrator(BaseCalibrator):
    """Detectnet_v2 calibrator class."""

    def __init__(self, experiment_spec, cache_filename,
                 n_batches, batch_size,
                 *args, **kwargs):
        """Init routine.

        This inherits from ``iva.common.export.base_calibrator.BaseCalibrator``
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
        super(ClassificationCalibrator, self).__init__(
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
        tf_session = tf.compat.v1.Session(
            config=session_config,
            graph=tf.get_default_graph()
        )
        # Setting the keras session.
        keras.backend.set_session(tf_session)
        self.session = keras.backend.get_session()

    def instantiate_data_source(self, experiment_spec):
        """Simple function to instantiate the data_source of the dataloader.

        Args:
            experiment_spec (iva.detectnet_v2.proto.experiment_pb2): Detectnet_v2
                experiment spec proto object.

        Returns:
            No explicit returns.
        """
        if not (hasattr(experiment_spec, 'train_config') or
                hasattr(experiment_spec, 'model_config')):
            raise ValueError(
                "Experiment spec doesnt' have train_config or "
                "model_config. Please make sure the train_config "
                "and model_config are both present in the experiment_spec "
                "file provided.")
        model_config = experiment_spec.model_config
        image_shape = model_config.input_image_size.split(",")
        n_channel = int(image_shape[0])
        image_height = int(image_shape[1])
        image_width = int(image_shape[2])
        assert n_channel in [1, 3], "Invalid input image dimension."
        assert image_height >= 16, "Image height should be greater than 15 pixels."
        assert image_width >= 16, "Image width should be greater than 15 pixels."
        img_mean = experiment_spec.train_config.image_mean
        if n_channel == 3:
            if img_mean:
                assert all(c in img_mean for c in ['r', 'g', 'b']) , (
                    "'r', 'g', 'b' should all be present in image_mean "
                    "for images with 3 channels."
                )
                img_mean = [img_mean['b'], img_mean['g'], img_mean['r']]
            else:
                img_mean = [103.939, 116.779, 123.68]
        else:
            if img_mean:
                assert 'l' in img_mean, (
                    "'l' should be present in image_mean for images "
                    "with 1 channel."
                )
                img_mean = [img_mean['l']]
            else:
                img_mean = [117.3786]

        # Define path to dataset.
        train_data = experiment_spec.train_config.train_dataset_path

        # Setting dataloader color_mode.
        color_mode = "rgb"
        if n_channel == 1:
            color_mode = "grayscale"

        preprocessing_func = partial(
            preprocess_input,
            data_format="channels_first",
            mode=experiment_spec.train_config.preprocess_mode,
            color_mode=color_mode,
            img_mean=img_mean
        )

        # Initialize the data generator.
        logger.info("Setting up input generator.")
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocessing_func,
            horizontal_flip=False,
            featurewise_center=False
        )
        logger.debug("Setting up iterator.")
        train_iterator = train_datagen.flow_from_directory(
            train_data,
            target_size=(
                image_height,
                image_width
            ),
            batch_size=self.batch_size,
            class_mode='categorical',
            color_mode=color_mode
        )
        logger.info("Number of samples from the dataloader: {}".format(train_iterator.n))
        num_available_batches = int(train_iterator.num_samples / self.batch_size)
        assert self._n_batches <= num_available_batches, (
            f"n_batches <= num_available_batches, n_batches={self._n_batches}, "
            f"num_available_batches={num_available_batches}"
        )
        self._data_source = train_iterator

    def get_data_from_source(self):
        """Simple function to get data from the defined data_source."""
        batch, _ = next(self._data_source)
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
