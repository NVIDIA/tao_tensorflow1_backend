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

"""Base class to export trained .tlt keras models to etlt file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod
import json
import logging
import os
import random
import struct

import numpy as np
from PIL import Image
from six.moves import xrange
from tqdm import tqdm

try:
    import tensorrt as trt  # noqa pylint: disable=W0611 pylint: disable=W0611
    from nvidia_tao_tf1.cv.common.export.tensorfile_calibrator import TensorfileCalibrator
    from nvidia_tao_tf1.cv.common.export.trt_utils import (
        NV_TENSORRT_MAJOR,
        NV_TENSORRT_MINOR,
        NV_TENSORRT_PATCH
    )
    trt_available = True
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Failed to import TensorRT package, exporting TLT to a TensorRT engine "
        "will not be available."
    )
    trt_available = False
from nvidia_tao_tf1.cv.common.export.tensorfile import TensorFile
from nvidia_tao_tf1.cv.common.logging import logging as status_logging


# Define valid backend available for the exporter.
VALID_BACKEND = ["uff", "onnx"]

logger = logging.getLogger(__name__)


class BaseExporter(object):
    """Base class for exporter."""

    def __init__(self, model_path=None,
                 key=None,
                 data_type="fp32",
                 strict_type=False,
                 backend="uff",
                 **kwargs):
        """Initialize the base exporter.

        Args:
            model_path (str): Path to the model file.
            key (str): Key to load the model.
            data_type (str): Path to the TensorRT backend data type.
            strict_type(bool): Apply TensorRT strict_type_constraints or not for INT8 mode.
            backend (str): TensorRT parser to be used.

        Returns:
            None.
        """
        self.data_type = data_type
        self.strict_type = strict_type
        self.model_path = model_path
        # if key is str, it will be converted to bytes in nvidia_tao_tf1.encoding
        self.key = key
        self.set_backend(backend)
        self.status_logger = status_logging.get_status_logger()
        self.tensor_scale_dict = None
        self._trt_version_number = NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + \
            NV_TENSORRT_PATCH
        self.image_depth = 8

    def set_session(self):
        """Set keras backend session."""
        raise NotImplementedError("To be implemented by the class being used.")

    def set_keras_backend_dtype(self):
        """Set the keras backend data type."""
        raise NotImplementedError(
            "To be implemented by the class being used.")

    @abstractmethod
    def set_input_output_node_names(self):
        """Set input output node names."""
        raise NotImplementedError(
            "This function is not implemented in the base class.")

    def extract_tensor_scale(self, model, backend):
        """Extract tensor scale from QAT trained model and de-quantize the model."""
        raise NotImplementedError(
            "This function is not implemented in the base class.")

    @abstractmethod
    def load_model(self, backend="uff"):
        """Simple function to get the keras model."""
        raise NotImplementedError(
            "This function is not implemented in the base class.")

    @abstractmethod
    def get_class_labels(self, experiment_spec=None):
        """Save the labels file required for deepstream implementation."""
        raise NotImplementedError("Base class doesn't implement get_class_labels function.")

    def set_backend(self, backend):
        """Set keras backend.

        Args:
            backend (str): Backend to be used.
                Currently only UFF is supported.
        """
        if backend not in VALID_BACKEND:
            raise NotImplementedError(
                'Invalid backend "{}" called'.format(backend))
        self.backend = backend

    @abstractmethod
    def generate_ds_config(self, input_dims, num_classes=None):
        """Generate the deepstream config for the exported model.

        Args:
            input_dims (tuple): Input dimensions of the model.

        Returns:
            ds_config (nvidia_tao_tf1.cv.common.types.base_ds_config.BaseDSConfig): Deepstream
                config element.
        """
        raise NotImplementedError(
            "Generate the DeepStream config for the model."
        )

    @abstractmethod
    def save_exported_file(self, model, output_file_name):
        """Save the exported model file.

        This routine converts a keras model to onnx/uff model
        based on the backend the exporter was initialized with.

        Args:
            model (keras.models.Model): Keras model to be saved.
            output_file_name (str): Path to the output etlt file.

        Returns:
            tmp_file_name (str): Path to the temporary uff file.
        """
        raise NotImplementedError(
            "This function is not implemented in the base class.")

    def get_calibrator(self,
                       calibration_cache,
                       data_file_name,
                       n_batches,
                       batch_size,
                       input_dims,
                       calibration_images_dir=None,
                       image_mean=None):
        """Simple function to get an int8 calibrator.

        Args:
            calibration_cache (str): Path to store the int8 calibration cache file.
            data_file_name (str): Path to the TensorFile. If the tensorfile doesn't exist
                at this path, then one is created with either n_batches of random tensors,
                images from the file in calibration_images_dir of dimensions
                (batch_size,) + (input_dims)
            n_batches (int): Number of batches to calibrate the model over.
            batch_size (int): Number of input tensors per batch.
            input_dims (tuple): Tuple of input tensor dimensions in CHW order.
            calibration_images_dir (str): Path to a directory of images to generate the
                data_file from.
            image_mean (list): image mean per channel.

        Returns:
            calibrator (nvidia_tao_tf1.cv.common.export.base_calibrator.TensorfileCalibrator):
                TRTEntropyCalibrator2 instance to calibrate the TensorRT engine.
        """
        if not os.path.exists(data_file_name):
            self.generate_tensor_file(data_file_name,
                                      calibration_images_dir,
                                      input_dims,
                                      n_batches=n_batches,
                                      batch_size=batch_size,
                                      image_mean=image_mean)
        calibrator = TensorfileCalibrator(data_file_name,
                                          calibration_cache,
                                          n_batches,
                                          batch_size)
        return calibrator

    def _calibration_cache_from_dict(self, tensor_scale_dict,
                                     calibration_cache=None,
                                     calib_json=None):
        """Write calibration cache file for QAT model.

        This function converts a tensor scale dictionary generated by processing
        QAT models to TRT readable format. By default we set it as a
        trt.IInt8.EntropyCalibrator2 cache file.

        Args:
            tensor_scale_dict (dict): The dictionary of parameters: scale_value file.
            calibration_cache (str): Path to output calibration cache file.

        Returns:
            No explicit returns.
        """
        if calibration_cache is not None:
            cal_cache_str = "TRT-{}-EntropyCalibration2\n".format(
                self._trt_version_number)
            assert not os.path.exists(calibration_cache), (
                "A pre-existing cache file exists. Please delete this "
                "file and re-run export."
            )
            # Converting float numbers to hex representation.
            for tensor in tensor_scale_dict:
                scaling_factor = tensor_scale_dict[tensor] / 127.0
                cal_scale = hex(struct.unpack(
                    "i", struct.pack("f", scaling_factor))[0])
                assert cal_scale.startswith(
                    "0x"), "Hex number expected to start with 0x."
                cal_scale = cal_scale[2:]
                cal_cache_str += tensor + ": " + cal_scale + "\n"
            with open(calibration_cache, "w") as f:
                f.write(cal_cache_str)

        if calib_json is not None:
            calib_json_data = {"tensor_scales": {}}
            for tensor in tensor_scale_dict:
                calib_json_data["tensor_scales"][tensor] = float(
                    tensor_scale_dict[tensor])
            with open(calib_json, "w") as outfile:
                json.dump(calib_json_data, outfile, indent=4)

    def set_data_preprocessing_parameters(self, input_dims, image_mean=None):
        """Set data pre-processing parameters for the int8 calibration."""
        num_channels = input_dims[0]
        if num_channels == 3:
            assert self.image_depth == 8, (
                f"RGB images only support image depth of 8, got {self.image_depth}"
            )
            if not image_mean:
                means = [103.939, 116.779, 123.68]
            else:
                assert len(image_mean) == 3, "Image mean should have 3 values for RGB inputs."
                means = image_mean
        elif num_channels == 1:
            if not image_mean:
                if self.image_depth == 8:
                    means = [117.3786]
                elif self.image_depth == 16:
                    means = [117.3786*256.]
                else:
                    raise ValueError(
                        "Single-channel images only support depth of 8 or 16, "
                        f"got {self.image_depth}"
                    )
            else:
                assert len(image_mean) == 1, "Image mean should have 1 value for grayscale inputs."
                means = image_mean
        else:
            raise NotImplementedError(
                "Invalid number of dimensions {}.".format(num_channels))
        self.preprocessing_arguments = {"scale": 1.0,
                                        "means": means,
                                        "flip_channel": True}

    def generate_tensor_file(self, data_file_name,
                             calibration_images_dir,
                             input_dims, n_batches=10,
                             batch_size=1, image_mean=None):
        """Generate calibration Tensorfile for int8 calibrator.

        This function generates a calibration tensorfile from a directory of images, or dumps
        n_batches of random numpy arrays of shape (batch_size,) + (input_dims).

        Args:
            data_file_name (str): Path to the output tensorfile to be saved.
            calibration_images_dir (str): Path to the images to generate a tensorfile from.
            input_dims (list): Input shape in CHW order.
            n_batches (int): Number of batches to be saved.
            batch_size (int): Number of images per batch.
            image_mean (list): Image mean per channel.

        Returns:
            No explicit returns.
        """
        if not os.path.exists(calibration_images_dir):
            logger.info("Generating a tensorfile with random tensor images. This may work well as "
                        "a profiling tool, however, it may result in inaccurate results at "
                        "inference. Please generate a tensorfile using the tlt-int8-tensorfile, "
                        "or provide a custom directory of images for best performance.")
            self.generate_random_tensorfile(data_file_name,
                                            input_dims,
                                            n_batches=n_batches,
                                            batch_size=batch_size)
        else:
            # Preparing the list of images to be saved.
            num_images = n_batches * batch_size
            valid_image_ext = ['jpg', 'jpeg', 'png']
            image_list = [os.path.join(calibration_images_dir, image)
                          for image in os.listdir(calibration_images_dir)
                          if image.split('.')[-1] in valid_image_ext]
            if len(image_list) < num_images:
                raise ValueError('Not enough number of images provided:'
                                 ' {} < {}'.format(len(image_list), num_images))
            image_idx = random.sample(xrange(len(image_list)), num_images)
            self.set_data_preprocessing_parameters(input_dims, image_mean)
            # Writing out processed dump.
            with TensorFile(data_file_name, 'w') as f:
                for chunk in tqdm(image_idx[x:x+batch_size] for x in xrange(0, len(image_idx),
                                                                            batch_size)):
                    dump_data = self.prepare_chunk(chunk, image_list,
                                                   image_width=input_dims[2],
                                                   image_height=input_dims[1],
                                                   channels=input_dims[0],
                                                   batch_size=batch_size,
                                                   **self.preprocessing_arguments)
                    f.write(dump_data)
            f.closed

    @staticmethod
    def generate_random_tensorfile(data_file_name, input_dims, n_batches=1, batch_size=1):
        """Generate a random tensorfile.

        This function generates a random tensorfile containing n_batches of random np.arrays
        of dimensions (batch_size,) + (input_dims).

        Args:
            data_file_name (str): Path to where the data tensorfile will be stored.
            input_dims (tuple): Input blob dimensions in CHW order.
            n_batches (int): Number of batches to save.
            batch_size (int): Number of images per batch.

        Return:
            No explicit returns.
        """
        sample_shape = (batch_size, ) + tuple(input_dims)
        with TensorFile(data_file_name, 'w') as f:
            for i in tqdm(xrange(n_batches)):
                logger.debug("Writing batch: {}".format(i))
                dump_sample = np.random.sample(sample_shape)
                f.write(dump_sample)

    @staticmethod
    def prepare_chunk(image_ids, image_list,
                      image_width=480,
                      image_height=272,
                      channels=3,
                      scale=1.0,
                      means=None,
                      flip_channel=False,
                      batch_size=1):
        """Prepare a single batch of data to dump into a Tensorfile."""
        dump_placeholder = np.zeros(
            (batch_size, channels, image_height, image_width))
        for i in xrange(len(image_ids)):
            idx = image_ids[i]
            im = Image.open(image_list[idx]).resize((image_width, image_height),
                                                    Image.ANTIALIAS)
            if channels == 1:
                logger.debug("Converting image from RGB to Grayscale")
                if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                    bg_colour = (255, 255, 255)
                    # Need to convert to RGBA if LA format due to a bug in PIL
                    alpha = im.convert('RGBA').split()[-1]
                    # Create a new background image of our matt color.
                    # Must be RGBA because paste requires both images have the same format
                    bg = Image.new("RGBA", im.size, bg_colour + (255,))
                    bg.paste(im, mask=alpha)

                im = im.convert('L')
                dump_input = np.asarray(im).astype(np.float32)
                dump_input = dump_input[:, :, np.newaxis]
            elif channels == 3:
                dump_input = np.asarray(im.convert('RGB')).astype(np.float32)
            else:
                raise NotImplementedError("Unsupported channel dimensions.")
            # flip channel: RGB --> BGR
            if flip_channel:
                dump_input = dump_input[:, :, ::-1]
            # means is a list of per-channel means, (H, W, C) - (C)
            if means is not None:
                dump_input -= np.array(means)
            # (H, W, C) --> (C, H, W)
            dump_input = dump_input.transpose(2, 0, 1) * scale
            dump_placeholder[i, :, :, :] = dump_input
        return dump_placeholder

    def get_input_dims(self, data_file_name=None, model=None):
        """Simple function to get input layer dimensions.

        Args:
            data_file_name (str): Path to the calibration tensor file.
            model (keras.models.Model): Keras model object.

        Returns:
            input_dims (list): Input dimensions in CHW order.
        """
        if not os.path.exists(data_file_name):
            logger.debug(
                "Data file doesn't exist. Pulling input dimensions from the network.")
            input_dims = self.get_input_dims_from_model(model)
        else:
            # Read the input dims from the Tensorfile.
            logger.debug("Reading input dims from tensorfile.")
            with TensorFile(data_file_name, "r") as tfile:
                batch = tfile.read()
                # Disabling pylint for py3 in this line due to a pylint issue.
                # Follow issue: https://github.com/PyCQA/pylint/issues/3139
                # and remove when ready.
                input_dims = np.array(batch).shape[1:]  # pylint: disable=E1136
        return input_dims

    @staticmethod
    def get_input_dims_from_model(model=None):
        """Read input dimensions from the model.

        Args:
            model (keras.models.Model): Model to get input dimensions from.

        Returns:
            input_dims (tuple): Input dimensions.
        """
        if model is None:
            raise IOError("Invalid model object.")
        input_dims = model.layers[0].input_shape[1:]
        return input_dims

    @abstractmethod
    def export(self, output_file_name, backend,
               calibration_cache="", data_file_name="",
               n_batches=1, batch_size=1, verbose=True,
               calibration_images_dir="", save_engine=False,
               engine_file_name="", max_workspace_size=1 << 30,
               max_batch_size=1, force_ptq=False):
        """Simple function to export a model.

        This function sets the first converts a keras graph to uff and then saves it to an etlt
        file. After which, it verifies the parsability of the etlt file by creating a TensorRT
        engine of desired backend datatype.

        Args:
            output_file_name (str): Path to the output etlt file.
            backend (str): Backend parser to be used. ("uff", "onnx).
            calibration_cache (str): Path to the output calibration cache file.
            data_file_name (str): Path to the data tensorfile for int8 calibration.
            n_batches (int): Number of batches to calibrate model for int8 calibration.
            batch_size (int): Number of images per batch.
            verbose (bool): Flag to set verbose logging.
            calibration_images_dir (str): Path to a directory of images for custom data
                to calibrate the model over.
            save_engine (bool): Flag to save the engine after training.
            engine_file_name (str): Path to the engine file name.
            force_ptq (bool): Flag to force post training quantization using TensorRT
                for a QAT trained model. This is required iff the inference platform is
                a Jetson with a DLA.

        Returns:
            No explicit returns.
        """
        raise NotImplementedError("Base Class doesn't implement this method.")
