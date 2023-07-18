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

"""Base class to export trained .tlt models to etlt file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

try:
    import tensorrt as trt  # noqa pylint: disable=W0611 pylint: disable=W0611
    from nvidia_tao_tf1.cv.common.export.tensorfile_calibrator import TensorfileCalibrator
    from nvidia_tao_tf1.cv.detectnet_v2.export.detectnet_calibrator import DetectNetCalibrator
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Failed to import TensorRT package, exporting TLT to a TensorRT engine "
        "will not be available."
    )
from nvidia_tao_tf1.cv.common.export.keras_exporter import KerasExporter as Exporter
from nvidia_tao_tf1.cv.detectnet_v2.spec_handler.spec_loader import load_experiment_spec

logger = logging.getLogger(__name__)
CUSTOM_OBJS = None


class DetectNetExporter(Exporter):
    """Define an exporter for trained DetectNet_v2 models."""

    def __init__(self, model_path=None,
                 key=None,
                 data_type="fp32",
                 strict_type=False,
                 experiment_spec_path=None,
                 backend="uff",
                 data_format="channels_first",
                 onnx_route="keras2onnx",
                 **kwargs):
        """Initialize the DetectNet_v2 exporter.

        Args:
            model_path (str): Path to the model file.
            key (str): Key to load the model.
            data_type (str): Path to the TensorRT backend data type.
            strict_type(bool): Apply TensorRT strict_type_constraints or not for INT8 mode.
            backend (str): TensorRT parser to be used.
            experiment_spec_path (str): Path to the experiment spec file.
            data_format (str): Format of the input_channels.

        Returns:
            None.
        """
        super(DetectNetExporter, self).__init__(model_path=model_path,
                                                key=key,
                                                data_type=data_type,
                                                strict_type=strict_type,
                                                backend=backend,
                                                data_format=data_format,
                                                onnx_route=onnx_route,
                                                **kwargs)
        if experiment_spec_path is not None:
            assert os.path.exists(experiment_spec_path), (
                "Experiment spec file is not found at: {}",
                format(experiment_spec_path)
            )
            self.experiment_spec = load_experiment_spec(
                spec_path=experiment_spec_path,
                merge_from_default=False,
                validation_schema="train_val")

    def get_class_labels(self):
        """Get list of class labels to serialize to a labels.txt file."""
        if self.experiment_spec is None:
            raise AttributeError(
                "Experiment spec wasn't loaded. To get class labels "
                "please provide the experiment spec file using the -e "
                "option.")
        if not self.experiment_spec.HasField("cost_function_config"):
            raise AttributeError(
                "cost_function_config not defined in the experiment spec file."
            )
        cf_config = self.experiment_spec.cost_function_config
        target_classes = [
            target_class.name for target_class in cf_config.target_classes
        ]
        return target_classes

    def set_input_output_node_names(self):
        """Set input output node names."""
        self.output_node_names = ["output_cov/Sigmoid", "output_bbox/BiasAdd"]
        self.input_node_names = ["input_1"]

    def set_data_preprocessing_parameters(self, input_dims, image_mean=None):
        """Set data pre-processing parameters for the int8 calibration."""
        logger.debug("Input dimensions: {}".format(input_dims))
        num_channels = input_dims[0]
        scale = 1.0/255.0
        if num_channels == 3:
            means = [0., 0., 0.]
        elif num_channels == 1:
            means = [0]
        else:
            raise NotImplementedError("Invalid number of dimensions {}.".format(num_channels))
        self.preprocessing_arguments = {"scale": scale,
                                        "means": means,
                                        "flip_channel": False}

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
            image_mean (tuple): Pixel mean for channel-wise mean subtraction.

        Returns:
            calibrator (nvidia_tao_tf1.cv.common.export.base_calibrator.TensorfileCalibrator):
                TRTEntropyCalibrator2 instance to calibrate the TensorRT engine.
        """
        if self.experiment_spec is not None:
            # Get calibrator based on the detectnet dataloader.
            calibrator = DetectNetCalibrator(
                self.experiment_spec,
                calibration_cache,
                n_batches,
                batch_size)
        else:
            if not os.path.exists(data_file_name):
                self.generate_tensor_file(data_file_name,
                                          calibration_images_dir,
                                          input_dims,
                                          n_batches=n_batches,
                                          batch_size=batch_size)
            calibrator = TensorfileCalibrator(data_file_name,
                                              calibration_cache,
                                              n_batches,
                                              batch_size)
        return calibrator
