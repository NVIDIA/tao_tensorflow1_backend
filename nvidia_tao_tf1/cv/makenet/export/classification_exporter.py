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

import json
import logging
import os
import sys
import tempfile

import keras

from numba import cuda
import tensorflow as tf

from nvidia_tao_tf1.core.export._onnx import keras_to_onnx
# Import quantization layer processing.
from nvidia_tao_tf1.core.export._quantized import (
    check_for_quantized_layers,
)
from nvidia_tao_tf1.core.export._uff import keras_to_pb, keras_to_uff
from nvidia_tao_tf1.core.export.app import get_model_input_dtype
from nvidia_tao_tf1.cv.common.export.keras_exporter import KerasExporter as Exporter
from nvidia_tao_tf1.cv.common.export.keras_exporter import SUPPORTED_ONNX_ROUTES
from nvidia_tao_tf1.cv.common.export.tensorfile_calibrator import TensorfileCalibrator
from nvidia_tao_tf1.cv.common.export.utils import pb_to_onnx
from nvidia_tao_tf1.cv.common.types.base_ds_config import BaseDSConfig
from nvidia_tao_tf1.cv.common.utils import CUSTOM_OBJS, get_decoded_filename, model_io
from nvidia_tao_tf1.cv.makenet.export.classification_calibrator import ClassificationCalibrator
from nvidia_tao_tf1.cv.makenet.spec_handling.spec_loader import load_experiment_spec

logger = logging.getLogger(__name__)


class ClassificationExporter(Exporter):
    """Define an exporter for classification models."""

    def __init__(self, model_path=None,
                 key=None,
                 data_type="fp32",
                 strict_type=False,
                 backend="uff",
                 classmap_file=None,
                 experiment_spec_path="",
                 onnx_route="keras2onnx",
                 **kwargs):
        """Initialize the classification exporter.

        Args:
            model_path (str): Path to the model file.
            key (str): Key to load the model.
            data_type (str): Path to the TensorRT backend data type.
            strict_type(bool): Apply TensorRT strict_type_constraints or not for INT8 mode.
            backend (str): TensorRT parser to be used.
            classmap_file (str): Path to classmap.json file.
            experiment_spec_path (str): Path to MakeNet experiment spec file.
            onnx_route (str): Package to be used to convert the keras model to
                ONNX.

        Returns:
            None.
        """
        super(ClassificationExporter, self).__init__(model_path=model_path,
                                                     key=key,
                                                     data_type=data_type,
                                                     strict_type=strict_type,
                                                     backend=backend,
                                                     **kwargs)
        self.classmap_file = classmap_file
        self.onnx_route = onnx_route
        assert self.onnx_route in SUPPORTED_ONNX_ROUTES, (
            f"Invaid onnx route {self.onnx_route} requested."
        )
        logger.info("Setting the onnx export rote to {}".format(
            self.onnx_route
        ))

        # Load experiment spec if available.
        if os.path.exists(experiment_spec_path):
            self.experiment_spec = load_experiment_spec(
                experiment_spec_path,
                merge_from_default=False,
                validation_schema="train_val"
            )
        self.eff_custom_objs = None

    def set_keras_backend_dtype(self):
        """Set the keras backend data type."""
        keras.backend.set_learning_phase(0)
        tmp_keras_file_name = get_decoded_filename(self.model_path,
                                                   self.key,
                                                   self.eff_custom_objs)
        model_input_dtype = get_model_input_dtype(tmp_keras_file_name)
        keras.backend.set_floatx(model_input_dtype)

    def load_model(self, backend="uff"):
        """Simple function to get the keras model."""
        keras.backend.clear_session()
        keras.backend.set_learning_phase(0)
        model = model_io(self.model_path, enc_key=self.key, custom_objects=self.eff_custom_objs)
        if check_for_quantized_layers(model):
            model, self.tensor_scale_dict = self.extract_tensor_scale(model, backend)
        return model

    def set_input_output_node_names(self):
        """Set input output node names."""
        self.output_node_names = ["predictions/Softmax"]
        self.input_node_names = ["input_1"]

    def load_classmap_file(self):
        """Load the classmap json."""
        data = None
        with open(self.classmap_file, "r") as cmap_file:
            try:
                data = json.load(cmap_file)
            except json.decoder.JSONDecodeError as e:
                print(f"Loading the {self.classmap_file} failed with error\n{e}")
                sys.exit(-1)
            except Exception as e:
                if e.output is not None:
                    print(f"Classification exporter failed with error {e.output}")
                sys.exit(-1)
        return data

    def get_class_labels(self):
        """Get list of class labels to serialize to a labels.txt file."""
        if not os.path.exists(self.classmap_file):
            raise FileNotFoundError(
                f"Classmap json file not found: {self.classmap_file}")
        data = self.load_classmap_file()
        if not data:
            return []
        labels = [""] * len(list(data.keys()))
        if not all([class_index < len(labels)
                    and isinstance(class_index, int)
                    for class_index in data.values()]):
            raise RuntimeError(
                "Invalid data in the json file. The class index must "
                "be < number of classes and an integer value.")
        for class_name, class_index in data.items():
            labels[class_index] = class_name
        return labels

    def generate_ds_config(self, input_dims, num_classes=None):
        """Generate Deepstream config element for the exported model."""
        channel_index = 0 if self.data_format == "channels_first" else -1
        if input_dims[channel_index] == 1:
            color_format = "l"
        else:
            color_format = "bgr" if self.preprocessing_arguments["flip_channel"] else "rgb"
        kwargs = {
            "data_format": self.data_format,
            "backend": self.backend,
            # Setting this to 1 for classification
            "network_type": 1
        }
        if num_classes:
            kwargs["num_classes"] = num_classes
        if self.backend == "uff":
            kwargs.update({
                "input_names": self.input_node_names,
                "output_names": self.output_node_names
            })

        ds_config = BaseDSConfig(
            self.preprocessing_arguments["scale"],
            self.preprocessing_arguments["means"],
            input_dims,
            color_format,
            self.key,
            **kwargs
        )
        return ds_config

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
        logger.debug("Saving etlt model file at: {}.".format(output_file_name))
        input_tensor_names = ""
        # @vpraveen: commented out the preprocessor kwarg from keras_to_uff.
        # todo: @vpraveen and @zhimeng, if required modify modulus code to add
        # this.
        if self.backend == "uff":
            input_tensor_names, _, _ = keras_to_uff(
                model,
                output_file_name,
                output_node_names=self.output_node_names,
                custom_objects=CUSTOM_OBJS)
        elif self.backend == "onnx":
            if self.onnx_route == "keras2onnx":
                keras_to_onnx(
                    model,
                    output_file_name,
                    custom_objects=CUSTOM_OBJS,
                    target_opset=self.target_opset
                )
            else:
                os_handle, tmp_pb_file = tempfile.mkstemp(
                    suffix=".pb"
                )
                os.close(os_handle)
                input_tensor_names, out_tensor_names, _ = keras_to_pb(
                    model,
                    tmp_pb_file,
                    self.output_node_names,
                    custom_objects=CUSTOM_OBJS
                )
                if self.output_node_names is None:
                    self.output_node_names = out_tensor_names
                logger.info("Model graph serialized to pb file.")
                input_tensor_names, out_tensor_names = pb_to_onnx(
                    tmp_pb_file,
                    output_file_name,
                    input_tensor_names,
                    self.output_node_names,
                    self.target_opset,
                    verbose=False
                )
                input_tensor_names = ""
        else:
            raise NotImplementedError("Incompatible backend.")
        return output_file_name

    def clear_gpus(self):
        """Clear GPU memory before TRT engine building."""
        tf.reset_default_graph()

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
            calibrator = ClassificationCalibrator(
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
