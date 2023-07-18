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
import tempfile
import tensorflow as tf

os.environ["TF_KERAS"] = "1"
from nvidia_tao_tf1.core.export._onnx import keras_to_onnx  # noqa
from nvidia_tao_tf1.cv.common.export.keras_exporter import KerasExporter as Exporter  # noqa
from nvidia_tao_tf1.cv.lprnet.models import eval_builder  # noqa
from nvidia_tao_tf1.cv.lprnet.utils.model_io import load_model  # noqa
from nvidia_tao_tf1.cv.lprnet.utils.spec_loader import load_experiment_spec, spec_validator, EXPORT_EXP_REQUIRED_MSG  # noqa

logger = logging.getLogger(__name__)


class LPRNetExporter(Exporter):
    """Exporter class to export a trained LPRNet model."""

    def __init__(self, model_path=None,
                 key=None,
                 data_type="fp32",
                 strict_type=False,
                 experiment_spec_path="",
                 backend="onnx",
                 **kwargs):
        """Instantiate the LPRNet exporter to export a trained LPRNet .tlt model.

        Args:
            model_path(str): Path to the LPRNet model file.
            key (str): Key to decode the model.
            data_type (str): Backend data-type for the optimized TensorRT engine.
            strict_type(bool): Apply TensorRT strict_type_constraints or not for INT8 mode.
            experiment_spec_path (str): Path to LPRNet experiment spec file.
            backend (str): Type of intermediate backend parser to be instantiated.
        """
        super(LPRNetExporter, self).__init__(model_path=model_path,
                                             key=key,
                                             data_type=data_type,
                                             strict_type=strict_type,
                                             backend=backend,
                                             **kwargs)
        self.experiment_spec_path = experiment_spec_path
        assert os.path.isfile(self.experiment_spec_path), \
            "Experiment spec file not found at {}.".format(self.experiment_spec_path)
        self.experiment_spec = None

    def load_model(self, backend="onnx"):
        """Simple function to load the LPRNet Keras model."""
        experiment_spec = load_experiment_spec(self.experiment_spec_path)
        spec_validator(experiment_spec, EXPORT_EXP_REQUIRED_MSG)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)
        tf.keras.backend.clear_session()  # Clear previous models from memory.
        tf.keras.backend.set_learning_phase(0)
        model = load_model(model_path=self.model_path,
                           max_label_length=experiment_spec.lpr_config.max_label_length,
                           key=self.key)

        # Build evaluation model
        model = eval_builder.build(model)

        self.experiment_spec = experiment_spec
        return model

    def save_exported_file(self, model, output_file_name):
        """Save the exported model file.

        This routine converts a keras model to onnx/uff model
        based on the backend the exporter was initialized with.

        Args:
            model (keras.model.Model): Decoded keras model to be exported.
            output_file_name (str): Path to the output file.

        Returns:
            tmp_uff_file (str): Path to the temporary uff file.
        """
        if self.backend == "onnx":
            keras_to_onnx(
                model,
                output_file_name,
                target_opset=self.target_opset)
            return output_file_name
        raise NotImplementedError("Invalid backend provided. {}".format(self.backend))

    def set_input_output_node_names(self):
        """Set input output node names."""
        self.output_node_names = ["tf_op_layer_ArgMax", "tf_op_layer_Max"]
        self.input_node_names = ["image_input"]

    def set_data_preprocessing_parameters(self, input_dims, image_mean=None):
        """Set data pre-processing parameters for the int8 calibration."""
        num_channels = input_dims[0]
        if num_channels == 3:
            means = [0, 0, 0]
        elif num_channels == 1:
            means = [0]
        else:
            raise NotImplementedError("Invalid number of dimensions {}.".format(num_channels))
        self.preprocessing_arguments = {"scale": 1.0 / 255.0,
                                        "means": means,
                                        "flip_channel": True}

    def get_input_dims_from_model(self, model=None):
        """Read input dimensions from the model.

        Args:
            model (keras.models.Model): Model to get input dimensions from.

        Returns:
            input_dims (tuple): Input dimensions.
        """
        if model is None:
            raise IOError("Invalid model object.")
        input_dims = model.layers[1].input_shape[1:]
        return input_dims
