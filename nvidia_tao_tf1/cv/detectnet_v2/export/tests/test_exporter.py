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

import logging
import os
import tempfile

from keras import backend as K
import pytest
import tensorflow as tf

from nvidia_tao_tf1.cv.common.export.app import (
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_MAX_WORKSPACE_SIZE,
    run_export
)
from nvidia_tao_tf1.cv.detectnet_v2.cost_function.cost_function_parameters import (
    get_target_class_names
)
from nvidia_tao_tf1.cv.detectnet_v2.export.exporter import DetectNetExporter as Exporter
from nvidia_tao_tf1.cv.detectnet_v2.model.build_model import build_model
from nvidia_tao_tf1.cv.detectnet_v2.spec_handler.spec_loader import load_experiment_spec

# Todo: <vpraveen> Use GB Feature extractor constructor to construct GB model and export
# to TRT serializable format for inference
detectnet_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))
training_spec = os.path.join(detectnet_root,
                             "experiment_specs/default_spec.txt")

ENC_KEY = 'tlt_encode'

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")

# Restricting the number of GPU's to be used.
gpu_options = tf.compat.v1.GPUOptions(
    per_process_gpu_memory_fraction=0.33,
    allow_growth=True
)
device_count = {'GPU': 0, 'CPU': 1}
config = tf.compat.v1.ConfigProto(
    gpu_options=gpu_options,
    device_count=device_count
)
K.set_session(tf.Session(config=config))

topologies = [
    ("resnet", 18, 'channels_first', 16, "fp32", (3, 960, 544), False, False, False, "uff"),
    # ("resnet", 10, 'channels_first', 2, "int8", (3, 480, 272), True, False, False, "onnx"),
    # ("vgg", 16, 'channels_first', 2, "int8", (3, 480, 272), True, True, True, "onnx"),
    ("resnet", 18, 'channels_first', 8, "int8", (3, 480, 272), False, False, False, "uff"),
    ("resnet", 10, 'channels_first', 8, "int8", (3, 480, 272), False, True, False, "uff"),
    ("efficientnet_b0", 10, 'channels_first', 8, "int8", (3, 272, 480), False, True, False, "onnx"),
    ("efficientnet_b0", 10, 'channels_first', 8, "int8", (3, 272, 480), False, True, False, "uff"),
]


def get_tmp_file(suffix=None):
    """Simple wrapper to get a temp file with a suffix.

    Args:
        suffix (str): String suffix to end the temp file path.

    Return:
        tmpfile_path (str): Path to the tmp file.
    """
    os_handle, temp_file = tempfile.mkstemp(suffix=suffix)
    os.close(os_handle)
    os.unlink(temp_file)
    return temp_file


class TestDetectnetExporter(object):
    """Class to test DetectNet exporter."""

    def _setup_gridbox_model_instance(self, enable_qat):
        """Simple function to generate a test bench for TRT Export."""
        experiment_spec = load_experiment_spec(training_spec, validation_schema="train_val")
        if hasattr(experiment_spec, "model_config"):
            model_config = experiment_spec.model_config
        else:
            raise ValueError(
                "Invalid spec file without model_config at {}".format(training_spec))

        if hasattr(experiment_spec, "cost_function_config"):
            cost_function_config = experiment_spec.cost_function_config
        else:
            raise ValueError(
                "Invalid spec without costfunction config at {}".format(training_spec))

        target_class_names = get_target_class_names(cost_function_config)
        self.gridbox_model = build_model(model_config, target_class_names, enable_qat=enable_qat)

    def _generate_keras_model(self, arch, num_layers, input_shape):
        """Simple function to construct a detectnet_v2 model."""
        self.gridbox_model.template = arch
        self.gridbox_model.num_layers = num_layers
        self.gridbox_model.construct_model(input_shape=input_shape,
                                           kernel_regularizer=None,
                                           bias_regularizer=None,
                                           pretrained_weights_file=None,
                                           enc_key=ENC_KEY)
        keras_model_file = get_tmp_file(suffix=".hdf5")
        # save keras model to a temp file.
        self.gridbox_model.save_model(keras_model_file, enc_key=ENC_KEY)
        self.gridbox_model.keras_model.summary()
        tf.reset_default_graph()
        K.clear_session()
        return keras_model_file

    def _common(self, exporter_args, backend):
        """Simple function to run common exporter test routines.

        Args:
            exporter_args (dict): Arguments of exporter.

        Returns:
            No explicit returns
        """
        run_export(Exporter, exporter_args, backend=backend)
        output_file = exporter_args["output_file"]
        calibration_cache_file = exporter_args["cal_cache_file"]
        data_type = exporter_args["data_type"]
        engine_file = exporter_args["engine_file"]
        gen_ds_config = exporter_args["gen_ds_config"]
        # Check if etlt file was written
        assert os.path.isfile(output_file), (
            "etlt file was not written."
        )
        assert os.path.isfile(engine_file), (
            "Engine file was not generated."
        )
        # Check if int8 calibration file was written.
        if data_type == "int8":
            assert os.path.isfile(calibration_cache_file), (
                "Calibration cache file wasn't written."
            )
        if gen_ds_config:
            output_root = os.path.dirname(output_file)
            output_ds_file = os.path.join(output_root, "nvinfer_config.txt")
            assert os.path.isfile(output_ds_file), (
                "DS config file wasn't generated"
            )

    def clear_previous_files(self):
        """Clear previously generated files."""
        removable_extensions = [
            ".tlt", ".json", ".etlt", ".bin",
            ".trt", ".json", ".txt", ".onnx",
            ".uff", ".hdf5"
        ]
        for item in os.listdir("/tmp"):
            for ext in removable_extensions:
                if item.endswith(ext):
                    os.remove(os.path.join("/tmp", item))

    @pytest.mark.skipif(os.getenv("RUN_ON_CI", "0") == "1", reason="Cannot be run on CI")
    @pytest.mark.parametrize(
        "arch, num_layers, data_format, batch_size, data_type, input_shape, enable_qat, from_spec, gen_ds_config, backend",  # noqa: E501
        topologies
    )
    def test_detectnet_v2_exporter(self,
                                   arch,
                                   num_layers,
                                   data_format,
                                   batch_size,
                                   data_type,
                                   input_shape,
                                   enable_qat,
                                   from_spec,
                                   gen_ds_config,
                                   backend):
        """Simple function to test the DetectNet_v2 exporter."""
        # Parsing command line arguments.
        self._setup_gridbox_model_instance(enable_qat)
        model_path = self._generate_keras_model(arch,
                                                num_layers,
                                                input_shape)

        cal_cache_file = get_tmp_file(suffix=".bin")
        output_file = get_tmp_file(suffix=f".{backend}")
        engine_file = get_tmp_file(suffix=".trt")
        tensorfile_path = get_tmp_file(suffix=".tensorfile")
        cal_json_file = ""
        if enable_qat:
            cal_json_file = get_tmp_file(suffix=".json")

        exporter_args = {
            'model': model_path,
            'export_module': "detectnet_v2",
            'key': ENC_KEY,
            "cal_cache_file": cal_cache_file,
            "cal_image_dir": "",
            "cal_data_file": tensorfile_path,
            "batch_size": batch_size,
            "batches": 2,
            "data_type": data_type,
            "output_file": output_file,
            "max_workspace_size": DEFAULT_MAX_WORKSPACE_SIZE,
            "max_batch_size": DEFAULT_MAX_BATCH_SIZE,
            "verbose": False,
            "engine_file": engine_file,
            "strict_type_constraints": True,
            "static_batch_size": -1,
            "force_ptq": False,
            "gen_ds_config": gen_ds_config,
            "min_batch_size": batch_size,
            "opt_batch_size": batch_size,
            "target_opset": 12,
            "cal_json_file": cal_json_file
        }
        if backend == "onnx":
            exporter_args["onnx_route"] = "tf2onnx"
        # Choose whether to calibrate from the spec file or not.
        if from_spec:
            exporter_args["experiment_spec"] = training_spec
        else:
            exporter_args["experiment_spec"] = None

        try:
            self._common(exporter_args, backend)
        except AssertionError:
            raise AssertionError("Exporter failed.")
        finally:
            self.clear_previous_files()
