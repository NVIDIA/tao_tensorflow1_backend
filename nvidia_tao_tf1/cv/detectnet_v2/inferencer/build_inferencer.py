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

"""Simple script to build inferencer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from nvidia_tao_tf1.cv.detectnet_v2.inferencer.tlt_inferencer import TLTInferencer
from nvidia_tao_tf1.cv.detectnet_v2.inferencer.trt_inferencer import DEFAULT_MAX_WORKSPACE_SIZE
from nvidia_tao_tf1.cv.detectnet_v2.inferencer.trt_inferencer import TRTInferencer

SUPPORTED_INFERENCERS = {'tlt': TLTInferencer,
                         'tensorrt': TRTInferencer}

TRT_PARSERS = {0: 'etlt',
               1: 'uff',
               2: 'caffe'}

TRT_BACKEND_DATATYPE = {0: "fp32",
                        1: "fp16",
                        2: "int8"}

logger = logging.getLogger(__name__)


def build_inferencer(inf_config=None, verbose=True, key=None):
    """Simple function to build inferencer.

    The function looks at the inference framework mentioned and then calls the right
    inferencer.

    Args:
        inf_config (InferencerConfig protobuf): Config container parameters to configure
            the inferencer
        verbose (bool): Flag to define the verbosity of the logger.
        key (str): Key to load the model.

    Returns:
        model(tlt_inferencer/trt_inferencer object): The inferencer object for the respective
            framework.

    Raises:
        NotImplementedError for the wrong frameworks.
    """
    if key is None:
        raise ValueError("The key to load a model cannot be set to None.")

    # Setting up common constructor arguments
    constructor_kwargs = {'batch_size': inf_config.batch_size if inf_config.batch_size else 1,
                          'gpu_set': inf_config.gpu_index if inf_config.gpu_index else 0,
                          'target_classes': inf_config.target_classes if inf_config.target_classes
                          else None,
                          'image_height': inf_config.image_height,
                          'image_width': inf_config.image_width,
                          'image_channels': inf_config.image_channels}

    # Extracting framework specific inferencer parameters.
    model_config_type = inf_config.WhichOneof('model_config_type')
    config = getattr(inf_config, model_config_type)

    if model_config_type == 'tlt_config':
        # Setting up tlt inferencer based on the config file parameters
        logger.debug("Initializing TLT inferencer.")
        framework = 'tlt'
        constructor_kwargs.update({'tlt_model': config.model,
                                   'enc_key': key,
                                   'framework': framework})
    elif model_config_type == 'tensorrt_config':
        # Setting up tensorrt inferencer based on the config file parameters.
        logger.debug("Initializing Tensorrt inferencer.")
        framework = 'tensorrt'
        constructor_kwargs.update({'framework': framework,
                                   'uff_model': config.uff_model if config.uff_model else None,
                                   'caffemodel': config.caffemodel if config.caffemodel else None,
                                   'prototxt': config.prototxt if config.prototxt else None,
                                   'etlt_model': config.etlt_model if config.etlt_model else None,
                                   'etlt_key': key,
                                   'parser': TRT_PARSERS[config.parser],
                                   'verbose': verbose,
                                   'max_workspace_size': DEFAULT_MAX_WORKSPACE_SIZE,
                                   'data_type': TRT_BACKEND_DATATYPE[config.backend_data_type],
                                   'trt_engine': config.trt_engine if config.trt_engine else None,
                                   'save_engine': config.save_engine})

        # Setting up calibrator if calibrator specific parameters are present.
        if TRT_BACKEND_DATATYPE[config.backend_data_type] == "int8":
            assert hasattr(config, "calibrator_config"), "Please instantiate an calibrator config "\
                "when running in int8 mode."

            calib_conf = getattr(config, "calibrator_config")
            # Set calibrator config arguments.
            n_batches = 1
            if calib_conf.n_batches:
                n_batches = calib_conf.n_batches
            calibration_cache = None
            if calib_conf.calibration_cache:
                calibration_cache = calib_conf.calibration_cache
            calibration_tensorfile = None
            if calib_conf.calibration_tensorfile:
                calibration_tensorfile = calib_conf.calibration_tensorfile

            constructor_kwargs.update({'calib_tensorfile': calibration_tensorfile,
                                       'n_batches': n_batches,
                                       'calib_file': calibration_cache})
    else:
        raise NotImplementedError("Unsupported framework: {}".format(model_config_type))

    logger.info("Constructing inferencer")
    return framework, SUPPORTED_INFERENCERS[framework](**constructor_kwargs)
