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

"""Build a DetectNet V2 model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.cv.detectnet_v2.model.detectnet_model import GridboxModel
from nvidia_tao_tf1.cv.detectnet_v2.model.tensorrt_detectnet_model import TensorRTGridboxModel
from nvidia_tao_tf1.cv.detectnet_v2.proto.model_config_pb2 import ModelConfig


def select_model_proto(experiment_spec):
    """Select the model proto depending on type defined in the spec.

    Args:
        experiment_spec: nvidia_tao_tf1.cv.detectnet_v2.proto.experiment proto message.
    Returns:
        model_proto (ModelConfig):
    Raises:
        ValueError: If model_config_type is not valid.
    """
    return experiment_spec.model_config


def get_base_model_config(experiment_spec):
    """Get the model config from the experiment spec.

    Args:
        experiment_spec: nvidia_tao_tf1.cv.detectnet_v2.proto.experiment proto message.
    Returns:
        model_config (ModelConfig): Model configuration proto.
    Raises:
        ValueError: If model config proto of the given experiment spec is of unknown type.
    """
    model_config = select_model_proto(experiment_spec)
    if isinstance(model_config, ModelConfig):
        return model_config
    raise ValueError("Model config is of unknown type.")


def build_model(m_config, target_class_names, enable_qat=False, framework="tlt"):
    """Build a DetectNet V2 model.

    The model is a GridboxModel or a TensorRTGridboxModel instance.

    Arguments:
        m_config (ModelConfig): Model configuration proto.
        target_class_names (list): A list of target class names.
        enable_qat (bool): Flag to enable tlt model to qat model conversion.
        framework (str): Model backend framework.
            Choices: ["tlt", "tensorrt"]. Default "tlt".

    Returns:
        A DetectNet V2 model. By default, a GridboxModel instance with resnet feature extractor
        is returned.
    """
    # model_config.num_layers is checked during GridboxModel.construct_model. Only certain values
    # are supported.

    # Initial dictionary of the arguments for building the model.
    model_constructor_arguments = dict()

    assert isinstance(m_config, ModelConfig),\
        "Unsupported model_proto message."

    # Check sanity of the parameters.
    if m_config.dropout_rate < 0.0 or m_config.dropout_rate > 1.0:
        raise ValueError("ModelConfig.dropout_rate must be >= 0 and <= 1")
    if target_class_names is None or not target_class_names:
        raise ValueError("target_class_names must contain at least one class")
    if m_config.freeze_pretrained_layers:
        assert m_config.pretrained_model_file, \
            "Freezing layers makes only sense if pretrained model is loaded."
    if m_config.freeze_blocks:
        assert m_config.pretrained_model_file, \
            "Freeze blocks is only possible if a pretrained model file is provided."

    assert framework in ["tlt", "tensorrt"], (
        "Detectnet model only supports either tlt or tensorrt frameworks."
        "Unsupported framework '{}' encountered.".format(framework)
    )

    # Common model building arguments for all model types.
    args = {'num_layers': m_config.num_layers if m_config.num_layers else 18,
            'use_pooling': m_config.use_pooling,
            'use_batch_norm': m_config.use_batch_norm,
            'dropout_rate': m_config.dropout_rate if m_config.dropout_rate else 0.0,
            'objective_set_config': m_config.objective_set,
            'activation_config': m_config.activation,
            'target_class_names': target_class_names,
            'freeze_pretrained_layers': m_config.freeze_pretrained_layers,
            'freeze_blocks': m_config.freeze_blocks if m_config.freeze_blocks else None,
            'freeze_bn': m_config.freeze_bn,
            'allow_loaded_model_modification': m_config.allow_loaded_model_modification,
            'all_projections': m_config.all_projections,
            'enable_qat': enable_qat}

    # Switch to default template if feature extractor template is missing.
    if not m_config.arch:
        pass
    else:
        args['template'] = m_config.arch

    model_constructor_arguments.update(args)

    # Defining model instance class.
    model_class = GridboxModel
    if framework == "tensorrt":
        model_class = TensorRTGridboxModel

    return model_class(**model_constructor_arguments)
