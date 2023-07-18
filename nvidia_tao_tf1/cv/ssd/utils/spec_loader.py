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

"""Load an experiment spec file to run SSD training, evaluation, pruning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from google.protobuf.text_format import Merge as merge_text_proto
from nvidia_tao_tf1.cv.common.spec_validator import eval_str, length, SpecValidator, ValueChecker
import nvidia_tao_tf1.cv.ssd.proto.experiment_pb2 as experiment_pb2

logger = logging.getLogger(__name__)


_SSD_OPTIONAL_CHEKER = {"aspect_ratios": ValueChecker("!=", ""),
                        "aspect_ratios_global": ValueChecker("!=", ""),
                        "scales": ValueChecker("!=", ""),
                        "steps": ValueChecker("!=", ""),
                        "offsets": ValueChecker("!=", "")}

_SSD_VALUE_CHECKER_ = {"aspect_ratios": [ValueChecker(">", 0, length, "The length of "),
                                         ValueChecker(">", 0, eval_str)],
                       "aspect_ratios_global": [ValueChecker(">", 0, length, "The length of "),
                                                ValueChecker(">", 0, eval_str)],
                       "scales": [ValueChecker(">", 0, length, "The length of "),
                                  ValueChecker(">", 0, eval)],
                       "steps": [ValueChecker(">", 0, length, "The length of "),
                                 ValueChecker(">", 0, eval)],
                       "offsets": [ValueChecker(">", 0, length, "The length of "),
                                   ValueChecker(">", 0, eval),
                                   ValueChecker("<", 1.0, eval)],
                       "variances": [ValueChecker("!=", ""),
                                     ValueChecker("=", 4, length, "The length of "),
                                     ValueChecker(">", 0, eval)],
                       "arch": [ValueChecker("!=", ""),
                                ValueChecker("in", ["resnet",
                                                    "vgg",
                                                    "darknet",
                                                    "mobilenet_v1",
                                                    "mobilenet_v2",
                                                    "squeezenet",
                                                    "googlenet",
                                                    "efficientnet_b0",
                                                    "efficientnet_b1"])],
                       "nlayers": [ValueChecker(">=", 0)],
                       "batch_size_per_gpu": [ValueChecker(">", 0)],
                       "num_epochs": [ValueChecker(">", 0)],
                       "min_learning_rate": [ValueChecker(">", 0)],
                       "max_learning_rate": [ValueChecker(">", 0)],
                       "soft_start": [ValueChecker(">", 0), ValueChecker("<", 1.0)],
                       "annealing": [ValueChecker(">", 0), ValueChecker("<", 1.0)],
                       "validation_period_during_training": [ValueChecker(">", 0)],
                       "batch_size": [ValueChecker(">", 0)],
                       "matching_iou_threshold": [ValueChecker(">", 0),
                                                  ValueChecker("<", 1.0)],
                       "confidence_threshold": [ValueChecker(">", 0),
                                                ValueChecker("<", 1.0)],
                       "clustering_iou_threshold": [ValueChecker(">", 0),
                                                    ValueChecker("<", 1.0)],
                       "top_k": [ValueChecker(">", 0)],
                       "output_width": [ValueChecker(">", 32)],
                       "output_height": [ValueChecker(">", 32)],
                       "output_channel": [ValueChecker("in", [1, 3])],
                       "monitor": [ValueChecker("in", ["loss", "validation_loss", "val_loss"])],
                       "min_delta": [ValueChecker(">=", 0)],
                       "patience": [ValueChecker(">=", 0)],
                       "checkpoint_interval": [ValueChecker(">=", 0)]}


TRAIN_EXP_REQUIRED_MSG = ["ssd_config", "training_config", "eval_config",
                          "augmentation_config", "nms_config", "dataset_config"]
EVAL_EXP_REQUIRED_MSG = ["ssd_config", "eval_config", "nms_config"
                         "augmentation_config", "dataset_config"]
INFERENCE_EXP_REQUIRED_MSG = ["ssd_config", "eval_config", "nms_config"
                              "augmentation_config", "dataset_config"]
EXPORT_EXP_REQUIRED_MSG = ["ssd_config", "nms_config"]


_REQUIRED_MSG_ = {"training_config": ["learning_rate", "regularizer"],
                  "learning_rate": ["soft_start_annealing_schedule"],
                  "soft_start_annealing_schedule": ["min_learning_rate",
                                                    "max_learning_rate",
                                                    "soft_start",
                                                    "annealing"],
                  "dataset_config": ["target_class_mapping"]}


def spec_validator(spec, required_msg=None, ssd_spec_validator=None):
    """do spec validation for SSD/DSSD."""
    if required_msg is None:
        required_msg = []
    if ssd_spec_validator is None:
        ssd_spec_validator = SpecValidator(required_msg_dict=_REQUIRED_MSG_,
                                           value_checker_dict=_SSD_VALUE_CHECKER_,
                                           option_checker_dict=_SSD_OPTIONAL_CHEKER)
    ssd_spec_validator.validate(spec, required_msg)


def validate_train_spec(spec):
    """do spec validation check for training spec."""
    # @TODO(tylerz): workaround for one-of behavior to check train dataset existence
    ssd_spec_validator = SpecValidator(required_msg_dict=_REQUIRED_MSG_,
                                       value_checker_dict=_SSD_VALUE_CHECKER_,
                                       option_checker_dict=_SSD_OPTIONAL_CHEKER)
    ssd_spec_validator.required_msg_dict["dataset_config"].append("data_sources")
    if spec.dataset_config.data_sources[0].tfrecords_path == "":
        ssd_spec_validator.value_checker_dict["label_directory_path"] = [ValueChecker("!=", "")]
        ssd_spec_validator.value_checker_dict["image_directory_path"] = [ValueChecker("!=", "")]
    # Remove the empty validation dataset to skip the check
    for idx in range(len(spec.dataset_config.validation_data_sources)):
        if spec.dataset_config.validation_data_sources[idx].image_directory_path == "" or \
                spec.dataset_config.validation_data_sources[idx].label_directory_path == "":
            del spec.dataset_config.validation_data_sources[idx]

    spec_validator(spec, TRAIN_EXP_REQUIRED_MSG, ssd_spec_validator)


def validate_eval_spec(spec):
    """do spec validation check for training spec."""
    # @TODO(tylerz): workaround for one-of behavior to check validation dataset existence
    ssd_spec_validator = SpecValidator(required_msg_dict=_REQUIRED_MSG_,
                                       value_checker_dict=_SSD_VALUE_CHECKER_,
                                       option_checker_dict=_SSD_OPTIONAL_CHEKER)
    ssd_spec_validator.required_msg_dict["dataset_config"].append("validation_data_sources")
    ssd_spec_validator.value_checker_dict["label_directory_path"] = [ValueChecker("!=", "")]
    ssd_spec_validator.value_checker_dict["image_directory_path"] = [ValueChecker("!=", "")]
    # Skip the check for label and image in data_sources
    # cause we don't care training dataset in evaluation
    for idx in range(len(spec.dataset_config.data_sources)):
        spec.dataset_config.data_sources[idx].image_directory_path = "./fake_dir"
        spec.dataset_config.data_sources[idx].label_directory_path = "./fake_dir"

    spec_validator(spec, EVAL_EXP_REQUIRED_MSG, ssd_spec_validator)


def load_proto(spec_path, proto_buffer, default_spec_path=None, merge_from_default=True):
    """Load spec from file and merge with given proto_buffer instance.

    Args:
        spec_path (str): location of a file containing the custom spec proto.
        proto_buffer(pb2): protocal buffer instance to be loaded.
        default_spec_path(str): location of default spec to use if merge_from_default is True.
        merge_from_default (bool): disable default spec, if False, spec_path must be set.

    Returns:
        proto_buffer(pb2): protocol buffer instance updated with spec.
    """
    def _load_from_file(filename, pb2):
        with open(filename, "r") as f:
            merge_text_proto(f.read(), pb2)

    # Setting this flag false prevents concatenating repeated-fields
    if merge_from_default:
        assert default_spec_path, \
               "default spec path has to be defined if merge_from_default is enabled"
        # Load the default spec
        _load_from_file(default_spec_path, proto_buffer)
    else:
        assert spec_path, "spec_path has to be defined, if merge_from_default is disabled"

    # Merge a custom proto on top of the default spec, if given
    if spec_path:
        logger.info("Merging specification from %s", spec_path)
        _load_from_file(spec_path, proto_buffer)

    return proto_buffer


def load_experiment_spec(spec_path=None, arch_check=None):
    """Load experiment spec from a .txt file and return an experiment_pb2.Experiment object.

    Args:
        spec_path (str): location of a file containing the custom experiment spec proto.
        dataset_export_spec_paths (list of str): paths to the dataset export specs.

    Returns:
        experiment_spec: protocol buffer instance of type experiment_pb2.Experiment. with network
            config always in ssd_config message.
        is_dssd: build dssd network?
    """

    merge_from_default = (spec_path is None)
    if merge_from_default:
        print("No spec file passed in. Loading default experiment spec!!!")
    experiment_spec = experiment_pb2.Experiment()
    file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_spec_path = os.path.join(file_path, 'experiment_specs/default_spec.txt')
    experiment_spec = load_proto(spec_path, experiment_spec, default_spec_path,
                                 merge_from_default)

    network_arch = experiment_spec.WhichOneof('network')
    assert network_arch is not None, 'Network config missing in spec file.'

    experiment_spec.ssd_config.CopyFrom(getattr(experiment_spec, network_arch))

    network_arch = network_arch.split('_')[0]

    assert arch_check is None or arch_check.lower() == network_arch, \
        'The spec file specifies %s but you typed %s in command line.' % (network_arch, arch_check)

    return experiment_spec, network_arch == 'dssd'
