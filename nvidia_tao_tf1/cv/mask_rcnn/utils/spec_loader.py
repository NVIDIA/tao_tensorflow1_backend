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

"""Load an experiment spec file to run MaskRCNN training, evaluation, export."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from google.protobuf.text_format import Merge as merge_text_proto

import nvidia_tao_tf1.cv.mask_rcnn.proto.experiment_pb2 as experiment_pb2

logger = logging.getLogger(__name__)


def load_proto(spec_path, proto_buffer):
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

    # Merge a custom proto on top of the default spec, if given
    if spec_path:
        logger.info("Loading specification from %s", spec_path)
        _load_from_file(spec_path, proto_buffer)

    return proto_buffer


def load_experiment_spec(spec_path=None, merge_from_default=False):
    """Load experiment spec from a .txt file and return an experiment_pb2.Experiment object.

    Args:
        spec_path (str): location of a file containing the custom experiment spec proto.
        dataset_export_spec_paths (list of str): paths to the dataset export specs.
        merge_from_default (bool): disable default spec, if False, spec_path must be set.

    Returns:
        experiment_spec: protocol buffer instance of type experiment_pb2.Experiment.
    """
    experiment_spec = experiment_pb2.Experiment()
    experiment_spec = load_proto(spec_path, experiment_spec)

    return experiment_spec
