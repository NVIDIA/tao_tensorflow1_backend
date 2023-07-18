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

"""Load an experiment spec file to run GridBox training, evaluation, pruning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from google.protobuf.text_format import Merge as merge_text_proto
import nvidia_tao_tf1.cv.unet.proto.experiment_pb2 as experiment_pb2

logger = logging.getLogger(__name__)


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
        if not os.path.exists(filename):
            raise IOError("Specfile not found at: {}".format(filename))
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


def load_experiment_spec(spec_path=None, merge_from_default=True):
    """Load experiment spec from a .txt file and return an experiment_pb2.Experiment object.

    Args:
        spec_path (str): location of a file containing the custom experiment spec proto.
        dataset_export_spec_paths (list of str): paths to the dataset export specs.
        merge_from_default (bool): disable default spec, if False, spec_path must be set.

    Returns:
        experiment_spec: protocol buffer instance of type experiment_pb2.Experiment.
    """
    experiment_spec = experiment_pb2.Experiment()
    file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_spec_path = os.path.join(file_path, 'experiment_specs/default_spec.txt')
    experiment_spec = load_proto(spec_path, experiment_spec, default_spec_path,
                                 merge_from_default)

    return experiment_spec
