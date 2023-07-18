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

"""Load experiment spec .txt files and return an experiment_pb2.Experiment object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from google.protobuf.text_format import Merge as merge_text_proto

import nvidia_tao_tf1.cv.faster_rcnn.proto.experiment_pb2 as experiment_pb2

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')
logger = logging.getLogger(__name__)


def _load_proto(spec_path, proto_buffer):
    """Load spec from file and merge with given proto_buffer instance.

    Args:
        spec_path (str): Location of a file containing the custom spec proto.
        proto_buffer (pb2): Protocal buffer instance to be loaded.

    Returns:
        proto_buffer (pb2): Protocol buffer instance updated with spec.
    """
    with open(spec_path, "r") as f:
        merge_text_proto(f.read(), proto_buffer)

    return proto_buffer


def load_experiment_spec(spec_path=None):
    """Load experiment spec from a .txt file and return an experiment_pb2.Experiment object.

    Args:
        spec_path (str): Location of a file containing the custom experiment spec proto. If None,
            then a default spec is used.

    Returns:
        experiment_spec: protocol buffer instance of type experiment_pb2.Experiment.
    """
    experiment_spec = experiment_pb2.Experiment()

    if spec_path is None:
        file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        default_spec_path = os.path.join(file_path, 'experiment_spec/default_spec.txt')
        spec_path = default_spec_path

    logger.info("Loading experiment spec at %s.", spec_path)

    experiment_spec = _load_proto(spec_path, experiment_spec)

    return experiment_spec


def write_spec_to_disk(experiment_spec, path):
    """Write experiment_pb2.Experiment object to a text file to the given path."""
    # Create the path if it does not exist yet.
    dirname = os.path.dirname(path)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(path, 'wb') as f:
        f.write(str.encode(str(experiment_spec)))
