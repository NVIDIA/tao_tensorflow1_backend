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

"""Test detectnet_v2 spec file loader and validator."""

import os

from google.protobuf.text_format import ParseError
import pytest

from nvidia_tao_tf1.cv.detectnet_v2.spec_handler.spec_loader import load_experiment_spec

detectnet_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))
    )
)
gt_training_spec = os.path.join(
    detectnet_root, "experiment_specs/default_spec.txt"
)
gt_inference_spec = os.path.join(
    detectnet_root, "experiment_specs/inferencer_spec_etlt.prototxt"
)

topologies = [
    ("train_val", gt_training_spec),
    ("inference", gt_inference_spec),
    ("train_val", gt_inference_spec),
    ("inference", gt_training_spec),
]


class TestDetectnetSpecloader():
    """Simple class to test the specification file loader."""

    @pytest.mark.parametrize(
        "schema_validation, spec_file_path",  # noqa: E501
        topologies
    )
    def test_spec_loader(self, schema_validation, spec_file_path):
        """Load and check if the spec file validator throws an error."""
        error_raises = False
        if schema_validation == "train_val":
            if os.path.basename(spec_file_path) == "inferencer_spec_etlt.prototxt":
                error_raises = True
        elif schema_validation == "inference":
            if os.path.basename(spec_file_path) == "default_spec.txt":
                error_raises = True
        if error_raises:
            with pytest.raises((AssertionError, ParseError)):
                load_experiment_spec(spec_path=spec_file_path,
                                     merge_from_default=False,
                                     validation_schema=schema_validation)
        else:
            load_experiment_spec(
                spec_path=spec_file_path,
                merge_from_default=False,
                validation_schema=schema_validation)
