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
"""TransformedExample examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from nvidia_tao_tf1.blocks.multi_source_loader.types.transformed_example import (
    TransformedExample,
)
from nvidia_tao_tf1.core.types import Example

FEATURE_CAMERA = "CAMERA"
FEATURE_SESSION = "SESSION"
LABEL_MAP = "MAP"
LABEL_OBJECT = "OBJECT"
LABEL_FREESPACE_REGRESSION = LABEL_MAP
# TODO(ehall): Add a configurable mapping from POLYGON->LABEL_FREESPACE_SEGMENTATION
LABEL_FREESPACE_SEGMENTATION = LABEL_MAP
# TODO(ehall): Add a configurable mapping from POLYGON->LABEL_PANOPTIC_SEGMENTATION
LABEL_PANOPTIC_SEGMENTATION = LABEL_MAP
# TODO(vkallioniemi): Add a configurable mapping from POLYLINE->LABEL_PATH
LABEL_PATH = LABEL_MAP
LABEL_DEPTH_FREESPACE = "DEPTH_FREESPACE"
LABEL_DEPTH_DENSE_MAP = "DEPTH_DENSE_MAP"


# This class and the associated TransformedExample class borrow ideas from the "lift-lower" pattern
# presented in this podcast:
# https://lispcast.com/a-cool-functional-programming-pattern-do-you-know-what-to-call-it/
#
# The basic idea of the pattern is to temporarily transform values to a richer type that makes
# manipulating them easier:
# 1. Lift the type to a richer type (TransformedExample) to make transformations easier.
# 2. Perform transforms on the TransformedExample
# 3. Lower the type back to the original type when apply gets called.
class SequenceExample(namedtuple("SequenceExample", Example._fields)):
    """SequenceExample - a collection of features and labels passed to the model + loss.

    Args:
        instances (dict): Data that will be transformed to features that are input to a model.
        labels (dict): Labels are transformed to targets that work as inputs to a model loss.
    """

    def transform(self, transformation):
        """Delayed transform of this value."""
        return TransformedExample(example=self, transformation=transformation)
