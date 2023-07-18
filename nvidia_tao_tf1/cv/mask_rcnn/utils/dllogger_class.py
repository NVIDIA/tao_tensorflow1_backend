"""DLLogger utils."""
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

from dllogger import JSONStreamBackend, Logger, StdOutBackend, Verbosity
from nvidia_tao_tf1.cv.mask_rcnn.utils.metaclasses import SingletonMetaClass


class dllogger_class(metaclass=SingletonMetaClass):
    """DLLogger singleton class."""

    def format_step(self, step):
        """Format logging step."""
        if isinstance(step, str):
            return step
        s = ""
        if len(step) > 0:
            s += "Iteration: {} ".format(step[0])
        if len(step) > 1:
            s += "Validation Iteration: {} ".format(step[1])
        if len(step) > 2:
            s += "Epoch: {} ".format(step[2])
        return s

    def __init__(self):
        """Initialize."""
        self.logger = Logger([
            StdOutBackend(Verbosity.DEFAULT, step_format=self.format_step),
            JSONStreamBackend(Verbosity.VERBOSE, "mrcnn_log.json"),
            ])
        self.logger.metadata("loss", {"format": ":.4f", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
        self.logger.metadata("val.loss", {"format": ":.4f", "GOAL": "MINIMIZE", "STAGE": "VAL"})
        self.logger.metadata(
            "speed",
            {"unit": "images/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"},
        )


dllogging = dllogger_class()
