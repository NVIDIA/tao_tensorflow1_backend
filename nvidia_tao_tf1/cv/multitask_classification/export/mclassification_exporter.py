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

"""Base class to export trained .tlt models to etlt file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from nvidia_tao_tf1.cv.common.export.keras_exporter import KerasExporter as Exporter
from nvidia_tao_tf1.cv.common.types.base_ds_config import BaseDSConfig

logger = logging.getLogger(__name__)


class MClassificationExporter(Exporter):
    """Define an exporter for classification models."""

    def __init__(self,
                 output_tasks,
                 model_path=None,
                 key=None,
                 data_type="fp32",
                 strict_type=False,
                 backend="uff",
                 **kwargs):
        """Initialize the classification exporter.

        Args:
            outout_task (List of str): The task names for outputs.
            model_path (str): Path to the model file.
            key (str): Key to load the model.
            data_type (str): Path to the TensorRT backend data type.
            strict_type(bool): Apply TensorRT strict_type_constraints or not for INT8 mode.
            backend (str): TensorRT parser to be used.

        Returns:
            None.
        """
        super(MClassificationExporter, self).__init__(model_path=model_path,
                                                      key=key,
                                                      data_type=data_type,
                                                      strict_type=strict_type,
                                                      backend=backend)
        self.output_tasks = output_tasks

    def set_input_output_node_names(self):
        """Set input output node names."""
        self.output_node_names = [n+"/Softmax" for n in self.output_tasks]
        self.input_node_names = ["input_1"]

    def generate_ds_config(self, input_dims, num_classes=None):
        """Generate Deepstream config element for the exported model."""
        if input_dims[0] == 1:
            color_format = "l"
        else:
            color_format = "bgr" if self.preprocessing_arguments["flip_channel"] else "rgb"
        kwargs = {
            "data_format": self.data_format,
            "backend": self.backend,
            # Setting this to 0 by default because there are more
            # detection networks.
            "network_type": 1
        }
        if num_classes:
            kwargs["num_classes"] = num_classes
        if self.backend == "uff":
            kwargs.update({
                "input_names": self.input_node_names,
                "output_names": self.output_node_names
            })

        ds_config = BaseDSConfig(
            self.preprocessing_arguments["scale"],
            self.preprocessing_arguments["means"],
            input_dims,
            color_format,
            self.key,
            **kwargs
        )
        return ds_config
