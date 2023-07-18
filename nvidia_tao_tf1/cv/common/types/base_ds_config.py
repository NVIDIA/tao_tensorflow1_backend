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

"""Configuration element for a DeepStream graph."""

VALID_COLOR_FORMATS = ["rgb", "bgr", "l"]
VALID_CHANNEL_ORDERS = ["channels_first", "channels_last"]
VALID_BACKENDS = ["uff", "onnx"]
VALID_NETWORK_TYPES = [0, 1, 2, 3, 100]


class BaseDSConfig(object):
    """Configuration element for an nvinfer ds plugin."""

    def __init__(self, scale, offsets, infer_dims,
                 color_format, key, network_type=0,
                 input_names=None, num_classes=None,
                 output_names=None, data_format="channels_first",
                 backend="uff", maintain_aspect_ratio=False,
                 output_tensor_meta=False):
        """Generate a Deepstream config element.

        Args:
            scale (float): Scale value to normalize the input.
            offsets (tuple): Tuple of floats for channels wise mean subtraction.
            infer_dims (tuple): Input dimensions of the model.
            color_format (str): Format of the color to be running inference on.
            key (str): Key to load the model.
            network_type (int): Type of model.
            input_names (list): List of input names.
            num_classes (int): Number of classes.
            output_names (list): List of output names.
            data_format (str): Format of the input data.
            backend (str): Backend format of the model.

        Returns:
            BaseDSConfig: Instance of BaseDSConfig element.
        """
        self.scale = scale
        self.offsets = offsets
        self.input_names = input_names
        self.output_names = output_names
        self.backend = backend
        self.infer_dims = infer_dims
        self.key = key
        self.network_type = network_type
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.output_tensor_meta = output_tensor_meta
        assert self.network_type in VALID_NETWORK_TYPES, (
            "Invalid Network type {} requested. Supported network types: {}".format(
                self.network_type, VALID_NETWORK_TYPES
            )
        )
        self.color_format = color_format.lower()
        if self.color_format not in VALID_COLOR_FORMATS:
            raise NotImplementedError(
                "Color format specified is not valid: {}. "
                "Valid color formats include {}".format(
                    color_format.lower(),
                    VALID_COLOR_FORMATS
                )
            )
        self.data_format = data_format
        if self.data_format not in VALID_CHANNEL_ORDERS:
            raise NotImplementedError("Invalid data format {} encountered.".format(
                data_format, VALID_CHANNEL_ORDERS
            ))
        self.channel_index = 0
        if data_format == "channels_last":
            self.channel_index = -1
        if self.color_format == "l":
            assert self.infer_dims[self.channel_index] == 1, (
                "Channel count mismatched with color_format. "
                "Provided\ndata_format: {}\n color_format: {}".format(
                    self.infer_dims[self.channel_index], self.color_format
                )
            )
        self.num_classes = num_classes
        self.initialized = True

    def get_config(self):
        """Generate config elements."""
        config_dict = {
            "net-scale-factor": self.scale,
            "offsets": ";".join([str(offset) for offset in self.offsets]),
            "infer-dims": ";".join([str(int(dims)) for dims in self.infer_dims]),
            "tlt-model-key": self.key,
            "network-type": self.network_type,
        }
        # Number of classes.
        if self.num_classes is not None:
            config_dict["num-detected-classes"] = self.num_classes

        if self.backend == "uff":
            assert self.input_names is not None, (
                "Input blob names cannot be None for a UFF model."
            )
            assert self.output_names is not None, (
                "Output blob names cannot be None for a UFF model."
            )
            config_dict.update(
                {
                    "uff-input-order": "0" if self.channel_index == 0 else "1",
                    "output-blob-names": ";".join([blob for blob in self.output_names]),
                    "uff-input-blob-name": ";".join([blob for blob in self.input_names])
                }
            )
        if self.infer_dims[self.channel_index] == 3:
            config_dict["model-color-format"] = 0
            if self.color_format == "bgr":
                config_dict["model-color-format"] = 1
        else:
            config_dict["model-color-format"] = 2

        if self.maintain_aspect_ratio:
            config_dict["maintain-aspect-ratio"] = 1
        else:
            config_dict["maintain-aspect-ratio"] = 0

        if self.output_tensor_meta:
            config_dict["output-tensor-meta"] = 1
        else:
            config_dict["output-tensor-meta"] = 0

        return config_dict

    def __str__(self):
        """Return the string data."""
        if not self.initialized:
            raise RuntimeError("Class wasn't initialized.")
        config_dict = self.get_config()
        config_string = ""
        for key, val in config_dict.items():
            config_string += "{}={}\n".format(key, str(val))
        return config_string
