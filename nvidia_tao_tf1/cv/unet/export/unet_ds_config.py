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

"""Configuration unet derived class for a DeepStream graph."""

from nvidia_tao_tf1.cv.common.types.base_ds_config import BaseDSConfig


class UnetDSConfig(BaseDSConfig):
    """Configuration element for an nvinfer ds plugin."""

    def __init__(self, *args, segmentation_threshold=None, output_blob_names=None,
                 segmentation_output_order=None, **kwargs):
        """Init function.

        Args:
            num_layers (int): Number of layers for scalable feature extractors.
            use_pooling (bool): Whether to add pooling layers to the feature extractor.
            use_batch_norm (bool): Whether to add batch norm layers.
            dropout_rate (float): Fraction of the input units to drop. 0.0 means dropout is
                not used.
            target_class_names (list): A list of target class names.
            freeze_pretrained_layers (bool): Prevent updates to pretrained layers' parameters.
            allow_loaded_model_modification (bool): Allow loaded model modification.
            template (str): Model template to use for feature extractor.
            freeze_bn (bool): The boolean to freeze BN or not.
            load_graph (bool): The boolean to laod graph for phase 1.
            segmentation_threshold (float): Threshold to classify a mask.
            output_blob_names (str): Output name of the model graph.
            segmentation_output_order (str): The output order it channel last.
        """
        super(UnetDSConfig, self).__init__(*args, **kwargs)

        self.segmentation_threshold = segmentation_threshold
        self.output_blob_names = output_blob_names
        self.segmentation_output_order = segmentation_output_order

    def get_config(self):
        """Generate config elements."""

        config_dict = super().get_config()
        config_dict.update({"segmentation-threshold": self.segmentation_threshold,
                            "output-blob-names": self.output_blob_names,
                            "segmentation-output-order": self.segmentation_output_order})

        return config_dict
