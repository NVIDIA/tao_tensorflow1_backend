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

"""Build a dataset converter object based on dataset export config."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import nvidia_tao_tf1.cv.common.logging.logging as status_logging
from nvidia_tao_tf1.cv.detectnet_v2.dataio.build_sample_modifier import build_sample_modifier

from nvidia_tao_tf1.cv.detectnet_v2.dataio.coco_converter_lib import COCOConverter
from nvidia_tao_tf1.cv.detectnet_v2.dataio.kitti_converter_lib import KITTIConverter


logger = logging.getLogger(__name__)


def build_converter(dataset_export_config, output_filename, validation_fold=None):
    """Build a DatasetConverter object.

    Build and return an object of desired subclass of DatasetConverter based on
    given dataset export configuration.

    Args:
        dataset_export_config (DatasetExportConfig): Dataset export configuration object
        output_filename (string): Path for the output file.
        validation_fold (int): Optional. Validation fold number (0-based). Indicates which
            partition is the validation fold. If samples are modified, then the modifications
            are applied only to the training set, while the validation set remains unchanged.

    Return:
        converter (DatasetConverter): An object of desired subclass of DatasetConverter.
    """
    convert_config_type = dataset_export_config.WhichOneof('convert_config_type')
    config = getattr(dataset_export_config, convert_config_type)

    if convert_config_type == "kitti_config":
        # Fetch the dataset configuration object first
        # This can be extended for other data formats such as Pascal VOC,
        # or open Images etc. For GA, we will stick to kitti configuration.
        config = getattr(dataset_export_config, "kitti_config")

        constructor_kwargs = {'root_directory_path': config.root_directory_path,
                              'partition_mode': config.partition_mode,
                              'num_partitions': config.num_partitions,
                              'num_shards': config.num_shards,
                              'output_filename': output_filename}

        # Create a SampleModifier
        sample_modifier_config = dataset_export_config.sample_modifier_config
        sample_modifier = build_sample_modifier(
            sample_modifier_config=sample_modifier_config,
            validation_fold=validation_fold,
            num_folds=config.num_partitions)

        constructor_kwargs['sample_modifier'] = sample_modifier
        constructor_kwargs['image_dir_name'] = config.image_dir_name
        constructor_kwargs['label_dir_name'] = config.label_dir_name
        # Those two directories are by default empty string in proto
        # Here we do some check to make them default to None(in constructor)
        # Otherwise it will raise error if we pass the empty strings
        # directly to constructors.
        if config.point_clouds_dir:
            constructor_kwargs['point_clouds_dir'] = config.point_clouds_dir
        if config.calibrations_dir:
            constructor_kwargs['calibrations_dir'] = config.calibrations_dir
        constructor_kwargs['extension'] = config.image_extension or '.png'
        constructor_kwargs['val_split'] = config.val_split
        if config.kitti_sequence_to_frames_file:
            constructor_kwargs['kitti_sequence_to_frames_file'] = \
                config.kitti_sequence_to_frames_file

        logger.info("Instantiating a kitti converter")
        status_logging.get_status_logger().write(
            message="Instantiating a kitti converter",
            status_level=status_logging.Status.STARTED)
        converter = KITTIConverter(**constructor_kwargs)

        return converter
    if convert_config_type == "coco_config":
                # Fetch the dataset configuration object first
        # This can be extended for other data formats such as Pascal VOC,
        # or open Images etc. For GA, we will stick to kitti configuration.
        config = getattr(dataset_export_config, "coco_config")

        constructor_kwargs = {'root_directory_path': config.root_directory_path,
                              'num_partitions': config.num_partitions,
                              'output_filename': output_filename}
        constructor_kwargs['annotation_files'] = [p for p in config.annotation_files]
        constructor_kwargs['image_dir_names'] = [p for p in config.img_dir_names]

        assert len(constructor_kwargs['image_dir_names']) == \
            len(constructor_kwargs['annotation_files'])
        assert len(constructor_kwargs['image_dir_names']) == \
            config.num_partitions

        # If only one value is given to num_shards, same num_shards
        # will be applied to each partition
        if len(config.num_shards) == 1:
            constructor_kwargs['num_shards'] = list(config.num_shards) * config.num_partitions
        elif len(config.num_shards) == config.num_partitions:
            constructor_kwargs['num_shards'] = config.num_shards
        else:
            raise ValueError("Number of Shards {} do not match the size of number of partitions "
                             "{}.".format(len(config.num_shards), config.num_partitions))

        # Create a SampleModifier
        sample_modifier_config = dataset_export_config.sample_modifier_config
        sample_modifier = build_sample_modifier(
            sample_modifier_config=sample_modifier_config,
            validation_fold=validation_fold,
            num_folds=config.num_partitions)

        constructor_kwargs['sample_modifier'] = sample_modifier

        logger.info("Instantiating a coco converter")
        status_logging.get_status_logger().write(
            message="Instantiating a coco converter",
            status_level=status_logging.Status.STARTED)
        converter = COCOConverter(**constructor_kwargs)

        return converter
    raise NotImplementedError("Only supports KITTI or COCO")
