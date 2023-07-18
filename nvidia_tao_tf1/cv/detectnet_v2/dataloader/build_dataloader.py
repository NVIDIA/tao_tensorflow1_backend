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

"""Build a Dataloader from proto."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

from nvidia_tao_tf1.cv.detectnet_v2.dataloader.augmentation.build_augmentation_config \
    import build_augmentation_config
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.data_source_config import DataSourceConfig
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.default_dataloader import DefaultDataloader
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.drivenet_dataloader import DriveNetDataloader
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.legacy_dataloader import LegacyDataloader
from nvidia_tao_tf1.cv.detectnet_v2.dataloader.utilities import get_absolute_data_path
from nvidia_tao_tf1.cv.detectnet_v2.proto.dataset_config_pb2 import DatasetConfig

FOLD_STRING = "fold-{:03d}-of-"

DATALOADER = {
    0: DriveNetDataloader,
    1: LegacyDataloader,
    2: DefaultDataloader
}

SAMPLING_MODE = {
    0: "user_defined",
    1: "proportional",
    2: "uniform"
}


def select_dataset_proto(experiment_spec):
    """Select the dataset proto depending on type defined in the spec.

    Args:
        experiment_spec: nvidia_tao_tf1.cv.detectnet_v2.proto.experiment proto message.
    Returns:
        A dataset_proto depending on type defined in the spec.
    """
    if hasattr(experiment_spec, 'dataset_config'):
        return experiment_spec.dataset_config
    raise ValueError("Invalid experiment spec file. Dataset config not mentioned.")


def _pattern_to_files(pattern):
    """Convert a file pattern to a list of absolute file paths.

    Args:
        pattern (str): File pattern.

    Returns:
        A list of absolute file paths.
    """
    abs_pattern = get_absolute_data_path(pattern)

    # Convert pattern to list of files.
    files = glob.glob(abs_pattern)

    assert len(files) > 0, \
        "No files match pattern {}.".format(abs_pattern)

    return files


def build_data_source_lists(dataset_proto):
    """Build training and validation data source lists from proto.

    Args:
        dataset_proto (nvidia_tao_tf1.cv.detectnet_v2.proto.dataset_config proto message)

    Returns:
        training_data_source_list (list): List of tuples (tfrecord_file_pattern,
            image_directory_path) to use for training.
        validation_data_source_list (list): List of tuples (tfrecord_file_pattern,
            image_directory_path) to use for validation. Can be None.
        validation_fold (int): Validation fold number (0-based). Indicates which fold from the
            training data to use as validation. Can be None.
    """
    # Determine how we are getting validation data sources.
    dataset_split_type = dataset_proto.WhichOneof('dataset_split_type')

    training_data_source_list = []
    validation_data_source_list = []
    validation_fold = None

    for data_source_proto in dataset_proto.data_sources:
        source_weight = data_source_proto.source_weight
        images_path = get_absolute_data_path(
            str(data_source_proto.image_directory_path)
        )
        tfrecords_path = str(data_source_proto.tfrecords_path)
        tfrecords_files = _pattern_to_files(tfrecords_path)

        # Filter out files based on validation fold only if validation fold specified.
        if dataset_split_type == "validation_fold":
            # Defining the fold number for the glob pattern.
            fold_identifier = FOLD_STRING.format(dataset_proto.validation_fold)
            validation_fold = dataset_proto.validation_fold

            # Take all .tfrecords files except the one matching the validation fold.
            training_tfrecords_files = [filename for filename in tfrecords_files
                                        if fold_identifier not in filename]

            # Take only the file matching the validation fold.
            validation_tfrecords_files = [filename for filename in tfrecords_files
                                          if fold_identifier in filename]

            validation_data_source_list.append(DataSourceConfig(
                dataset_type='tfrecord',
                dataset_files=validation_tfrecords_files,
                images_path=images_path,
                export_format=None,
                split_db_path=None,
                split_tags=None,
                source_weight=source_weight))
        else:
            training_tfrecords_files = tfrecords_files

        training_data_source_list.append(DataSourceConfig(
            dataset_type='tfrecord',
            dataset_files=training_tfrecords_files,
            images_path=images_path,
            export_format=None,
            split_db_path=None,
            split_tags=None,
            source_weight=source_weight))

    # Get validation data sources, if available.
    if dataset_split_type == "validation_data_source":
        data_source_proto = dataset_proto.validation_data_source
        source_weight = data_source_proto.source_weight
        images_path = get_absolute_data_path(
            str(data_source_proto.image_directory_path)
        )
        tfrecords_path = str(data_source_proto.tfrecords_path)
        tfrecords_files = _pattern_to_files(tfrecords_path)
        validation_data_source_list.append(DataSourceConfig(
            dataset_type='tfrecord',
            dataset_files=tfrecords_files,
            images_path=images_path,
            export_format=None,
            split_db_path=None,
            split_tags=None,
            source_weight=source_weight))

    return training_data_source_list, validation_data_source_list, validation_fold


def build_legacy_data_source_lists(dataset_proto):
    """Build training and validation data source lists from proto.

    Args:
        dataset_proto (nvidia_tao_tf1.cv.detectnet_v2.proto.dataset_config proto message)

    Returns:
        training_data_source_list (list): List of tuples (tfrecord_file_pattern,
            image_directory_path) to use for training.
        validation_data_source_list (list): List of tuples (tfrecord_file_pattern,
            image_directory_path) to use for validation. Can be None.
        validation_fold (int): Validation fold number (0-based). Indicates which fold from the
            training data to use as validation. Can be None.
    """
    # Determine how we are getting validation data sources.
    dataset_split_type = dataset_proto.WhichOneof('dataset_split_type')

    def build_data_source_list(data_source_proto):
        """Build a list of data sources from a DataSource proto.

        Args:
            data_source_proto (nvidia_tao_tf1.cv.detectnet_v2.proto.DataSource).

        Returns:
            Tuple (tfrecord_file_pattern, image_directory_path).
        """
        return str(data_source_proto.tfrecords_path), str(data_source_proto.image_directory_path)

    training_data_source_list = map(
        build_data_source_list, dataset_proto.data_sources)

    # Get the validation data sources, depending on the scenario.
    if dataset_split_type == "validation_fold":
        validation_fold = dataset_proto.validation_fold
        validation_data_source_list = None
    elif dataset_split_type == "validation_data_source":
        validation_fold = None
        validation_data_source_list = [build_data_source_list(
            dataset_proto.validation_data_source)]
    else:
        validation_fold = None
        validation_data_source_list = None

    return training_data_source_list, validation_data_source_list, validation_fold


def build_dataloader(dataset_proto, augmentation_proto):
    """Build a Dataloader from a proto.

    Args:
        dataset_proto (nvidia_tao_tf1.cv.detectnet_v2.proto.dataset_config.DatasetConfig)
        augmentation_proto (nvidia_tao_tf1.cv.detectnet_v2.proto.augmentation_config.
            AugmentationConfig)

    Returns:
        dataloader (nvidia_tao_tf1.cv.detectnet_v2.dataloader.default_dataloader.DefaultDataloader).
    """
    if isinstance(dataset_proto, DatasetConfig):
        dataset_config = dataset_proto
    else:
        raise ValueError('Unsupported dataset_proto message.')

    # First, build the augmentation related parameters.
    augmentation_config = build_augmentation_config(augmentation_proto)

    # Now, get the class mapping.
    source_to_target_class_mapping = dict(dataset_config.target_class_mapping)

    # Image file encoding.
    image_file_encoding = dataset_config.image_extension

    # Get the data source lists.
    training_data_source_list, validation_data_source_list, validation_fold = \
        build_data_source_lists(dataset_config)

    try:
        assert isinstance(dataset_proto, DatasetConfig)
    except NotImplementedError:
        raise NotImplementedError("Unsupported dataloader called for.")

    dataloader_mode = dataset_config.dataloader_mode
    sampling_mode = dataset_config.sampling_mode
    dataloader_kwargs = dict(
        training_data_source_list=training_data_source_list,
        image_file_encoding=image_file_encoding,
        augmentation_config=augmentation_config
    )
    if dataloader_mode == 1:
        # Generate the legacy dataloader from TLT v1.0.
        training_data_source_list, validation_data_source_list, validation_fold = \
            build_legacy_data_source_lists(dataset_config)
        dataloader_kwargs = dict(
            training_data_source_list=training_data_source_list,
            target_class_mapping=source_to_target_class_mapping,
            image_file_encoding=image_file_encoding,
            augmentation_config=augmentation_config,
            validation_fold=validation_fold,
            validation_data_source_list=validation_data_source_list)
    elif dataloader_mode == 2:
        # Generate default dataloader with dict features.
        dataloader_kwargs.update(dict(
            validation_fold=validation_fold,
            target_class_mapping=source_to_target_class_mapping,
            validation_data_source_list=validation_data_source_list
        ))
    else:
        # Generate the multisource dataloader with sampling.
        dataloader_kwargs.update(dict(
            validation_data_source_list=validation_data_source_list,
            target_class_mapping=source_to_target_class_mapping,
            auto_resize=augmentation_proto.preprocessing.enable_auto_resize,
            sampling_mode=SAMPLING_MODE[sampling_mode]
        ))
    return DATALOADER[dataloader_mode](**dataloader_kwargs)
