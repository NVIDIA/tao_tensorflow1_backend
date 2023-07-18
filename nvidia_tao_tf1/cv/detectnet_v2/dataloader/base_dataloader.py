# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""
Dataloader base class defining the interface to data loading.

All data loader classes are expected to conform to the interface defined here.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import six


class BaseDataloader(six.with_metaclass(ABCMeta, object)):
    """Dataloader base class defining the interface to data loading."""

    @abstractmethod
    def __init__(self,
                 training_data_source_list,
                 augmentation_config=None,
                 validation_fold=None,
                 validation_data_source_list=None):
        """Instantiate the dataloader.

        Args:
            training_data_source_list (list): List of tuples (tfrecord_file_pattern,
                image_directory_path) to use for training.
            augmentation_config (nvidia_tao_tf1.cv.detectnet_v2.dataloader.augmentation_config.
                AugmentationConfig): Holds the parameters for augmentation and preprocessing.
            validation_fold (int): Indicates which fold from the training data to use as validation.
                Can be None.
            validation_data_source_list (list): List of tuples (tfrecord_file_pattern,
                image_directory_path) to use for validation. Can be None.
        """
        pass

    @abstractmethod
    def get_dataset_tensors(self, batch_size, training, enable_augmentation, repeat=True):
        """Interface for getting tensors for training and validation.

        Args:
            batch_size (int): Minibatch size.
            training (bool): Get samples from the training (True) or validation (False) set.
            enable_augmentation (bool): Whether to augment input images and labels.
            repeat (bool): Whether the dataset can be looped over multiple times or only once.
        """
        pass

    @abstractmethod
    def get_data_tensor_shape(self):
        """Interface for querying data tensor shape.

        Returns:
            Data tensor shape as a tuple without the batch dimension.
        """
        pass

    @abstractmethod
    def get_num_samples(self, training):
        """Get number of dataset samples.

        Args:
            training (bool): Get number of samples in the training (true) or
                validation (false) set.

        Returns:
            Number of samples in the chosen set.
        """
        pass
