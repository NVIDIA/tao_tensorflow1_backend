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

"""TLT LPRNet data sequence."""

import logging
import math
import os
import random
import cv2
import numpy as np

from tensorflow.compat.v1.keras.utils import Sequence
from nvidia_tao_tf1.cv.lprnet.utils.img_utils import preprocess

logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level='INFO')
logger = logging.getLogger(__name__)
DEFAULT_STRIDE = 4


class LPRNetDataGenerator(Sequence):
    """Data generator for license plate dataset."""

    def __init__(self, experiment_spec, is_training=True, shuffle=True, time_step=None):
        """initialize data generator."""

        self.image_paths = []
        self.label_paths = []
        if is_training:
            data_sources = experiment_spec.dataset_config.data_sources
            self.batch_size = experiment_spec.training_config.batch_size_per_gpu
        else:
            data_sources = experiment_spec.dataset_config.validation_data_sources
            self.batch_size = experiment_spec.eval_config.batch_size

        for data_source in data_sources:
            self._add_source(data_source)

        self.data_inds = np.arange(len(self.image_paths))

        self.is_training = is_training
        self.n_samples = len(self.image_paths)
        self.output_width = experiment_spec.augmentation_config.output_width
        self.output_height = experiment_spec.augmentation_config.output_height
        self.output_channel = experiment_spec.augmentation_config.output_channel
        self.keep_original_prob = experiment_spec.augmentation_config.keep_original_prob
        self.max_rotate_degree = experiment_spec.augmentation_config.max_rotate_degree
        self.rotate_prob = experiment_spec.augmentation_config.rotate_prob
        self.gaussian_kernel_size = list(experiment_spec.augmentation_config.gaussian_kernel_size)
        self.blur_prob = experiment_spec.augmentation_config.blur_prob
        self.reverse_color_prob = experiment_spec.augmentation_config.reverse_color_prob

        # Load the characters list:
        characters_list_file = experiment_spec.dataset_config.characters_list_file
        with open(characters_list_file, "r") as f:
            temp_list = f.readlines()
        classes = [i.strip() for i in temp_list]
        self.class_dict = {classes[index]: index for index in range(len(classes))}
        self.classes = classes

        self.time_step = time_step
        self.max_label_length = experiment_spec.lpr_config.max_label_length
        suggest_width = (self.max_label_length * 2 + 1) * DEFAULT_STRIDE
        if self.output_width < suggest_width:
            logger.info("To avoid NaN loss, " +
                        "please set the output_width >= {}. ".format(suggest_width) +
                        "And then restart the training.")
            exit()
        self.shuffle = shuffle
        self.on_epoch_end()

    def _add_source(self, data_source):
        """Add image/label paths."""

        img_files = os.listdir(data_source.image_directory_path)
        label_files = set(os.listdir(data_source.label_directory_path))
        supported_img_format = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

        for img_file in img_files:
            file_name, img_ext = os.path.splitext(img_file)
            if img_ext in supported_img_format and file_name + ".txt" in label_files:
                self.image_paths.append(os.path.join(data_source.image_directory_path,
                                                     img_file))
                self.label_paths.append(os.path.join(data_source.label_directory_path,
                                                     file_name+".txt"))

    def __len__(self):
        """return number of batches in dataset."""

        return int(math.ceil(len(self.image_paths)/self.batch_size))

    def __getitem__(self, idx):
        """preprare processed data for training and evaluation."""

        begin_id = idx*self.batch_size
        end_id = min(len(self.data_inds), (idx+1)*self.batch_size)
        batch_x_file_list = [self.image_paths[i] for i in
                             self.data_inds[begin_id:end_id]]
        batch_y_file_list = [self.label_paths[i] for i in
                             self.data_inds[begin_id:end_id]]

        read_flag = cv2.IMREAD_COLOR
        if self.output_channel == 1:
            read_flag = cv2.IMREAD_GRAYSCALE

        batch_x = [np.array(cv2.imread(file_name, read_flag), dtype=np.float32)
                   for file_name in batch_x_file_list]

        # preprocess the image batch
        batch_x = preprocess(batch_x,
                             is_training=self.is_training,
                             output_width=self.output_width,
                             output_height=self.output_height,
                             output_channel=self.output_channel,
                             keep_original_prob=self.keep_original_prob,
                             max_rotate_degree=self.max_rotate_degree,
                             rotate_prob=self.rotate_prob,
                             gaussian_kernel_size=self.gaussian_kernel_size,
                             blur_prob=self.blur_prob,
                             reverse_color_prob=self.reverse_color_prob)

        # preprare sequence labels
        if self.is_training:
            batch_y = []
            batch_input_length = []
            batch_label_length = []
            for file_name in batch_y_file_list:
                with open(file_name, "r") as f:
                    label_line = f.readline().strip()
                label = np.array([self.class_dict[char] for char in label_line])
                batch_input_length.append(self.time_step)
                batch_label_length.append(len(label))
                batch_y.append(np.pad(label, (0, self.max_label_length - len(label))))

            batch_y = np.array(batch_y)
            batch_input_length = np.array(batch_input_length)
            batch_input_length = batch_input_length[:, np.newaxis]
            batch_label_length = np.array(batch_label_length)
            batch_label_length = batch_label_length[:, np.newaxis]
            batch_final_label = np.concatenate((batch_y, batch_input_length, batch_label_length),
                                               axis=-1)
        else:
            batch_y = []
            for file_name in batch_y_file_list:
                with open(file_name, "r") as f:
                    label = f.readline().strip()
                batch_y.append(label)
            batch_final_label = batch_y

        return batch_x, batch_final_label

    def on_epoch_end(self):
        """shuffle the dataset on epoch end."""

        if self.shuffle is True:
            random.shuffle(self.data_inds)
