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

"""Dataset class encapsulates the data loading."""
import logging
import os
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import tensorflow as tf
from nvidia_tao_tf1.core.utils.path_utils import expand_path

logger = logging.getLogger(__name__)

UNKNOWN_CLASS = '-1'


class Dataset():
    """Load, separate and prepare the data for training, prediction and evaluation."""

    def __init__(self, batch_size, fold=1, augment=False, gpu_id=0,
                 num_gpus=1, params=None, phase="train", target_classes=None,
                 buffer_size=None, data_options=True, filter_data=False):
        """Instantiate the dataloader.

        Args:
            data_dir (str): List of tuples (tfrecord_file_pattern,
                image_directory_path) to use for training.
            batch_size (int): Batch size to be used for training
            fold (int): The fold to be used for benchmarking.
            augment (bool): Holds the parameters for augmentation and
                            preprocessing.
            gpu_id (int): The GPU id to be used for training.
            params (dic): Dictionary containing the parameters for the run_config
                          of the estimator
            num_gpus (int): Number of GPU's to be used for training
            phase(str): train/ infer/ val
            target_classes(list): list of target class objects
            buffer_size(int): Buffer size to use the no. of samples per step
            data_options(bool): Data options to vectorize the data loading
            filter_data(bool): Filter images/ masks that are not present

        """

        self._batch_size = batch_size
        self._augment = augment
        self.filter_data = filter_data
        self.image_global_index = 0
        self._seed = params.seed
        self.resize_padding = params.resize_padding
        self.resize_method = params.resize_method
        self.backbone = params.experiment_spec.model_config.arch
        self.model_input_height = params.experiment_spec.model_config.model_input_height
        self.model_input_width = params.experiment_spec.model_config.model_input_width
        self.model_input_channels = params.experiment_spec.model_config.model_input_channels
        self.model_arch = params.experiment_spec.model_config.arch
        self.model_output_height, self.model_output_width = self.get_output_dimensions()
        self.input_image_type = params.experiment_spec.dataset_config.input_image_type
        # Setting the default input image type to color
        if not self.input_image_type:
            self.input_image_type = "color"
        self._num_gpus = num_gpus
        self._gpu_id = gpu_id
        self.phase = phase
        self.supported_img_formats = ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]
        self.dataset = params.experiment_spec.dataset_config.dataset
        self.train_data_sources = \
            params.experiment_spec.dataset_config.train_data_sources.data_source
        self.val_data_sources = params.experiment_spec.dataset_config.val_data_sources.data_source
        self.test_data_sources = params.experiment_spec.dataset_config.test_data_sources.data_source

        self.train_images_path = params.experiment_spec.dataset_config.train_images_path
        self.train_masks_path = params.experiment_spec.dataset_config.train_masks_path
        self.val_images_path = params.experiment_spec.dataset_config.val_images_path
        self.val_masks_path = params.experiment_spec.dataset_config.val_masks_path
        self.test_images_path = params.experiment_spec.dataset_config.test_images_path
        self.test_image_names = []
        self.test_mask_names = []
        self.image_names_list, self.masks_names_list = self.get_images_masks_lists()
        self.buffer_size = buffer_size if buffer_size else len(self.image_names_list)
        assert(self.buffer_size <= len(self.image_names_list)), \
            "Buffer size should not be more than total dataset size."
        self.decode_fn_img, self.decode_fn_label = self.validate_extension()
        self.img_input_height, self.img_input_width, self.img_input_channels = \
            self.get_input_shape()
        self.target_classes = target_classes
        self.lookup_table = None
        self.label_train_dic = self.get_label_train_dic()
        self.num_classes = params.num_classes
        self.use_amp = params.use_amp
        self.preprocess = params.experiment_spec.dataset_config.preprocess if \
            params.experiment_spec.dataset_config.preprocess else "min_max_-1_1"
        self.augmentation_params = params.experiment_spec.dataset_config.augmentation_config
        self.data_options = data_options

        print("\nPhase %s: Total %d files." % (self.phase, len(self.image_names_list)))

    def validate_extension(self):
        """Function to validate the image/ label extension and computing extension."""

        assert(len(self.image_names_list) > 0), \
            "Please check images path. The input image list is empty."
        img_ext = self.image_names_list[0].split(".")[-1]
        assert(img_ext in self.supported_img_formats), "Image Extension is not supported."
        decode_fn_img = self.get_decode_fn(img_ext)
        decode_fn_label = None
        if self.masks_names_list[0]:
            label_ext = self.masks_names_list[0].split(".")[-1]
            assert(label_ext in self.supported_img_formats), "Label Extension is not supported."
            decode_fn_label = self.get_decode_fn(label_ext)

        return decode_fn_img, decode_fn_label

    def get_input_shape(self):
        """Function to get input shape."""

        img_name = self.image_names_list[0]
        img_arr = np.array(Image.open(img_name))
        input_img_shape = img_arr.shape
        img_input_height = input_img_shape[0]
        img_input_width = input_img_shape[1]
        if len(img_arr.shape) == 2:
            img_input_channels = 1
        else:
            img_input_channels = input_img_shape[2]
        if self.model_arch == "vanilla_unet":
            if(self.model_input_height != 572 and self.model_input_width != 572):
                logging.info("The input height and width for vanilla unet is defaulted to \
                572")
                self.model_input_height = 572
                self.model_input_width = 572
        else:
            try:
                assert(self.model_input_height % 16 == 0 and self.model_input_width % 16 == 0)
            except Exception:
                raise ValueError("The input height and width for Resnet and VGG backbones \
                                  should be multiple of 16")

        return img_input_height, img_input_width, img_input_channels

    def extract_image_mask_names_from_datasource(self, data_sources):
        """Function to get the image and mask paths from multiple data sources."""

        images_list = []
        masks_list = []

        for data_source in data_sources:
            image_path = data_source.image_path if data_source.image_path else None
            mask_path = data_source.masks_path if data_source.masks_path else None
            images_list.append(image_path)
            # The mask list is None when the masks are not provided
            masks_list.append(mask_path)

        return images_list, masks_list

    def read_data_image_dir(self, images_dir, masks_dir):
        """Function to get the image and mask paths."""

        image_names_list = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
        if masks_dir:
            masks_names_list = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir)]
        else:
            # It is inference
            masks_names_list = [None for _ in image_names_list]

        return image_names_list, masks_names_list

    def get_images_masks_lists(self):
        """Function to get the image and mask get_images_masks_paths."""

        if self.phase == "train":
            data_sources = self.train_data_sources
            if not data_sources:
                # Set the images/ masks path
                images_dir = self.train_images_path
                masks_dir = self.train_masks_path
        elif self.phase == "val":
            data_sources = self.val_data_sources
            if not data_sources:
                # Set the images/ masks path
                images_dir = self.val_images_path
                masks_dir = self.val_masks_path
        elif self.phase == "test":
            data_sources = self.test_data_sources
            if not data_sources:
                # Set the images/ masks path
                images_dir = self.test_images_path
                # Masks are not required for test
                masks_dir = None

        if data_sources:
            images_list, masks_list = self.extract_image_mask_names_from_datasource(
                data_sources)
            image_names_list, masks_names_list = self.read_data_list_files(images_list,
                                                                           masks_list)
        elif images_dir:
            image_names_list, masks_names_list = self.read_data_image_dir(images_dir,
                                                                          masks_dir)

        return shuffle(image_names_list, masks_names_list)

    def get_label_train_dic(self):
        """"Function to get mapping between class and train id's."""

        label_train_dic = {}
        for target in self.target_classes:
            label_train_dic[target.label_id] = target.train_id

        return label_train_dic

    def read_data_list_files(self, images_list, masks_list):
        """" Reads text files specifying the list of x and y items."""

        x_set_filt = []
        y_set_filt = []

        for imgs, lbls in zip(images_list, masks_list):

            print("Reading Imgs : {}, Reading Lbls : {}".format(imgs, lbls))
            if self.phase == "train":
                assert os.path.isfile(imgs) and imgs.endswith(".txt"), (
                    f"Image file doesn't exist at {imgs}"
                )
                assert os.path.isfile(lbls) and lbls.endswith(".txt"), (
                    f"Label file doesn't exist at {lbls}"
                )

            # Both are valid text files. So read from them.
            with open(imgs) as f:
                x_set = f.readlines()

            # Remove whitespace characters like `\n` at the end of each line
            if lbls:
                with open(lbls) as f:
                    y_set = f.readlines()
                    for f_im, f_label in zip(x_set, y_set):
                        # Ensuring all image files are present
                        f_im = f_im.strip()
                        f_label = f_label.strip()
                        if self.filter_data:
                            if os.path.isfile(expand_path(f_im)) and os.path.isfile(expand_path(f_label)):
                                x_set_filt.append(f_im)
                                y_set_filt.append(f_label)
                        else:
                            x_set_filt.append(f_im)
                            y_set_filt.append(f_label)
            else:
                # During inference we do not filter
                y_set_filt += [None for _ in x_set]
                x_set_filt += [x.strip() for x in x_set]

        return x_set_filt, y_set_filt

    def augment(self, x, y, x_orig=None):
        """"Map function to augment input x and y."""

        if self._augment:
            # Default values
            # If the user provides augment alone and not the aug config.
            hflip_probability = 0.5
            vflip_probability = 0.5
            crop_and_resize_prob = 0.5
            crop_and_resize_ratio = 0.1
            delta = 0.2
            if self.augmentation_params:
                # If spatial augmentation params are provided
                if self.augmentation_params.spatial_augmentation:
                    hflip_probability = \
                        self.augmentation_params.spatial_augmentation.hflip_probability
                    vflip_probability = \
                        self.augmentation_params.spatial_augmentation.vflip_probability
                    crop_and_resize_prob = \
                        self.augmentation_params.spatial_augmentation.crop_and_resize_prob
                    crop_and_resize_ratio = \
                        self.augmentation_params.spatial_augmentation.crop_and_resize_ratio

                    # Reverting to default values if not present
                    if not hflip_probability:
                        hflip_probability = 0.5
                    if not vflip_probability:
                        vflip_probability = 0.5
                    if not crop_and_resize_prob:
                        crop_and_resize_prob = 0.5

                if self.augmentation_params.brightness_augmentation:
                    delta = self.augmentation_params.brightness_augmentation.delta

                    if not delta:
                        delta = 0.2

            # Horizontal flip
            h_flip = tf.random_uniform([]) < hflip_probability
            x = tf.cond(h_flip, lambda: tf.image.flip_left_right(x), lambda: x)
            y = tf.cond(h_flip, lambda: tf.image.flip_left_right(y), lambda: y)
            # Vertical flip
            v_flip = tf.random_uniform([]) < vflip_probability
            x = tf.cond(v_flip, lambda: tf.image.flip_up_down(x), lambda: x)
            y = tf.cond(v_flip, lambda: tf.image.flip_up_down(y), lambda: y)
            # Prepare for batched transforms
            x = tf.expand_dims(x, 0)
            y = tf.expand_dims(y, 0)

            # Random crop and resize
            crop_and_resize = tf.random_uniform([]) < crop_and_resize_prob
            left = tf.random_uniform([]) * crop_and_resize_ratio
            right = 1 - tf.random_uniform([]) * crop_and_resize_ratio
            top = tf.random_uniform([]) * crop_and_resize_ratio
            bottom = 1 - tf.random_uniform([]) * crop_and_resize_ratio
            x = tf.cond(
                crop_and_resize,
                lambda: tf.image.crop_and_resize(x, [[top, left, bottom, right]], [0],
                                                 (self.model_input_height,
                                                 self.model_input_width)), lambda: x)
            y = tf.cond(
                crop_and_resize,
                lambda: tf.image.crop_and_resize(y, [[top, left, bottom, right]], [0],
                                                 (self.model_input_height,
                                                 self.model_input_width),
                                                 method="nearest"), lambda: y)

            # Adjust brightness and keep values in range
            x = tf.image.random_brightness(x, max_delta=delta)
            if self.preprocess == "min_max_-1_1":
                x = tf.clip_by_value(x, clip_value_min=-1, clip_value_max=1)
            elif self.preprocess == "min_max_0_1":
                x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=1)
            x = tf.squeeze(x, 0)
            y = tf.squeeze(y, 0)

            if x_orig is not None:
                # Apply respective transformations except normalization to input image
                x_orig = tf.cond(h_flip, lambda: tf.image.flip_left_right(x_orig), lambda: x_orig)
                x_orig = tf.cond(v_flip, lambda: tf.image.flip_up_down(x_orig), lambda: x_orig)
                x_orig = tf.expand_dims(x_orig, 0)
                x_orig = tf.cond(
                    crop_and_resize,
                    lambda: tf.image.crop_and_resize(x_orig, [[top, left, bottom, right]], [0],
                                                     (self.model_input_height,
                                                     self.model_input_width)), lambda: x_orig)
                x_orig = tf.squeeze(x_orig, 0)

        return x, y, x_orig

    def resize_vanilla(self, x, y, x_orig=None):
        """Function to resize the output to mode output size for Vanilla Unet."""

        if y is not None:
            if self.model_arch == "vanilla_unet":
                y = tf.image.resize_image_with_crop_or_pad(
                    y, target_width=self.model_output_width,
                    target_height=self.model_output_height)
            return x, y, x_orig
        return x

    @property
    def train_size(self):
        """Function to get the size of the training set."""
        return len(self.image_names_list)

    @property
    def eval_size(self):
        """Function that returns the size the eval dataset."""
        return len(self.image_names_list)

    @property
    def test_size(self):
        """Function that returns the size the test dataset."""
        return len(self.image_names_list)

    def get_output_dimensions(self):
        """Function to return model input heights and width."""
        if self.model_arch == "vanilla_unet":
            return 388, 388
        return self.model_input_height, self.model_input_width

    def get_test_image_names(self):
        """Function that returns the test image names."""
        return self.image_names_list

    def get_test_mask_names(self):
        """Function that returns the test image names."""
        return self.test_mask_names

    @staticmethod
    def get_decode_fn(ext):
        """Function to assign the decode function."""
        if ext.lower() in ["jpg", "jpeg"]:
            decode_fn = tf.io.decode_jpeg
        else:
            # EXT should be png
            decode_fn = tf.io.decode_png
        return decode_fn

    def read_image_and_label_tensors(self, img_path, label=None):
        """Function to read image tensor."""

        self.test_image_names.append(img_path)
        x_str = tf.io.read_file(img_path)
        x = self.decode_fn_img(contents=x_str, channels=self.model_input_channels)
        x_orig = x
        if self.input_image_type == "grayscale":
            # Grayscale needs to be normalized before resizing
            x = tf.cast(x, dtype="float32")
            x = tf.divide(x, 127.5) - 1
        if label is not None:
            y_str = tf.io.read_file(label)
            y = self.decode_fn_label(contents=y_str, channels=1)
            if self.input_image_type == "grayscale":
                y = tf.divide(y, 255)
                y = tf.cast(y, dtype="float32")
            return x, y, x_orig
        return x

    def apply_label_mapping_tf(self, x, y=None, x_orig=None):
        """Map Function to apply class mapping."""

        if self.input_image_type == "grayscale":
            return x, y, x_orig
        if y is not None:

            y = tf.cast(y, dtype=tf.int64)
            if self.lookup_table is None:
                keys = list(self.label_train_dic.keys())
                values = list(self.label_train_dic.values())
                keys = tf.cast(tf.constant(keys), dtype=tf.int64)
                values = tf.cast(tf.constant(values), dtype=tf.int64)
                self.lookup_table = tf.contrib.lookup.HashTable(
                    tf.contrib.lookup.KeyValueTensorInitializer(keys, values), 0)
            y = self.lookup_table.lookup(y)
        return x, y, x_orig

    def rgb_to_bgr_tf(self, x, y=None, x_orig=None):
        """Map Function to convert image to channel first."""

        if self.input_image_type != "grayscale":
            x = tf.reverse(x, axis=[-1])
        if y is not None:
            return x, y, x_orig
        return x

    def cast_img_lbl_dtype_tf(self, img, label=None, x_orig=None):
        """Map Function to cast labels to float32."""

        img_cast = tf.cast(img, dtype="float32")

        if label is not None:
            label_cast = tf.cast(label, dtype="float32")

            return img_cast, label_cast, x_orig
        return img_cast

    def resize_image_helper(self, img):
        """Helper function to resize the input image."""

        resize_methods = {'BILINEAR': tf.image.ResizeMethod.BILINEAR,
                          'NEAREST_NEIGHBOR': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                          'BICUBIC': tf.image.ResizeMethod.BICUBIC,
                          'AREA': tf.image.ResizeMethod.AREA}
        if self.model_arch == "vanilla_unet":
            img = tf.image.resize_images(img, (self.model_output_height,
                                               self.model_output_width))
            x = tf.image.resize_image_with_crop_or_pad(img, self.model_input_height,
                                                       self.model_input_width)
        else:
            if self.resize_padding:
                x = tf.image.resize_image_with_pad(img,
                                                   target_height=self.model_input_height,
                                                   target_width=self.model_input_width,
                                                   method=resize_methods[self.resize_method])
            else:
                x = tf.image.resize_images(img, (self.model_input_height,
                                                 self.model_input_width),
                                           method=resize_methods[self.resize_method])

        return x

    def resize_image_and_label_tf(self, img, label=None, x_orig=None):
        """Map Function to preprocess and resize images/ labels."""

        x = self.resize_image_helper(img)
        if x_orig is not None:
            x_orig = self.resize_image_helper(x_orig)
        if label is not None:
            if self.model_arch == "vanilla_unet":
                y = tf.image.resize_images(label, (self.model_output_height,
                                                   self.model_output_width))
                y = tf.image.resize_image_with_crop_or_pad(y, self.model_input_height,
                                                           self.model_input_width)
            else:
                # Labels should be always nearest neighbour, as they are integers.
                if self.resize_padding:
                    y = tf.image.resize_image_with_pad(
                        label, target_height=self.model_output_height,
                        target_width=self.model_output_width,
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                else:
                    y = tf.image.resize_images(
                        label, (self.model_output_height, self.model_output_width),
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            return x, y, x_orig
        return x

    def normalize_img_tf(self, img_tensor, y=None, x_orig=None):
        """Map Function to normalize input image."""

        if self.input_image_type != "grayscale":
            # img_tensor = tf.divide(img_tensor, 127.5) - 1

            if self.preprocess == "div_by_255":
                # A way to normalize an image tensor by dividing them by 255.
                # This assumes images with max pixel value of
                # 255. It gives normalized image with pixel values in range of >=0 to <=1.
                img_tensor /= 255.0

            elif self.preprocess == "min_max_0_1":
                img_tensor /= 255.0

            elif self.preprocess == "min_max_-1_1":
                img_tensor = tf.divide(img_tensor, 127.5) - 1
        if y is not None:
            return img_tensor, y, x_orig
        return img_tensor

    def categorize_image_tf(self, img_tensor, number_of_classes, img_width,
                            img_height, data_format):
        """
        Label Pre-processor.

        Converts a 1 channel image tensor containing class values from 0 to N-1.
        where N is total number of classes using TensorFlow.
        :param img_tensor: Input image tensor.
        :param number_of_classes: Total number of classes.
        :param img_width: The width of input the image.
        :param img_height: The height of input image.
        :param data_format: Either channels_last or channels_first way of storing data.
        :return: Categorized image tensor with as many channels as there were number of classes.
        Each channel would have a 1 if the class was present otherwise a 0.
        """
        # Here we assume the image_tensor to have an tf.uint8 dtype. No asserts have been put.
        # Flatten the image out first.
        if self.input_image_type == "grayscale":
            cond = tf.less(img_tensor, 0.5 * tf.ones(tf.shape(img_tensor)))
            img_tensor = tf.where(cond, tf.zeros(tf.shape(img_tensor)),
                                  tf.ones(tf.shape(img_tensor)))
            labels = tf.cast(img_tensor, tf.int32)
            if self.num_classes != 1:
                # We need not do one-hot vectorization if num classes > 1
                labels = tf.one_hot(labels, self.num_classes,
                                    axis=0)
            labels = tf.cast(labels, tf.float32)
            labels = tf.reshape(labels,
                                [img_height, img_width, number_of_classes] if
                                data_format == "channels_last" else
                                [number_of_classes, img_width, img_height])
            labels = tf.cast(labels, dtype="float32")
            return labels

        img_flatten = tf.reshape(img_tensor, [-1])
        img_flatten_uint8 = tf.cast(img_flatten, tf.uint8)
        # Give it to one hot.
        img_cat = img_flatten_uint8
        if self.num_classes != 1:
            img_cat = tf.one_hot(img_flatten_uint8, depth=number_of_classes, axis=-1
                                 if data_format == "channels_last" else 0,
                                 dtype=img_flatten_uint8.dtype)
        im_cat_dtype_cast = tf.cast(img_cat, img_tensor.dtype)
        # Un-flatten it back.
        img_cat_unflatten = tf.reshape(im_cat_dtype_cast,
                                       [img_height, img_width, number_of_classes]
                                       if data_format == "channels_last" else
                                       [number_of_classes, img_height, img_width])

        img_cat_unflatten = tf.cast(img_cat_unflatten, dtype="float32")

        return img_cat_unflatten

    def dictionarize_labels_eval(self, x, y=None, x_orig=None):
        """Map Function to return labels for evaluation."""

        x_dic = {"x_orig": x_orig, "features": x, "labels": y}
        return x_dic, y

    def transpose_to_nchw(self, x, y=None, x_orig=None):
        """Map function image to first channel."""

        x = tf.transpose(x, perm=[2, 0, 1])  # Brings channel dimension to first. from HWC to CHW.

        if y is not None:
            y = tf.transpose(y, perm=[2, 0, 1])
            return x, y, x_orig
        return x

    def prednn_categorize_label(self, img, label_img, x_orig=None):
        """Map function to convert labels to integer labels."""

        if label_img is not None:
            return img, self.categorize_image_tf(label_img, number_of_classes=self.num_classes,
                                                 img_width=self.model_output_width,
                                                 img_height=self.model_output_height,
                                                 data_format="channels_first"), x_orig
        return img

    def input_fn(self, drop_remainder=False):
        """Function to input images and labels."""

        return self.input_fn_aigs_tf()

    def input_fn_aigs_tf(self):
        """Input function for training."""

        dataset = tf.data.Dataset.from_tensor_slices((self.image_names_list, self.masks_names_list))
        if self.phase == "train":
            dataset = dataset.shuffle(buffer_size=self.buffer_size,
                                      seed=self._seed,
                                      reshuffle_each_iteration=True)
        dataset = dataset.shard(self._num_gpus, self._gpu_id)
        dataset = dataset.map(self.read_image_and_label_tensors,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x, y=None,
                              x_orig=None: self.apply_label_mapping_tf(x, y, x_orig),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.rgb_to_bgr_tf,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.cast_img_lbl_dtype_tf,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.resize_image_and_label_tf,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x, y=None, x_orig=None: self.normalize_img_tf(x, y, x_orig),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.phase == "train":
            dataset = dataset.map(lambda x, y=None, x_orig=None: self.augment(x, y, x_orig),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x, y=None, x_orig=None: self.resize_vanilla(x, y, x_orig),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.transpose_to_nchw,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x, y=None,
                              x_orig=None: self.prednn_categorize_label(x, y, x_orig),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())

        if self.phase == "train":
            dataset = dataset.repeat()
        dataset = dataset.map(lambda x, y=None,
                              x_orig=None: self.dictionarize_labels_eval(x, y, x_orig),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        if self.data_options:
            dataset = dataset.with_options(self._data_options)

        return dataset

    def eval_fn(self, count=1):
        """Input function for Evaluation."""

        return self.input_fn()

    def test_fn(self, count=1):
        """Input function for Testing."""

        dataset = tf.data.Dataset.from_tensor_slices((self.image_names_list))
        dataset = dataset.shard(self._num_gpus, self._gpu_id)
        dataset = dataset.map(self.read_image_and_label_tensors,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.rgb_to_bgr_tf,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.cast_img_lbl_dtype_tf,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.resize_image_and_label_tf,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x, y=None, x_orig=None: self.normalize_img_tf(x),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x, y=None, x_orig=None: self.resize_vanilla(x, y),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.transpose_to_nchw,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    @property
    def _data_options(self):
        """Constructs tf.data.Options for this dataset."""
        data_options = tf.data.Options()
        data_options.experimental_optimization.parallel_batch = True
        data_options.experimental_slack = True
        data_options.experimental_optimization.map_parallelization = True
        map_vectorization_options = tf.data.experimental.MapVectorizationOptions()
        map_vectorization_options.enabled = True
        map_vectorization_options.use_choose_fastest = True
        data_options.experimental_optimization.map_vectorization = map_vectorization_options

        return data_options
