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
"""Patch keras_preprocessing.image.utils.load_img() with cropping support."""
import random
import keras_preprocessing.image

from nvidia_tao_tf1.cv.makenet.utils.helper import color_augmentation

# padding size.
# We firstly resize to (target_width + CROP_PADDING, target_height + CROP_PADDING)
# , then crop to (target_width, target_height).
# for standard ImageNet size: 224x224 the ratio is 0.875(224 / (224 + 32)).
# but for EfficientNet B1-B7, larger resolution is used, hence this ratio
# is no longer 0.875
# ref:
# https://github.com/tensorflow/tpu/blob/r1.15/models/official/efficientnet/preprocessing.py#L110
CROP_PADDING = 32
COLOR_AUGMENTATION = False


def _set_color_augmentation(flag):
    global COLOR_AUGMENTATION  # pylint: disable=global-statement
    COLOR_AUGMENTATION = flag


def load_and_crop_img(path, grayscale=False, color_mode='rgb', target_size=None,
                      interpolation='nearest'):
    """Wraps keras_preprocessing.image.utils.load_img() and adds cropping.

    Cropping method enumarated in interpolation
    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation and crop methods used to resample and crop the image
            if the target size is different from that of the loaded image.
            Methods are delimited by ":" where first part is interpolation and second is crop
            e.g. "lanczos:random".
            Supported interpolation methods are "nearest", "bilinear", "bicubic", "lanczos",
            "box", "hamming" By default, "nearest" is used.
            Supported crop methods are "none", "center", "random".
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """

    # Decode interpolation string. Allowed Crop methods: none, center, random
    interpolation, crop = interpolation.split(":") \
        if ":" in interpolation else (interpolation, "none")

    if crop == "none":
        return keras_preprocessing.image.utils.load_img(
            path,
            grayscale=grayscale,
            color_mode=color_mode,
            target_size=target_size,
            interpolation=interpolation)

    # Load original size image using Keras
    img = keras_preprocessing.image.utils.load_img(
        path,
        grayscale=grayscale,
        color_mode=color_mode,
        target_size=None,
        interpolation=interpolation)

    # Crop fraction of total image
    target_width = target_size[1]
    target_height = target_size[0]

    if target_size is not None:
        if img.size != (target_width, target_height):

            if crop not in ["center", "random"]:
                raise ValueError('Invalid crop method %s specified.' % crop)

            if interpolation not in keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(
                            keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS.keys())))

            resample = keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS[interpolation]

            width, height = img.size

            if crop == 'random':
                # Resize keeping aspect ratio
                # result should be no smaller than the targer size, include crop fraction overhead
                crop_fraction = random.uniform(0.45, 1.0)
                target_size_before_crop = (
                    target_width / crop_fraction,
                    target_height / crop_fraction
                )
                ratio = max(
                    target_size_before_crop[0] / width,
                    target_size_before_crop[1] / height
                )
                target_size_before_crop_keep_ratio = int(width * ratio), int(height * ratio)
                img = img.resize(target_size_before_crop_keep_ratio, resample=resample)

            if crop == 'center':
                # Resize keeping aspect ratio
                # result should be no smaller than the targer size, include crop fraction overhead
                target_size_before_crop = (
                    target_width + CROP_PADDING,
                    target_height + CROP_PADDING
                )
                ratio = max(
                    target_size_before_crop[0] / width,
                    target_size_before_crop[1] / height
                )
                target_size_before_crop_keep_ratio = int(width * ratio), int(height * ratio)
                img = img.resize(target_size_before_crop_keep_ratio, resample=resample)

            width, height = img.size

            if crop == "center":
                left_corner = int(round(width/2)) - int(round(target_width/2))
                top_corner = int(round(height/2)) - int(round(target_height/2))
                return img.crop(
                    (left_corner,
                     top_corner,
                     left_corner + target_width,
                     top_corner + target_height))
            if crop == "random":
                # random crop
                left_shift = random.randint(0, int((width - target_width)))
                down_shift = random.randint(0, int((height - target_height)))
                img = img.crop(
                    (left_shift,
                     down_shift,
                     target_width + left_shift,
                     target_height + down_shift))
                # color augmentation
                if COLOR_AUGMENTATION and img.mode == "RGB":
                    return color_augmentation(img)
                return img
            raise ValueError("Crop mode not supported.")

    return img


# Monkey patch
keras_preprocessing.image.iterator.load_img = load_and_crop_img
