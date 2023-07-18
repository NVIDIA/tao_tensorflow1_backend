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
"""Images2DReference is used to represent references to image assets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types.images2d import (
    Images2D, LabelledImages2D
)

_CHANNELS = 3
_DEPTH = 8
_MIN_SIDE = 0
_MAX_SIDE = 0
_AUTO_RESIZE = False
# input height tensor for image height
H_TENSOR = None
H_TENSOR_VAL = None
# input width tensor for image width
W_TENSOR = None
W_TENSOR_VAL = None
# augmentations
AUGMENTATIONS = None
AUGMENTATIONS_VAL = None
# TODO(vkallioniemi): The loadig fuctionality here is mostly copied from modulus'
# LoadAndDecodeFrame


def set_image_channels(channel_count):
    """Set the image channel count."""
    global _CHANNELS  # noqa pylint: disable=W0603
    _CHANNELS = channel_count


def set_min_side(min_side):
    """set the minimal side of images."""
    global _MIN_SIDE  # noqa pylint: disable=W0603
    _MIN_SIDE = min_side


def set_max_side(max_side):
    """set the maximal side of images."""
    global _MAX_SIDE  # noqa pylint: disable=W0603
    _MAX_SIDE = max_side


def set_auto_resize(auto_resize):
    """set the automatic resize flag."""
    global _AUTO_RESIZE # noqa pylint: disable=W0603
    _AUTO_RESIZE = auto_resize


def set_h_tensor(h_tensor):
    """Set the height tensor."""
    global H_TENSOR # noqa pylint: disable=W0603
    assert (H_TENSOR is None) or (H_TENSOR is h_tensor), (
        "H_TENSOR can only be assigned once."
    )
    H_TENSOR = h_tensor


def set_h_tensor_val(h_tensor):
    """Set the height tensor."""
    global H_TENSOR_VAL # noqa pylint: disable=W0603
    assert (H_TENSOR_VAL is None) or (H_TENSOR_VAL is h_tensor), (
        "H_TENSOR_VAL can only be assigned once."
    )
    H_TENSOR_VAL = h_tensor


def set_w_tensor(w_tensor):
    """Set the width tensor."""
    global W_TENSOR # noqa pylint: disable=W0603
    assert (W_TENSOR is None) or (W_TENSOR is w_tensor), (
        "W_TENSOR can only be assigned once."
    )
    W_TENSOR = w_tensor


def set_w_tensor_val(w_tensor):
    """Set the width tensor."""
    global W_TENSOR_VAL # noqa pylint: disable=W0603
    assert (W_TENSOR_VAL is None) or (W_TENSOR_VAL is w_tensor), (
        "W_TENSOR_VAL can only be assigned once."
    )
    W_TENSOR_VAL = w_tensor


def set_image_depth(depth):
    """Set image depth."""
    global _DEPTH # noqa pylint: disable=W0603
    assert depth in [8, 16], (
        f"Image depth only supports 8 and 16, got {depth}"
    )
    _DEPTH = depth


def set_augmentations(augmentations):
    """set augmentations list."""
    global AUGMENTATIONS # noqa pylint: disable=W0603
    assert (AUGMENTATIONS is None) or (AUGMENTATIONS is augmentations), (
        "AUGMENTATIONS can only be assigned once."
    )
    AUGMENTATIONS = augmentations


def set_augmentations_val(augmentations):
    """set augmentations list."""
    global AUGMENTATIONS_VAL # noqa pylint: disable=W0603
    assert (AUGMENTATIONS_VAL is None) or (AUGMENTATIONS_VAL is augmentations), (
        "AUGMENTATIONS_VAL can only be assigned once."
    )
    AUGMENTATIONS_VAL = augmentations


def decode_image(data, channels, extension, depth=8):
    """decode jpeg and png images."""
    is_jpeg = tf.reduce_any(
            input_tensor=[
                tf.equal(extension, ".jpg"),
                tf.equal(extension, ".jpeg"),
            ]
        )
    # setting dct_method='INTEGER_ACCURATE' will produce same result
    # as PIL/openCV
    out_jpeg = tf.image.decode_jpeg(
        data,
        channels,
        dct_method='INTEGER_ACCURATE'
    )
    # Only PNG can support 16-bit
    if depth == 16:
        # Make sure the 2 branches of tf.cond have the same data type
        out_jpeg = tf.cast(out_jpeg, tf.dtypes.uint16)
        out_png = tf.image.decode_png(
            data,
            channels,
            tf.dtypes.uint16
        )
    else:
        out_png = tf.image.decode_png(
            data,
            channels,
            tf.dtypes.uint8
        )
    return tf.cond(is_jpeg, lambda: out_jpeg, lambda: out_png)


class Images2DReference(
    collections.namedtuple(
        "Images2DReference",
        ["path", "extension", "canvas_shape", "input_height", "input_width"],
    )
):
    """Reference to an image.

    Args:
        path (tf.Tensor): Tensor of type tf.string that contains the path to an image. The shape
            of this tensor can be either 1D (temporally batched) or 2D (temporal + mini batch).
        extension (tf.Tensor): Tensor of type tf.string that contains the file extension of the
            image referenced by `path`. The extension is used to determine the encoding of the
            image.  The shape of this tensor can be either 1D (temporally batched) or 2D
            (temporal + mini batch).
        canvas_shape (Canvas2D): Structure that contains the height and width of the output image.
            The static shapes of the width and height fields are used to represent the
            width and height of the image so that the static shape of the loaded image can be
            set correctly.  The shape of the tensors contained in this structure can be either
            # 1D (temporally batched) or 2D (temporal + mini batch).
        input_height (tf.Tensor): 1-D or 2-D tensor for the heights of the images on disk.
            The 1-D and 2-D case respectively correspond to the absence and presence of temporal
            batching.
        input_width (tf.Tensor): 1-D or 2-D tensor for the widths of the images on disk.
            The 1-D and 2-D case respectively correspond to the absence and presence of temporal
            batching.
    """

    def load(self, output_dtype):
        """
        Load referenced images.

        Arguments:
            output_dtype (tf.dtypes.DType): Output image dtype.

        Returns:
            (tf.Tensor) Tensor representing the loaded image. The type will be tf.float16 with
                height matching the static size of canvas_shape.height and width having the same
                size as the static size of canvas_shape.width. The shape of the tensor will be
                4D when `paths` is a 1D tensor and 5D when `paths` is a 2D tensor.
        """
        # The _decode_* functions below only handle floating point types for now.
        assert output_dtype in (tf.float32, tf.float16, tf.uint8)

        # TODO (weich): The delayed dtype cast/normalization currently doesn't support eager mode.
        # For eager mode, we force the output dtype to be float and apply normalization here
        # regardless of the image type.
        if tf.executing_eagerly():
            output_dtype = tf.float32

        path_shape = self.path.shape.as_list()
        image_count = 1
        for d in path_shape:
            if d is None:
                raise ValueError(
                    """Shape of the input was not known statically, i.e one
                of the dimensions of image paths tensor was 'None': {}. This can happen,
                for example if tf.dataset.batch(n, drop_remainder=False), as then the
                batch dimension can be variable. Image loader requires statically known batch
                dimensions, so set drop_remainder=True, or use set_shape().""".format(
                        path_shape
                    )
                )
            image_count *= d

        # Flatten tensors so that we need only a single code path to load from (via map_fn) both
        # 1D and 2D paths.
        paths_flat = tf.reshape(self.path, (-1,))
        extension_flat = tf.reshape(self.extension, (-1,))
        input_height_flat = tf.reshape(self.input_height, (-1,))
        input_width_flat = tf.reshape(self.input_width, (-1,))
        channels = _CHANNELS
        images = []
        for index in range(image_count):
            images.append(
                self._load_and_decode(
                    path=paths_flat[index],
                    extension=extension_flat[index],
                    height=input_height_flat[index],
                    width=input_width_flat[index],
                    output_dtype=output_dtype,
                    channels=channels,
                    min_side=_MIN_SIDE,
                    max_side=_MAX_SIDE,
                )
            )

        images = tf.stack(images, 0)

        # Images are always loaded as a 4D tensor because we flatten the paths before calling
        # map_fn (to avoid two code paths.) The original paths could have been a 1D tensor
        # (temporally batched) or a 2D tensor (temporal + mini batched.) We slice the
        # spatial shape of the image from the loaded 4D tensor and concatenate it with the
        # batch and/or time demensions of the paths to match the batch & time dimensionality of
        # the paths.
        if _MIN_SIDE == 0:
            # _MIN_SIDE == 0 is the normal case, image loader works in static shape
            images_shape = self.path.shape.concatenate(images.shape[1:]).as_list()
            images_shape = [-1 if dim is None else dim for dim in images_shape]
            images = tf.reshape(images, images_shape)

        return Images2D(images=images, canvas_shape=self.canvas_shape)

    def _load_and_decode(self, path, extension, height, width, output_dtype, channels=3,
                         min_side=0, max_side=0):
        if _AUTO_RESIZE:
            image = self._decode_and_resize_image(
                    height=height,
                    width=width,
                    frame_path=path,
                    channels=channels,
                    output_dtype=output_dtype,
                    extension=extension
            )
        else:
            image = self._decode_and_pad_image(
                    height=height,
                    width=width,
                    frame_path=path,
                    channels=channels,
                    output_dtype=output_dtype,
                    extension=extension
            )
        # if in min_side mode, resize and keep AR
        if min_side > 0:
            image_hwc = tf.transpose(a=image, perm=[1, 2, 0])
            target_size = self._calculate_target_size(height, width)
            image_resized = tf.image.resize_images(image_hwc, target_size)
            image = tf.cast(tf.transpose(a=image_resized, perm=[2, 0, 1]), output_dtype)
        return image

    def _calculate_target_size(self, height, width):
        """Calculate the target size for resize and keep aspect ratio."""
        height = tf.cast(height, tf.float32)
        width = tf.cast(width, tf.float32)
        min_side = tf.cast(tf.convert_to_tensor(_MIN_SIDE, dtype=tf.int32), tf.float32)
        # resize the smaller side to target_size, keep aspect ratio
        target_size = tf.cond(
            pred=tf.less_equal(height, width),
            true_fn=lambda: tf.stack([
                tf.cast(min_side, tf.int32),
                tf.cast((min_side / height) * width, tf.int32)]),
            false_fn=lambda: tf.stack([
                tf.cast((min_side / width) * height, tf.int32),
                tf.cast(min_side, tf.int32)])
        )
        # if the larger side exceeds _MAX_SIDE
        max_side = tf.cast(tf.convert_to_tensor(_MAX_SIDE, dtype=tf.int32), tf.float32)
        target_size2 = tf.cond(
            pred=tf.less_equal(height, width),
            true_fn=lambda: tf.stack([
                tf.cast((max_side / width) * height, tf.int32),
                tf.cast(max_side, tf.int32)]),
            false_fn=lambda: tf.stack([
                tf.cast(max_side, tf.int32),
                tf.cast((max_side / height) * width, tf.int32)])
        )
        target_size = tf.minimum(target_size, target_size2)
        return target_size

    def _decode_and_pad_image(
        self, height, width, frame_path,
        channels, output_dtype, extension
    ):
        """Load and decode JPG/PNG image, and pad."""
        data = tf.io.read_file(frame_path)
        image = decode_image(data, channels, extension)
        image = tf.transpose(a=image, perm=[2, 0, 1])
        if output_dtype == tf.uint8:
            image = tf.cast(image, output_dtype)
        else:
            image = tf.cast(image, output_dtype) / 255
        # For JPG and PNG with dtype uint8 we delay normalization to later in the pipeline.
        if _MIN_SIDE == 0:
            return self._pad_image_to_right_and_bottom(image)
        return image

    def _decode_and_resize_image(
        self, height, width, frame_path,
        channels, output_dtype, extension
    ):
        """Decode and resize images to target size."""
        data = tf.io.read_file(frame_path)
        image = decode_image(data, channels, extension)
        image = tf.transpose(a=image, perm=[2, 0, 1])
        if output_dtype == tf.uint8:
            image = tf.cast(image, output_dtype)
        else:
            image = tf.cast(image, output_dtype) / 255
        # For JPG and PNG with dtype uint8 we delay normalization to later in the pipeline.
        if _MIN_SIDE == 0:
            return self._resize_image_to_target_size(image)
        return image

    def _decode_and_pad_png(self, height, width, frame_path, channels, output_dtype):
        """Load and decode a PNG frame, and pad."""
        data = tf.io.read_file(frame_path)
        image = tf.image.decode_image(data, channels=channels)
        image = tf.transpose(a=image, perm=[2, 0, 1])
        if output_dtype == tf.uint8:
            image = tf.cast(image, output_dtype)
        else:
            image = tf.cast(image, output_dtype) / 255
        # For JPG and PNG with dtype uint8 we delay normalization to later in the pipeline.
        return self._pad_image_to_right_and_bottom(image)

    def _pad_image_to_right_and_bottom(self, image):
        """Pad image to the canvas shape.

        NOTE: the padding, if needed, happens to the right and bottom of the input image.

        Args:
            image (tf.Tensor): Expected to be a 3-D Tensor in [C, H, W] order.

        Returns:
            padded_image (tf.Tensor): 3-D Tensor in [C, H, W] order, where H and W may be different
                from those of the input (due to padding).
        """
        height = tf.shape(input=image)[1]
        width = tf.shape(input=image)[2]
        output_height = tf.shape(input=self.canvas_shape.height)[-1]
        output_width = tf.shape(input=self.canvas_shape.width)[-1]
        pad_height = output_height - height
        pad_width = output_width - width

        padding = [
            [0, 0],  # channels.
            [0, pad_height],  # height.
            [0, pad_width],
        ]  # width.

        padded_image = tf.pad(tensor=image, paddings=padding, mode="CONSTANT")
        # This is needed because downstream processors rely on knowing the shape statically.
        padded_image.set_shape(
            [_CHANNELS, self.canvas_shape.height.shape[-1], self.canvas_shape.width.shape[-1]]
        )

        return padded_image

    def _resize_image_to_target_size(self, image):
        """Resize image to the canvas shape.

        Args:
            image (tf.Tensor): Expected to be a 3-D Tensor in [C, H, W] order.

        Returns:
            image (tf.Tensor): 3-D Tensor in [C, H, W] order, where H and W may be different
                from those of the input (due to resize).
        """
        output_height = tf.shape(input=self.canvas_shape.height)[-1]
        output_width = tf.shape(input=self.canvas_shape.width)[-1]
        image_original = image
        image_hwc = tf.transpose(a=image, perm=[1, 2, 0])
        image_resized = tf.image.resize_images(image_hwc, [output_height, output_width])
        image_resized = tf.cast(tf.transpose(a=image_resized, perm=[2, 0, 1]), image.dtype)
        # if images are already in target size, do not resize
        # this can prevent some mistake in tensorflow resize because
        # if we still resize it the output can be different from original image.
        # so here we try to keep it untouched if it is already in target size,
        # this is compatible with old behavior if we manually pad the images
        # in dataset and it will not do resize. This is a hack to ensure
        # compatiblity of mAP.
        no_resize = tf.logical_and(
            tf.equal(tf.shape(image_original)[1], output_height),
            tf.equal(tf.shape(image_original)[2], output_width)
        )
        image = tf.cond(
            no_resize,
            true_fn=lambda: image_original,
            false_fn=lambda: image_resized
        )
        image.set_shape(
            [_CHANNELS, self.canvas_shape.height.shape[-1], self.canvas_shape.width.shape[-1]]
        )
        return image


class LabelledImages2DReference(
    collections.namedtuple(
        "LabelledImages2DReference",
        [
            "path", "extension", "canvas_shape", "input_height",
            "input_width", "labels"
        ],
    )
):
    """Reference to an image.

    Args:
        path (tf.Tensor): Tensor of type tf.string that contains the path to an image. The shape
            of this tensor can be either 1D (temporally batched) or 2D (temporal + mini batch).
        extension (tf.Tensor): Tensor of type tf.string that contains the file extension of the
            image referenced by `path`. The extension is used to determine the encoding of the
            image.  The shape of this tensor can be either 1D (temporally batched) or 2D
            (temporal + mini batch).
        canvas_shape (Canvas2D): Structure that contains the height and width of the output image.
            The static shapes of the width and height fields are used to represent the
            width and height of the image so that the static shape of the loaded image can be
            set correctly.  The shape of the tensors contained in this structure can be either
            # 1D (temporally batched) or 2D (temporal + mini batch).
        input_height (tf.Tensor): 1-D or 2-D tensor for the heights of the images on disk.
            The 1-D and 2-D case respectively correspond to the absence and presence of temporal
            batching.
        input_width (tf.Tensor): 1-D or 2-D tensor for the widths of the images on disk.
            The 1-D and 2-D case respectively correspond to the absence and presence of temporal
            batching.
    """

    def load(self, output_dtype):
        """
        Load referenced images.

        Arguments:
            output_dtype (tf.dtypes.DType): Output image dtype.

        Returns:
            (tf.Tensor) Tensor representing the loaded image. The type will be tf.float16 with
                height matching the static size of canvas_shape.height and width having the same
                size as the static size of canvas_shape.width. The shape of the tensor will be
                4D when `paths` is a 1D tensor and 5D when `paths` is a 2D tensor.
        """
        # The _decode_* functions below only handle floating point types for now.
        assert output_dtype in (tf.float32, tf.float16, tf.uint8)

        output_dtype = tf.float32

        path_shape = self.path.shape.as_list()
        image_count = 1
        for d in path_shape:
            if d is None:
                raise ValueError(
                    """Shape of the input was not known statically, i.e one
                of the dimensions of image paths tensor was 'None': {}. This can happen,
                for example if tf.dataset.batch(n, drop_remainder=False), as then the
                batch dimension can be variable. Image loader requires statically known batch
                dimensions, so set drop_remainder=True, or use set_shape().""".format(
                        path_shape
                    )
                )
            image_count *= d

        # Flatten tensors so that we need only a single code path to load from (via map_fn) both
        # 1D and 2D paths.
        paths_flat = tf.reshape(self.path, (-1,))
        extension_flat = tf.reshape(self.extension, (-1,))
        channels = _CHANNELS
        depth = _DEPTH
        images = []
        assert image_count == 1
        for index in range(image_count):
            img = self._decode_image(
                paths_flat[index],
                channels,
                depth,
                output_dtype,
                extension_flat[index]
            )
            # convert grayscale to RGB
            if channels == 1:
                img = tf.image.grayscale_to_rgb(img)
            sparse_boxes = tf.sparse.reorder(self.labels.vertices.coordinates)
            gt_labels = tf.reshape(sparse_boxes.values, [-1, 4])
            image_width = tf.cast(tf.shape(img)[1], tf.float32)
            if AUGMENTATIONS is not None:
                # ratio is only valid for single scale training
                ratio = W_TENSOR / H_TENSOR
                img, gt_labels = AUGMENTATIONS(img, gt_labels, ratio, image_width)
            new_labels = self._update_augmented_labels(gt_labels, sparse_boxes)
            img = tf.expand_dims(img, axis=0)
            img.set_shape([1, None, None, 3])
            images.append(img)
        shapes = tf.reshape(tf.shape(images[0])[1:3], (1, 2))
        images = tf.image.pad_to_bounding_box(
            images[0],
            0,
            0,
            tf.shape(self.canvas_shape.height)[-1],
            tf.shape(self.canvas_shape.width)[-1]
        )
        return LabelledImages2D(images=images, labels=new_labels, shapes=shapes)

    def _get_dense_bboxes(self):
        coords = tf.sparse.reshape(self.labels.vertices.coordinates, [-1, 4])
        bboxes = tf.reshape(
            tf.sparse.to_dense(coords),
            (-1, 4)
        )
        h = tf.cast(tf.reshape(self.input_height, []), tf.float32)
        w = tf.cast(tf.reshape(self.input_width, []), tf.float32)
        # normalized coordinates for augmentations
        bboxes /= tf.stack([w, h, w, h], axis=-1)
        return bboxes

    def _update_augmented_labels(self, bboxes, sparse_bboxes):
        sparsed = tf.sparse.SparseTensor(
            indices=sparse_bboxes.indices,
            values=tf.reshape(bboxes, [-1]),
            dense_shape=sparse_bboxes.dense_shape
        )
        new_vertices = self.labels.vertices._replace(coordinates=sparsed)
        new_labels = self.labels._replace(vertices=new_vertices)
        return new_labels

    def _load_and_decode(self, path, extension, height, width, output_dtype, channels=3,
                         min_side=0, max_side=0, h_tensor=None, w_tensor=None):
        if _AUTO_RESIZE:
            image = self._decode_and_resize_image(
                    height=height,
                    width=width,
                    frame_path=path,
                    channels=channels,
                    output_dtype=output_dtype,
                    extension=extension
            )
        elif (h_tensor is None) and (w_tensor is None):
            image = self._decode_and_pad_image(
                    height=height,
                    width=width,
                    frame_path=path,
                    channels=channels,
                    output_dtype=output_dtype,
                    extension=extension
            )
        else:
            # dynamic shape with h_tensor and w_tensor
            image = self._decode_and_resize_image(
                    height=height,
                    width=width,
                    frame_path=path,
                    channels=channels,
                    output_dtype=output_dtype,
                    extension=extension,
                    h_tensor=h_tensor,
                    w_tensor=w_tensor
            )
        # if in min_side mode, resize and keep AR
        if min_side > 0:
            image_hwc = tf.transpose(a=image, perm=[1, 2, 0])
            target_size = self._calculate_target_size(height, width)
            image_resized = tf.image.resize_images(image_hwc, target_size)
            image = tf.cast(tf.transpose(a=image_resized, perm=[2, 0, 1]), output_dtype)
        return image

    def _calculate_target_size(self, height, width):
        """Calculate the target size for resize and keep aspect ratio."""
        height = tf.cast(height, tf.float32)
        width = tf.cast(width, tf.float32)
        min_side = tf.cast(tf.convert_to_tensor(_MIN_SIDE, dtype=tf.int32), tf.float32)
        # resize the smaller side to target_size, keep aspect ratio
        target_size = tf.cond(
            pred=tf.less_equal(height, width),
            true_fn=lambda: tf.stack([
                tf.cast(min_side, tf.int32),
                tf.cast((min_side / height) * width, tf.int32)]),
            false_fn=lambda: tf.stack([
                tf.cast((min_side / width) * height, tf.int32),
                tf.cast(min_side, tf.int32)])
        )
        # if the larger side exceeds _MAX_SIDE
        max_side = tf.cast(tf.convert_to_tensor(_MAX_SIDE, dtype=tf.int32), tf.float32)
        target_size2 = tf.cond(
            pred=tf.less_equal(height, width),
            true_fn=lambda: tf.stack([
                tf.cast((max_side / width) * height, tf.int32),
                tf.cast(max_side, tf.int32)]),
            false_fn=lambda: tf.stack([
                tf.cast(max_side, tf.int32),
                tf.cast((max_side / height) * width, tf.int32)])
        )
        target_size = tf.minimum(target_size, target_size2)
        return target_size

    def _decode_and_pad_image(
        self, height, width, frame_path,
        channels, output_dtype, extension
    ):
        """Load and decode JPG/PNG image, and pad."""
        data = tf.io.read_file(frame_path)
        image = decode_image(data, channels, extension)
        image = tf.transpose(a=image, perm=[2, 0, 1])
        if output_dtype == tf.uint8:
            image = tf.cast(image, output_dtype)
        else:
            image = tf.cast(image, output_dtype) / 255
        # For JPG and PNG with dtype uint8 we delay normalization to later in the pipeline.
        if _MIN_SIDE == 0:
            return self._pad_image_to_right_and_bottom(image)
        return image

    def _decode_and_resize_image(
        self, height, width, frame_path,
        channels, output_dtype, extension,
        h_tensor=None, w_tensor=None
    ):
        """Decode and resize images to target size."""
        data = tf.io.read_file(frame_path)
        image = decode_image(data, channels, extension)
        image = tf.transpose(a=image, perm=[2, 0, 1])
        if output_dtype == tf.uint8:
            image = tf.cast(image, output_dtype)
        else:
            image = tf.cast(image, output_dtype) / 255
        # For JPG and PNG with dtype uint8 we delay normalization to later in the pipeline.
        if _MIN_SIDE == 0:
            return self._resize_image_to_target_size(
                image,
                h_tensor=h_tensor,
                w_tensor=w_tensor
            )
        return image

    def _decode_image(
        self, frame_path, channels, depth, output_dtype, extension
    ):
        """Decode single image."""
        data = tf.io.read_file(frame_path)
        image = tf.cast(decode_image(data, channels, extension, depth=depth), output_dtype)
        return image

    def resize_image(
        self,
        image,
        h_tensor,
        w_tensor
    ):
        """Resize image to target size."""
        image_original = image
        image_resized = tf.image.resize_images(image, tf.stack([h_tensor, w_tensor], axis=0))
        # if images are already in target size, do not resize
        # this can prevent some mistake in tensorflow resize because
        # if we still resize it the output can be different from original image.
        # so here we try to keep it untouched if it is already in target size,
        # this is compatible with old behavior if we manually pad the images
        # in dataset and it will not do resize. This is a hack to ensure
        # compatiblity of mAP.
        no_resize = tf.logical_and(
            tf.equal(tf.shape(image_original)[0], h_tensor),
            tf.equal(tf.shape(image_original)[1], w_tensor)
        )
        image = tf.cond(
            no_resize,
            true_fn=lambda: image_original,
            false_fn=lambda: image_resized
        )
        # HWC to CHW
        image = tf.transpose(image, (2, 0, 1))
        return image

    def _decode_and_pad_png(self, height, width, frame_path, channels, output_dtype):
        """Load and decode a PNG frame, and pad."""
        data = tf.io.read_file(frame_path)
        image = tf.image.decode_image(data, channels=channels)
        image = tf.transpose(a=image, perm=[2, 0, 1])
        if output_dtype == tf.uint8:
            image = tf.cast(image, output_dtype)
        else:
            image = tf.cast(image, output_dtype) / 255
        # For JPG and PNG with dtype uint8 we delay normalization to later in the pipeline.
        return self._pad_image_to_right_and_bottom(image)

    def _pad_image_to_right_and_bottom(self, image):
        """Pad image to the canvas shape.

        NOTE: the padding, if needed, happens to the right and bottom of the input image.

        Args:
            image (tf.Tensor): Expected to be a 3-D Tensor in [C, H, W] order.

        Returns:
            padded_image (tf.Tensor): 3-D Tensor in [C, H, W] order, where H and W may be different
                from those of the input (due to padding).
        """
        height = tf.shape(input=image)[1]
        width = tf.shape(input=image)[2]
        output_height = tf.shape(input=self.canvas_shape.height)[-1]
        output_width = tf.shape(input=self.canvas_shape.width)[-1]
        pad_height = output_height - height
        pad_width = output_width - width

        padding = [
            [0, 0],  # channels.
            [0, pad_height],  # height.
            [0, pad_width],
        ]  # width.

        padded_image = tf.pad(tensor=image, paddings=padding, mode="CONSTANT")
        # This is needed because downstream processors rely on knowing the shape statically.
        padded_image.set_shape(
            [_CHANNELS, self.canvas_shape.height.shape[-1], self.canvas_shape.width.shape[-1]]
        )

        return padded_image

    def _resize_image_to_target_size(self, image, h_tensor=None, w_tensor=None):
        """Resize image to the canvas shape.

        Args:
            image (tf.Tensor): Expected to be a 3-D Tensor in [C, H, W] order.

        Returns:
            image (tf.Tensor): 3-D Tensor in [C, H, W] order, where H and W may be different
                from those of the input (due to resize).
        """
        if h_tensor is None:
            output_height = tf.shape(input=self.canvas_shape.height)[-1]
        else:
            output_height = h_tensor
        if w_tensor is None:
            output_width = tf.shape(input=self.canvas_shape.width)[-1]
        else:
            output_width = tf.shape(input=self.canvas_shape.width)[-1]
        image_original = image
        image_hwc = tf.transpose(a=image, perm=[1, 2, 0])
        image_resized = tf.image.resize_images(image_hwc, [output_height, output_width])
        image_resized = tf.cast(tf.transpose(a=image_resized, perm=[2, 0, 1]), image.dtype)
        # if images are already in target size, do not resize
        # this can prevent some mistake in tensorflow resize because
        # if we still resize it the output can be different from original image.
        # so here we try to keep it untouched if it is already in target size,
        # this is compatible with old behavior if we manually pad the images
        # in dataset and it will not do resize. This is a hack to ensure
        # compatiblity of mAP.
        no_resize = tf.logical_and(
            tf.equal(tf.shape(image_original)[1], output_height),
            tf.equal(tf.shape(image_original)[2], output_width)
        )
        image = tf.cond(
            no_resize,
            true_fn=lambda: image_original,
            false_fn=lambda: image_resized
        )
        if (h_tensor is None) and (w_tensor is None):
            image.set_shape(
                [_CHANNELS, self.canvas_shape.height.shape[-1], self.canvas_shape.width.shape[-1]]
            )
        return image


class LabelledImages2DReferenceVal(
    collections.namedtuple(
        "LabelledImages2DReferenceVal",
        [
            "path", "extension", "canvas_shape", "input_height",
            "input_width", "labels"
        ],
    )
):
    """Reference to an image.

    Args:
        path (tf.Tensor): Tensor of type tf.string that contains the path to an image. The shape
            of this tensor can be either 1D (temporally batched) or 2D (temporal + mini batch).
        extension (tf.Tensor): Tensor of type tf.string that contains the file extension of the
            image referenced by `path`. The extension is used to determine the encoding of the
            image.  The shape of this tensor can be either 1D (temporally batched) or 2D
            (temporal + mini batch).
        canvas_shape (Canvas2D): Structure that contains the height and width of the output image.
            The static shapes of the width and height fields are used to represent the
            width and height of the image so that the static shape of the loaded image can be
            set correctly.  The shape of the tensors contained in this structure can be either
            # 1D (temporally batched) or 2D (temporal + mini batch).
        input_height (tf.Tensor): 1-D or 2-D tensor for the heights of the images on disk.
            The 1-D and 2-D case respectively correspond to the absence and presence of temporal
            batching.
        input_width (tf.Tensor): 1-D or 2-D tensor for the widths of the images on disk.
            The 1-D and 2-D case respectively correspond to the absence and presence of temporal
            batching.
    """

    def load(self, output_dtype):
        """
        Load referenced images.

        Arguments:
            output_dtype (tf.dtypes.DType): Output image dtype.

        Returns:
            (tf.Tensor) Tensor representing the loaded image. The type will be tf.float16 with
                height matching the static size of canvas_shape.height and width having the same
                size as the static size of canvas_shape.width. The shape of the tensor will be
                4D when `paths` is a 1D tensor and 5D when `paths` is a 2D tensor.
        """
        # The _decode_* functions below only handle floating point types for now.
        assert output_dtype in (tf.float32, tf.float16, tf.uint8)

        output_dtype = tf.float32

        path_shape = self.path.shape.as_list()
        image_count = 1
        for d in path_shape:
            if d is None:
                raise ValueError(
                    """Shape of the input was not known statically, i.e one
                of the dimensions of image paths tensor was 'None': {}. This can happen,
                for example if tf.dataset.batch(n, drop_remainder=False), as then the
                batch dimension can be variable. Image loader requires statically known batch
                dimensions, so set drop_remainder=True, or use set_shape().""".format(
                        path_shape
                    )
                )
            image_count *= d

        # Flatten tensors so that we need only a single code path to load from (via map_fn) both
        # 1D and 2D paths.
        paths_flat = tf.reshape(self.path, (-1,))
        extension_flat = tf.reshape(self.extension, (-1,))
        input_height_flat = tf.reshape(self.input_height, (-1,))
        input_width_flat = tf.reshape(self.input_width, (-1,))
        bbox_normalizer = tf.concat(
            [
                input_width_flat, input_height_flat,
                input_width_flat, input_height_flat
            ],
            axis=-1
        )
        channels = _CHANNELS
        depth = _DEPTH
        images = []
        assert image_count == 1
        for index in range(image_count):
            img = self._decode_image(
                paths_flat[index],
                channels,
                depth,
                output_dtype,
                extension_flat[index]
            )
            # convert grayscale to RGB
            if channels == 1:
                img = tf.image.grayscale_to_rgb(img)
            sparse_boxes = tf.sparse.reorder(self.labels.vertices.coordinates)
            gt_labels = tf.reshape(sparse_boxes.values, [-1, 4])
            gt_labels /= tf.cast(bbox_normalizer, tf.float32)
            if AUGMENTATIONS_VAL is not None:
                # inference/evaluation
                target_shape = tf.stack([W_TENSOR_VAL, H_TENSOR_VAL], axis=0)
                img, gt_labels = AUGMENTATIONS_VAL(img, gt_labels, target_shape)
                gt_labels *= tf.cast(tf.concat([target_shape, target_shape], axis=-1), tf.float32)
            new_labels = self._update_augmented_labels(gt_labels, sparse_boxes)
            img = tf.expand_dims(img, axis=0)
            img.set_shape([1, None, None, 3])
            images.append(img)
        shapes = tf.reshape(tf.shape(images[0])[1:3], (1, 2))
        return LabelledImages2D(images=images[0], labels=new_labels, shapes=shapes)

    def _get_dense_bboxes(self):
        coords = tf.sparse.reshape(self.labels.vertices.coordinates, [-1, 4])
        bboxes = tf.reshape(
            tf.sparse.to_dense(coords),
            (-1, 4)
        )
        h = tf.cast(tf.reshape(self.input_height, []), tf.float32)
        w = tf.cast(tf.reshape(self.input_width, []), tf.float32)
        # normalized coordinates for augmentations
        bboxes /= tf.stack([w, h, w, h], axis=-1)
        return bboxes

    def _update_augmented_labels(self, bboxes, sparse_bboxes):
        sparsed = tf.sparse.SparseTensor(
            indices=sparse_bboxes.indices,
            values=tf.reshape(bboxes, [-1]),
            dense_shape=sparse_bboxes.dense_shape
        )
        new_vertices = self.labels.vertices._replace(coordinates=sparsed)
        new_labels = self.labels._replace(vertices=new_vertices)
        return new_labels

    def _load_and_decode(self, path, extension, height, width, output_dtype, channels=3,
                         min_side=0, max_side=0, h_tensor=None, w_tensor=None):
        if _AUTO_RESIZE:
            image = self._decode_and_resize_image(
                    height=height,
                    width=width,
                    frame_path=path,
                    channels=channels,
                    output_dtype=output_dtype,
                    extension=extension
            )
        elif (h_tensor is None) and (w_tensor is None):
            image = self._decode_and_pad_image(
                    height=height,
                    width=width,
                    frame_path=path,
                    channels=channels,
                    output_dtype=output_dtype,
                    extension=extension
            )
        else:
            # dynamic shape with h_tensor and w_tensor
            image = self._decode_and_resize_image(
                    height=height,
                    width=width,
                    frame_path=path,
                    channels=channels,
                    output_dtype=output_dtype,
                    extension=extension,
                    h_tensor=h_tensor,
                    w_tensor=w_tensor
            )
        # if in min_side mode, resize and keep AR
        if min_side > 0:
            image_hwc = tf.transpose(a=image, perm=[1, 2, 0])
            target_size = self._calculate_target_size(height, width)
            image_resized = tf.image.resize_images(image_hwc, target_size)
            image = tf.cast(tf.transpose(a=image_resized, perm=[2, 0, 1]), output_dtype)
        return image

    def _calculate_target_size(self, height, width):
        """Calculate the target size for resize and keep aspect ratio."""
        height = tf.cast(height, tf.float32)
        width = tf.cast(width, tf.float32)
        min_side = tf.cast(tf.convert_to_tensor(_MIN_SIDE, dtype=tf.int32), tf.float32)
        # resize the smaller side to target_size, keep aspect ratio
        target_size = tf.cond(
            pred=tf.less_equal(height, width),
            true_fn=lambda: tf.stack([
                tf.cast(min_side, tf.int32),
                tf.cast((min_side / height) * width, tf.int32)]),
            false_fn=lambda: tf.stack([
                tf.cast((min_side / width) * height, tf.int32),
                tf.cast(min_side, tf.int32)])
        )
        # if the larger side exceeds _MAX_SIDE
        max_side = tf.cast(tf.convert_to_tensor(_MAX_SIDE, dtype=tf.int32), tf.float32)
        target_size2 = tf.cond(
            pred=tf.less_equal(height, width),
            true_fn=lambda: tf.stack([
                tf.cast((max_side / width) * height, tf.int32),
                tf.cast(max_side, tf.int32)]),
            false_fn=lambda: tf.stack([
                tf.cast(max_side, tf.int32),
                tf.cast((max_side / height) * width, tf.int32)])
        )
        target_size = tf.minimum(target_size, target_size2)
        return target_size

    def _decode_and_pad_image(
        self, height, width, frame_path,
        channels, output_dtype, extension
    ):
        """Load and decode JPG/PNG image, and pad."""
        data = tf.io.read_file(frame_path)
        image = decode_image(data, channels, extension)
        image = tf.transpose(a=image, perm=[2, 0, 1])
        if output_dtype == tf.uint8:
            image = tf.cast(image, output_dtype)
        else:
            image = tf.cast(image, output_dtype) / 255
        # For JPG and PNG with dtype uint8 we delay normalization to later in the pipeline.
        if _MIN_SIDE == 0:
            return self._pad_image_to_right_and_bottom(image)
        return image

    def _decode_and_resize_image(
        self, height, width, frame_path,
        channels, output_dtype, extension,
        h_tensor=None, w_tensor=None
    ):
        """Decode and resize images to target size."""
        data = tf.io.read_file(frame_path)
        image = decode_image(data, channels, extension)
        image = tf.transpose(a=image, perm=[2, 0, 1])
        if output_dtype == tf.uint8:
            image = tf.cast(image, output_dtype)
        else:
            image = tf.cast(image, output_dtype) / 255
        # For JPG and PNG with dtype uint8 we delay normalization to later in the pipeline.
        if _MIN_SIDE == 0:
            return self._resize_image_to_target_size(
                image,
                h_tensor=h_tensor,
                w_tensor=w_tensor
            )
        return image

    def _decode_image(
        self, frame_path, channels, depth, output_dtype, extension
    ):
        """Decode single image."""
        data = tf.io.read_file(frame_path)
        image = tf.cast(decode_image(data, channels, extension, depth=depth), output_dtype)
        return image

    def resize_image(
        self,
        image,
        h_tensor,
        w_tensor
    ):
        """Resize image to target size."""
        image_original = image
        image_resized = tf.image.resize_images(image, tf.stack([h_tensor, w_tensor], axis=0))
        # if images are already in target size, do not resize
        # this can prevent some mistake in tensorflow resize because
        # if we still resize it the output can be different from original image.
        # so here we try to keep it untouched if it is already in target size,
        # this is compatible with old behavior if we manually pad the images
        # in dataset and it will not do resize. This is a hack to ensure
        # compatiblity of mAP.
        no_resize = tf.logical_and(
            tf.equal(tf.shape(image_original)[0], h_tensor),
            tf.equal(tf.shape(image_original)[1], w_tensor)
        )
        image = tf.cond(
            no_resize,
            true_fn=lambda: image_original,
            false_fn=lambda: image_resized
        )
        # HWC to CHW
        image = tf.transpose(image, (2, 0, 1))
        return image

    def _decode_and_pad_png(self, height, width, frame_path, channels, output_dtype):
        """Load and decode a PNG frame, and pad."""
        data = tf.io.read_file(frame_path)
        image = tf.image.decode_image(data, channels=channels)
        image = tf.transpose(a=image, perm=[2, 0, 1])
        if output_dtype == tf.uint8:
            image = tf.cast(image, output_dtype)
        else:
            image = tf.cast(image, output_dtype) / 255
        # For JPG and PNG with dtype uint8 we delay normalization to later in the pipeline.
        return self._pad_image_to_right_and_bottom(image)

    def _pad_image_to_right_and_bottom(self, image):
        """Pad image to the canvas shape.

        NOTE: the padding, if needed, happens to the right and bottom of the input image.

        Args:
            image (tf.Tensor): Expected to be a 3-D Tensor in [C, H, W] order.

        Returns:
            padded_image (tf.Tensor): 3-D Tensor in [C, H, W] order, where H and W may be different
                from those of the input (due to padding).
        """
        height = tf.shape(input=image)[1]
        width = tf.shape(input=image)[2]
        output_height = tf.shape(input=self.canvas_shape.height)[-1]
        output_width = tf.shape(input=self.canvas_shape.width)[-1]
        pad_height = output_height - height
        pad_width = output_width - width

        padding = [
            [0, 0],  # channels.
            [0, pad_height],  # height.
            [0, pad_width],
        ]  # width.

        padded_image = tf.pad(tensor=image, paddings=padding, mode="CONSTANT")
        # This is needed because downstream processors rely on knowing the shape statically.
        padded_image.set_shape(
            [_CHANNELS, self.canvas_shape.height.shape[-1], self.canvas_shape.width.shape[-1]]
        )

        return padded_image

    def _resize_image_to_target_size(self, image, h_tensor=None, w_tensor=None):
        """Resize image to the canvas shape.

        Args:
            image (tf.Tensor): Expected to be a 3-D Tensor in [C, H, W] order.

        Returns:
            image (tf.Tensor): 3-D Tensor in [C, H, W] order, where H and W may be different
                from those of the input (due to resize).
        """
        if h_tensor is None:
            output_height = tf.shape(input=self.canvas_shape.height)[-1]
        else:
            output_height = h_tensor
        if w_tensor is None:
            output_width = tf.shape(input=self.canvas_shape.width)[-1]
        else:
            output_width = tf.shape(input=self.canvas_shape.width)[-1]
        image_original = image
        image_hwc = tf.transpose(a=image, perm=[1, 2, 0])
        image_resized = tf.image.resize_images(image_hwc, [output_height, output_width])
        image_resized = tf.cast(tf.transpose(a=image_resized, perm=[2, 0, 1]), image.dtype)
        # if images are already in target size, do not resize
        # this can prevent some mistake in tensorflow resize because
        # if we still resize it the output can be different from original image.
        # so here we try to keep it untouched if it is already in target size,
        # this is compatible with old behavior if we manually pad the images
        # in dataset and it will not do resize. This is a hack to ensure
        # compatiblity of mAP.
        no_resize = tf.logical_and(
            tf.equal(tf.shape(image_original)[1], output_height),
            tf.equal(tf.shape(image_original)[2], output_width)
        )
        image = tf.cond(
            no_resize,
            true_fn=lambda: image_original,
            false_fn=lambda: image_resized
        )
        if (h_tensor is None) and (w_tensor is None):
            image.set_shape(
                [_CHANNELS, self.canvas_shape.height.shape[-1], self.canvas_shape.width.shape[-1]]
            )
        return image
