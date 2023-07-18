"""Utilities for ImageNet data preprocessing & prediction decoding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from keras import backend as K
import numpy as np
logger = logging.getLogger(__name__)

CLASS_INDEX = None
CLASS_INDEX_PATH = ('https://storage.googleapis.com/download.tensorflow.org/'
                    'data/imagenet_class_index.json')

# Global tensor of imagenet mean for preprocessing symbolic inputs
_IMAGENET_MEAN = None

# Keras constants.
_KERAS_BACKEND = K
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None


def get_submodules_from_kwargs(kwargs):
    """Simple function to extract submodules from kwargs."""
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in list(kwargs.keys()):
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: {}'.format(key))
    return backend, layers, models, utils


def _preprocess_numpy_input(x, data_format, mode, color_mode, img_mean, **kwargs):
    """Preprocesses a Numpy array encoding a batch of images.

    # Arguments
        x: Input array, 3D or 4D.
        data_format: Data format of the image array.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.

    # Returns
        Preprocessed Numpy array.
    """
    backend, _, _, _ = get_submodules_from_kwargs(kwargs)
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(backend.floatx(), copy=False)

    if mode == 'tf':
        if img_mean and len(img_mean) > 0:
            logger.debug("image_mean is ignored in tf mode.")
        x /= 127.5
        x -= 1.
        return x

    if mode == 'torch':
        if img_mean and len(img_mean) > 0:
            logger.debug("image_mean is ignored in torch mode.")
        x /= 255.
        if color_mode == "rgb":
            mean = [0.485, 0.456, 0.406]
            std = [0.224, 0.224, 0.224]
        elif color_mode == "grayscale":
            mean = [0.449]
            std = [0.224]
        else:
            raise NotImplementedError("Invalid color mode: {}".format(color_mode))
    else:
        if color_mode == "rgb":
            if data_format == 'channels_first':
                # 'RGB'->'BGR'
                if x.ndim == 3:
                    x = x[::-1, ...]
                else:
                    x = x[:, ::-1, ...]
            else:
                # 'RGB'->'BGR'
                x = x[..., ::-1]
            if not img_mean:
                mean = [103.939, 116.779, 123.68]
            else:
                assert len(img_mean) == 3, "image_mean must be a list of 3 values \
                    for RGB input."
                mean = img_mean
            std = None
        else:
            if not img_mean:
                mean = [117.3786]
            else:
                assert len(img_mean) == 1, "image_mean must be a list of a single value \
                    for gray image input."
                mean = img_mean
            std = None

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        for idx in range(len(mean)):
            if x.ndim == 3:
                x[idx, :, :] -= mean[idx]
                if std is not None:
                    x[idx, :, :] /= std[idx]
            else:
                x[:, idx, :, :] -= mean[idx]
                if std is not None:
                    x[:, idx, :, :] /= std[idx]
    else:
        for idx in range(len(mean)):
            x[..., idx] -= mean[idx]
            if std is not None:
                x[..., idx] /= std[idx]
    return x


def _preprocess_symbolic_input(x, data_format, mode, color_mode, img_mean, **kwargs):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: Input tensor, 3D or 4D.
        data_format: Data format of the image tensor.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.

    # Returns
        Preprocessed tensor.
    """
    global _IMAGENET_MEAN  # noqa pylint: disable=global-statement

    backend, _, _, _ = get_submodules_from_kwargs(kwargs)

    if mode == 'tf':
        if img_mean and len(img_mean) > 0:
            logger.debug("image_mean is ignored in tf mode.")
        x /= 127.5
        x -= 1.
        return x

    if mode == 'torch':
        if img_mean and len(img_mean) > 0:
            logger.debug("image_mean is ignored in torch mode.")
        x /= 255.
        if color_mode == "rgb":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif color_mode == "grayscale":
            mean = [0.449]
            std = [0.226]
        else:
            raise NotImplementedError("Invalid color mode: {}".format(color_mode))
    else:
        if color_mode == "rgb":
            if data_format == 'channels_first':
                # 'RGB'->'BGR'
                if backend.ndim(x) == 3:
                    x = x[::-1, ...]
                else:
                    x = x[:, ::-1, ...]
            else:
                # 'RGB'->'BGR'
                x = x[..., ::-1]
            if not img_mean:
                mean = [103.939, 116.779, 123.68]
            else:
                assert len(img_mean) == 3, "image_mean must be a list of 3 values \
                    for RGB input."
                mean = img_mean
            std = None
        else:
            if not img_mean:
                mean = [117.3786]
            else:
                assert len(img_mean) == 1, "image_mean must be a list of a single value \
                    for gray image input."
                mean = img_mean
            std = None

    if _IMAGENET_MEAN is None:
        _IMAGENET_MEAN = backend.constant(-np.array(mean))

    # Zero-center by mean pixel
    if backend.dtype(x) != backend.dtype(_IMAGENET_MEAN):
        x = backend.bias_add(
            x, backend.cast(_IMAGENET_MEAN, backend.dtype(x)),
            data_format=data_format)
    else:
        x = backend.bias_add(x, _IMAGENET_MEAN, data_format)
    if std is not None:
        x /= std
    return x


def preprocess_input(x, data_format=None, mode='caffe', color_mode="rgb", img_mean=None, **kwargs):
    """Preprocesses a tensor or Numpy array encoding a batch of images.

    # Arguments
        x: Input Numpy or symbolic tensor, 3D or 4D.
            The preprocessed data is written over the input data
            if the data types are compatible. To avoid this
            behaviour, `numpy.copy(x)` can be used.
        data_format: Data format of the image tensor/array.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.

    # Returns
        Preprocessed tensor or Numpy array.

    # Raises
        ValueError: In case of unknown `data_format` argument.
    """
    backend, _, _, _ = get_submodules_from_kwargs(kwargs)

    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))

    if isinstance(x, np.ndarray):
        return _preprocess_numpy_input(x, data_format=data_format,
                                       mode=mode, color_mode=color_mode,
                                       img_mean=img_mean, **kwargs)
    return _preprocess_symbolic_input(x, data_format=data_format,
                                      mode=mode, color_mode=color_mode,
                                      img_mean=img_mean, **kwargs)
