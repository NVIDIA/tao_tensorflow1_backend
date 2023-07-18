# Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.

"""Saver callback."""

import os
from keras.callbacks import Callback


def _save_model(keras_model, model_path, key, save_format=None):
    """Save a model to either .h5 or .hdf5 format."""

    _, ext = os.path.splitext(model_path)
    if (save_format is not None) and (save_format != ext):
        # recursive call to save a correct model
        return _save_model(keras_model, model_path + save_format, key, None)

    if ext == '.h5':
        keras_model.save_weights(model_path)
    elif ext == '.hdf5':
        # include optimiizer for training resume
        keras_model.save(model_path, overwrite=True, include_optimizer=True)
    else:
        raise NotImplementedError("{0} file is not supported for saving. Please use .hdf5!".format(ext))

    return model_path


class KerasModelSaver(Callback):
    """Save the encrypted model after every epoch.

    Attributes:
        filepath: formated string for saving models. E.g.: 'ssd_resnet18_epoch_{epoch:03d}.tlt'
        key: API key to encrypt the model.
        save_period: save model every k epoch. If save_period = 10, saver will save 10-th, 20th etc.
            epoch models
        verbose: Whether to print out save message.
    """

    def __init__(self,
                 filepath,
                 key,
                 save_period,
                 last_epoch=None,
                 verbose=1):
        """Initialization with encryption key."""
        self.filepath = filepath
        self._ENC_KEY = str.encode(key)
        self.verbose = verbose
        self.save_period = int(save_period)
        self.last_epoch = last_epoch
        self.ema = None
        assert self.save_period > 0, "save_period must be a positive integer!"

    def _save_model(self, save_epoch):
        fname = self.filepath.format(epoch=save_epoch)

        if self.ema:
            orig_weights = self.model.get_weights()
            self.model.set_weights(self.ema)
            fname = _save_model(self.model, fname, self._ENC_KEY, '.hdf5')
            self.model.set_weights(orig_weights)
        else:
            fname = _save_model(self.model, fname, self._ENC_KEY, '.hdf5')

        if self.verbose > 0:
            print('\nEpoch %05d: saving model to %s' % (save_epoch, fname))

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""

        if (epoch + 1) % self.save_period == 0 or self.last_epoch == (epoch + 1):
            self._save_model(epoch + 1)
