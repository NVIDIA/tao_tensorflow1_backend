# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Barebones timer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import timedelta
from time import time

from decorator import decorator

from nvidia_tao_tf1.core import distribution


class time_function(object):
    """Decorator that prints the runtime of a wrapped function."""

    def __init__(self, prefix=""):
        """Constructor.

        Args:
            prefix (str): Prefix to append to the time print out. Defaults to no prefix (empty
                string). This can be e.g. a module's name, or a helpful descriptive message.
        """
        self._prefix = prefix
        self._is_master = distribution.get_distributor().is_master()

    def __call__(self, fn):
        """Wrap the call to the function.

        Args:
            fn (function): Function to be wrapped.

        Returns:
            wrapped_fn (function): Wrapped function.
        """
        @decorator
        def wrapped_fn(fn, *args, **kwargs):
            if self._is_master:
                # Only time if in master process.
                start = time()

            # Run function as usual.
            return_args = fn(*args, **kwargs)

            if self._is_master:
                time_taken = timedelta(seconds=(time() - start))
                print("Time taken to run %s: %s." %
                      (self._prefix + ":" + fn.__name__, time_taken))

            return return_args

        return wrapped_fn(fn)
