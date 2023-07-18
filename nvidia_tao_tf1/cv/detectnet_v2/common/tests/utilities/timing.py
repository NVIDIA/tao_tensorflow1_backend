# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Timing related test utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


TIME_DELTA = 1.5


class FakeTime(object):
    """Can be used to replace to built-in time function."""

    _NUM_CALLS = 0

    @classmethod
    def time(cls):
        """Time method."""
        new_timestamp = cls._NUM_CALLS * TIME_DELTA
        # Next time this is called, returns (_NUM_CALLS + 1) * TIME_DELTA.
        cls._NUM_CALLS += 1

        return new_timestamp
