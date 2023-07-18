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

"""Base visualizer element defining basic elements."""

from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class Descriptor(object):
    """Descriptor setter and getter for class properties."""

    def __init__(self, attr):
        """Constructor for the descriptor."""
        self.attr = attr

    def __get__(self, instance, owner):
        """Getter of the property.

        Checks that the Visualizer is actually built before returning the value.

        Args:
            instance: Instance of the Visualizer (Will be None, because the
                descriptor is called for the class instead of an instance).
            owner: The owner class.
        """
        if not owner._built:
            raise RuntimeError(f"The {type(owner)} wasn't built.")
        return getattr(owner, self.attr)


class BaseVisualizer(object):
    """Base visualizer class for TAO-TF."""

    def __init_sublass__(cls, **kwargs):
        """Constructor for the base visualizer."""
        cls._enabled = False
        cls._num_images = 3
        cls._built = False

    @classmethod
    def build(cls, enabled, num_images):
        """Build the visualizer."""
        cls._enabled = enabled
        cls._num_images = num_images
        cls._built = True

    enabled = Descriptor("_enabled")
    num_images = Descriptor("_num_images")
    built = Descriptor("_built")

    @classmethod
    @contextmanager
    def disable(cls):
        """Context manager for temporarily disabling the visualizations."""
        if not cls._built:
            raise RuntimeError("Visualizer was not built to disabled.")
        old_state = cls._enabled
        cls._enabled = False
        yield
        cls._enabled = old_state

    @classmethod
    def image(cls, tensor_name, tensor_value, value_range=None):
        """Visualizer function to render image."""
        raise NotImplementedError("This method is not implemented in the base class.")

    @classmethod
    def histogram(cls, tensor_name, tensor_value):
        """Visualize histogram for a given tensor."""
        raise NotImplementedError("This method hasn't been implemented in the base class.")

    @classmethod
    def scalar(cls, tensor_name, tensor_value):
        """Render a scalar in the visualizer."""
        raise NotImplementedError("This method hasn't been implemented in the base class.")
