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

"""Some test fixtures and utility functions."""

import numpy as np


def get_random_boolean_mask(shape):
    """Get a random boolean mask.

    Args:
        shape (list or tuple): Ints indicating the shape of the returned array.

    Returns:
        mask (np.array): int32 array with 0s and 1s.
    """
    num_values = np.prod(shape)
    mask = np.zeros(num_values, dtype=np.int32)
    # Set half of the values to True.
    mask[: num_values // 2] = True
    # In place shuffle.
    np.random.shuffle(mask)
    mask = mask.reshape(shape)

    # Sanity check there's at least one 0.0 and one 1.0.
    assert mask.min() == 0
    assert mask.max() == 1

    return mask
