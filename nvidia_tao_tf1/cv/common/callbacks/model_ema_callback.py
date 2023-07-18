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

"""ModelEMA."""

import math
from keras.callbacks import Callback


class ModelEMACallback(Callback):
    """Model Exponential Moving Average for keras."""

    def __init__(self, model, decay=0.999, init_step=0):
        """Init."""
        self.ema = model.get_weights()
        self.decay = lambda x: decay * (1 - math.exp(-float(x) / 2000))
        self.updates = init_step

    def on_batch_end(self, batch, logs=None):
        """On batch end call."""
        self.updates += 1
        d = self.decay(self.updates)
        new_weights = self.model.get_weights()
        for w1, w2 in zip(self.ema, new_weights):
            w1 *= d
            w1 += (1.0 - d) * w2
