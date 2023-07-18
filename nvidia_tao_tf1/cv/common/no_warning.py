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

"""Include this in wrapper to suppress all warnings."""
# Code below to suppress as many warnings as possible
import os
if str(os.getenv('SUPPRES_VERBOSE_LOGGING', '0')) == '1':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import warnings
    warnings.filterwarnings("ignore")
    import logging
    logging.getLogger('tensorflow').disabled = True
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.FATAL)
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
