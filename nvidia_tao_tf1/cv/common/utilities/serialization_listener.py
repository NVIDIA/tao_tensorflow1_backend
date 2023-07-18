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

"""Hook for serializing Keras models."""

import glob
import logging
import os
import re

from keras import backend as K
from nvidia_tao_tf1.core.hooks import KerasCheckpointListener

logger = logging.getLogger(__name__)


class EpochModelSerializationListener(KerasCheckpointListener):
    """Adds metadata to serialized keras model."""

    def __init__(
            self,
            checkpoint_dir,
            model,
            key,
            steps_per_epoch=None,
            max_to_keep=None,
            after_save_callable=None,
            prefix="model"):
        """Constructor.

        Args:
            checkpoint_dir (str): Base directory for the checkpoint files.
            model (keras.models.Model): Instance of the model to serialize.
            postprocessing_config: postprocessing_config_pb2.PostProcessingConfig object.
            key (str): A key string to serialize the model during the experiment.
        """
        super(EpochModelSerializationListener, self).__init__(
            model=model,
            checkpoint_dir=checkpoint_dir,
            max_to_keep=max_to_keep,
            after_save_callable=after_save_callable,
            prefix=prefix)
        self._key = key
        self._steps_per_epoch = steps_per_epoch

    def begin(self):
        """Called after starting the session."""
        pattern = r"^%s.epoch-(\d+)\.hdf5$" % re.escape(
            os.path.join(self._checkpoint_dir, self._prefix)
        )
        compiled = re.compile(pattern)

        def extract_model_number(filename):
            s = compiled.findall(filename)
            return int(s[0]) if s else -1, filename

        filenames = glob.glob(os.path.join(self._checkpoint_dir, "*.hdf5"))
        # Weed out filenames that do not match the pattern.
        filenames = [
            filename for filename in filenames if compiled.match(filename) is not None
        ]
        sorted_filenames = sorted(filenames, key=extract_model_number)
        self._latest_checkpoints.extend(sorted_filenames)

    def after_save(self, session, global_step_value):
        """Serialize metadata to the tlt file after it has been saved."""
        if session:
            K.set_session(session)
            K.manual_variable_initialization(True)
        epoch = int(global_step_value / self._steps_per_epoch)
        model_path = os.path.join(self._checkpoint_dir, 'model.epoch-%s.hdf5' % epoch)
        self._model.save_model(file_name=model_path)
        self._cleanup(model_path)

    def end(self, session, global_step_value):
        """Run at the end of the session, reset the old variale initialization setting."""
        K.manual_variable_initialization(self._previous_MANUAL_VAR_INIT)
