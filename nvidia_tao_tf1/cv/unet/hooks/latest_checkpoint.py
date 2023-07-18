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

"""Hook for resuming from the recent most checkpoint."""

import logging
import os
import tempfile
from zipfile import BadZipFile, ZipFile

from nvidia_tao_tf1.encoding import encoding

logger = logging.getLogger(__name__)


class LatestCheckpoint(object):
    """Latest checkpoint hook to retrieve the recent checkpoint."""

    def __init__(self, key, model_dir):
        """Initialize LatestCheckpoint.

        Args:
            key (str): The key to encrypt the trained model.
            model_dir (str): The path to retrieve the latest checkpoint saved.
        """

        self._temp_dir = tempfile.mkdtemp()
        self.ckpt = None
        self.model_json = None
        tmp_path, model_json, encrypted = self.get_latest_checkpoint(model_dir, key)
        self.ckpt = tmp_path
        self.model_json = model_json
        if tmp_path and encrypted:
            with open(os.path.join(self._temp_dir, "checkpoint"), "r") as f:
                old_path = f.readline()
                old_path = old_path.split(":")[-1]
                old_dir = os.path.dirname(old_path)
                self._temp_dir = old_dir
                

    def get_latest_checkpoint(self, results_dir, key):
        """Get the latest checkpoint path from a given results directory.

        Parses through the directory to look for the latest checkpoint file
        and returns the path to this file.

        Args:
            results_dir (str): Path to the results directory.

        Returns:
            ckpt_path (str): Path to the latest checkpoint.
        """

        # Adding this to avoid error on NGC when the results is not created yet
        if not os.path.exists(results_dir):
            return None, None, None
        trainable_ckpts = [int(item.split('.')[1].split('-')[1])
                           for item in os.listdir(results_dir) if item.endswith(".tlt")]
        num_ckpts = len(trainable_ckpts)
        if num_ckpts == 0:
            return None, None, None
        latest_step = sorted(trainable_ckpts, reverse=True)[0]
        latest_checkpoint = os.path.join(results_dir,
                                         "model.epoch-{}.tlt".format(latest_step))
        logger.info("Getting the latest checkpoint for restoring {}".format(latest_checkpoint))
        return self.get_tf_ckpt(latest_checkpoint, key, latest_step)

    def get_tf_ckpt(self, ckzip_path, enc_key, latest_step):
        """Simple function to extract and get a trainable checkpoint.

        Args:
            ckzip_path (str): Path to the encrypted checkpoint.

        Returns:
            tf_ckpt_path (str): Path to the decrypted tf checkpoint
        """
        encrypted = False
        if os.path.isdir(ckzip_path):
            temp_checkpoint_path = ckzip_path
        else:
            encrypted = True
            # Set-up the temporary directory.
            temp_checkpoint_path = self._temp_dir
            os_handle, temp_zip_path = tempfile.mkstemp()
            temp_zip_path = temp_zip_path+".zip"
            os.close(os_handle)
            # Decrypt the checkpoint file.
            with open(ckzip_path, 'rb') as encoded_file, open(temp_zip_path, 'wb') as tmp_zip_file:
                encoding.decode(encoded_file, tmp_zip_file, bytes(enc_key, 'utf-8'))
            encoded_file.closed
            tmp_zip_file.closed

            # Load zip file and extract members to a tmp_directory.
            try:
                with ZipFile(temp_zip_path, 'r') as zip_object:
                    for member in zip_object.namelist():
                        zip_object.extract(member, path=temp_checkpoint_path)
            except BadZipFile:
                raise ValueError(
                    "The zipfile extracted was corrupt. Please check your key or "
                    "re-launch the training."
                )
            except Exception:
                raise IOError(
                    "The last checkpoint file is not saved properly. "
                    "Please delete it and rerun the script."
                )
            # Removing the temporary zip path.
            os.remove(temp_zip_path)
        json_files = [os.path.join(temp_checkpoint_path, f) for f in
                      os.listdir(temp_checkpoint_path) if f.endswith(".json")]
        if len(json_files) > 0:
            model_json = json_files[0]
        else:
            model_json = None
        return os.path.join(temp_checkpoint_path,
                            "model.ckpt-{}".format(latest_step)), model_json, encrypted
