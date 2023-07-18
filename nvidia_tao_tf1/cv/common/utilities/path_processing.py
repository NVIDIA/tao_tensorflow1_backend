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

"""Utility function definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import json
import os
import yaml


def mkdir_p(new_path):
    """Makedir, making also non-existing parent dirs."""
    try:
        print(new_path)
        os.makedirs(new_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(new_path):
            pass
        else:
            raise


def get_file_name_noext(filepath):
    """Return file name witout extension."""
    return os.path.splitext(os.path.basename(filepath))[0]


def get_file_ext(filepath):
    """Return file extension."""
    return os.path.splitext(os.path.basename(filepath))[1]


def check_file_or_directory(path):
    """Check if the input path is a file or a directory."""
    if not os.path.isfile(path) and not os.path.isdir:
        raise FileNotFoundError('%s is not a directory or a file.')


def check_file(file_path, extension=None):
    """Check if the input path is a file.

    Args:
        file_path (str): Full path to the file
        extension (str): File extension. If provided,
            checks if the extensions match.
            Example choices: [`.yaml`, `.json` ...]
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError('The file %s does not exist' % file_path)
    # Check if the extension is right
    if extension is not None and get_file_ext(file_path) != extension:
        raise FileNotFoundError('The file %s is not a %s file' % extension)


def check_dir(dir_path):
    """Check if the input path is a directory."""
    if not os.path.isdir(dir_path):
        raise NotADirectoryError('The directory %s does not exist' % dir_path)


def load_yaml_file(file_path, mode='r'):
    """Load a yaml file.

    Args:
        file_path (str): path to the yaml file.
        mode (str): mode to load the file in. ex. 'r', 'w' etc.
    """
    check_file(file_path, '.yaml')
    with open(file_path, mode) as yaml_file:
        file_ = yaml.load(yaml_file.read())
    return file_


def load_json_file(file_path, mode='r'):
    """Load a json file.

    Args:
        file_path (str): path to the json file.
        mode (str): mode to load the file in. ex. 'r', 'w' etc.
    """
    check_file(file_path, '.json')
    with open(file_path, mode) as json_file:
        file_ = json.load(json_file)
    return file_
