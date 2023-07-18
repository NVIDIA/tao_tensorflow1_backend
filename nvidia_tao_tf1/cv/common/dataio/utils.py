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

"""Commonly used methods and classes across dataio."""

from collections import defaultdict
import datetime
import errno
import logging
import os
from nvidia_tao_tf1.core.utils.path_utils import expand_path

def is_kpi_set(set_id):
    """Return if set id is a KPI set."""
    lowercase_set_id = set_id.lower()
    return ('kpi' in lowercase_set_id or
            's532-gaze-' in lowercase_set_id or
            's534-gaze-' in lowercase_set_id or
            's593-gaze-' in lowercase_set_id or
            's594-gaze-' in lowercase_set_id or
            's595-gaze-' in lowercase_set_id or
            's596-gaze-' in lowercase_set_id or
            's597-gaze-' in lowercase_set_id)


def is_int(num_str):
    """Return if python string is a number."""
    try:
        int(num_str)
        return True
    except Exception:
        return False


def get_file_name_noext(filepath):
    """Return file name witout extension."""
    return os.path.splitext(os.path.basename(filepath))[0]


def get_file_ext(filepath):
    """Return file extension."""
    return os.path.splitext(os.path.basename(filepath))[1]


def mkdir(path):
    """Make a directory if one does not exist already."""
    try:
        os.makedirs(expand_path(path))
    except OSError as ex:
        if ex.errno != errno.EEXIST:
            raise


class SysOutLogger(object):
    """Class which logs messages to output when FileHandler cannot establish cosmos connection."""

    def __init__(self):
        """Constructor for SysOutLogger."""
        pass

    @staticmethod
    def _format_output(msg_type, msg):
        return '{} {}: {}'.format(
            msg_type,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            msg)

    def info(self, msg):
        """Output info msg."""
        print(self._format_output('INFO', msg))

    def error(self, msg):
        """Output error msg."""
        print(self._format_output('ERROR', msg))

    def warning(self, msg):
        """Output warning msg."""
        print(self._format_output('WARN', msg))


class Logger(object):
    """Class which formats logging file."""

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')

    def __init__(self, loggername, filename, level):
        """Initialize logger."""
        try:
            handler = logging.FileHandler(filename)
            handler.setFormatter(Logger.formatter)

            self._logger = logging.getLogger(loggername)
            self._logger.setLevel(level)
            self._logger.addHandler(handler)
        except IOError:
            print('Check NFS cosmos connection.')
            self._logger = SysOutLogger()

    def get_logger(self):
        """Return formatted logger."""
        return self._logger


class PipelineReporter(object):
    """Class which logs information about runs."""

    def __init__(self, log_path, script_name, set_id):
        """Initialize parameters, create log folders, and format logging."""
        self.set_id = set_id
        self.set_to_info = defaultdict(list)
        self.set_to_errors = defaultdict(list)
        self.set_to_warnings = defaultdict(list)
        self._create_log_folders(
            log_path,
            script_name)
        self._create_set_loggers()

    def _create_log_folders(self, log_path, script_name):
        run_folder_name = script_name + '_' + datetime.datetime.now().strftime('%Y-%m-%d')
        log_set_path = os.path.join(
            log_path,
            run_folder_name)

        self.error_set_path = os.path.join(log_set_path, 'Error')
        self.warning_set_path = os.path.join(log_set_path, 'Warning')
        mkdir(self.error_set_path)
        mkdir(self.warning_set_path)

    def _create_set_loggers(self):
            self.err_logger = Logger(
                self.set_id + '_error',
                os.path.join(
                    self.error_set_path,
                    self.set_id + '.log'),
                logging.INFO).get_logger()
            self.warning_logger = Logger(
                self.set_id + '_warning',
                os.path.join(
                    self.warning_set_path,
                    self.set_id + '.log'),
                logging.WARNING).get_logger()

    def add_info(self, info_msg):
        """Log info under set_id."""
        self.set_to_info[self.set_id].append(info_msg)

    def add_error(self, err_msg):
        """Log error under set_id."""
        self.set_to_errors[self.set_id].append(err_msg)

    def add_warning(self, warning_msg):
        """Log warning under set_id."""
        self.set_to_warnings[self.set_id].append(warning_msg)

    def write_to_log(self):
        """Write stored dicts of msgs to logs."""
        # Currently info log to error log for convenience of reading num of tfrecords
        for info_msg in self.set_to_info[self.set_id]:
            self.err_logger.info(info_msg)

        for err_msg in self.set_to_errors[self.set_id]:
            self.err_logger.error(err_msg)

        for warning_msg in self.set_to_warnings[self.set_id]:
            self.warning_logger.warning(warning_msg)
