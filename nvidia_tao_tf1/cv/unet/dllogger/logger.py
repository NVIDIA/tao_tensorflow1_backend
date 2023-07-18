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

"""Logger functions."""

from abc import ABC, abstractmethod
import atexit
from collections import defaultdict
from datetime import datetime
import json


class Backend(ABC):
    """Class Backend to set the verbosity logs."""

    def __init__(self, verbosity):
        """Instantiate the Verbosity backend.

        Args:
            verbosity (bool): Flag to set to give additional information during.
            training.
        """
        self._verbosity = verbosity

    @property
    def verbosity(self):
        """Instantiate the Verbosity backend."""

        return self._verbosity

    @abstractmethod
    def log(self, timestamp, elapsedtime, step, data):
        """Instantiate the logger."""

        pass

    @abstractmethod
    def metadata(self, timestamp, elapsedtime, metric, metadata):
        """Function to get the metadata."""

        pass


class Verbosity:
    """Class for verbosity logs."""

    OFF = -1
    DEFAULT = 0
    VERBOSE = 1


class Logger:
    """Class for logger."""

    def __init__(self, backends):
        """Instantiate the Verbosity backend.

        Args:
            backend (type): The type of backend to be used for logging e.g.,Json.
        """
        self.backends = backends
        atexit.register(self.flush)
        self.starttime = datetime.now()

    def metadata(self, metric, metadata):
        """Function to return the metadata."""

        timestamp = datetime.now()
        elapsedtime = (timestamp - self.starttime).total_seconds()
        for b in self.backends:
            b.metadata(timestamp, elapsedtime, metric, metadata)

    def log(self, step, data, verbosity=1):
        """Function to return the log."""

        timestamp = datetime.now()
        elapsedtime = (timestamp - self.starttime).total_seconds()
        for b in self.backends:
            if b.verbosity >= verbosity:
                b.log(timestamp, elapsedtime, step, data)

    def flush(self):
        """Function to return the step value."""

        for b in self.backends:
            b.flush()


def default_step_format(step):
    """Function to return the step value."""

    return str(step)


def default_metric_format(metric, metadata, value):
    """Function to return the metric format."""

    unit = metadata["unit"] if "unit" in metadata.keys() else ""
    formatm = "{" + metadata["format"] + "}" if "format" in metadata.keys() else "{}"
    return "{}:{} {}".format(
        metric, formatm.format(value) if value is not None else value, unit
    )


def default_prefix_format(timestamp):
    """Function to return the prefix format."""

    return "DLL {} - ".format(timestamp)


class StdOutBackend(Backend):
    """Class for StdOutBackend."""

    def __init__(
        self,
        verbosity,
        step_format=default_step_format,
        metric_format=default_metric_format,
        prefix_format=default_prefix_format,
    ):
        """Instantiate the dataloader.

        Args:
            verbosity (bool): List of tuples (tfrecord_file_pattern,
                image_directory_path) to use for training.
            step_format (int): Number related to the time format.
            metric_format (int): Number related to the time format.
            prefix_format (int): Number related to the time format.
        """
        super().__init__(verbosity=verbosity)

        self._metadata = defaultdict(dict)
        self.step_format = step_format
        self.metric_format = metric_format
        self.prefix_format = prefix_format
        self.elapsed = 0.0

    def metadata(self, timestamp, elapsedtime, metric, metadata):
        """Function to return the prefix format."""

        self._metadata[metric].update(metadata)

    def log(self, timestamp, elapsedtime, step, data):
        """Function to return the prefix format."""

        print(
            "{}{} {}{}".format(
                self.prefix_format(timestamp),
                self.step_format(step),
                " ".join(
                    [
                        self.metric_format(m, self._metadata[m], v)
                        for m, v in data.items()
                    ]
                ),
                "elapsed_time:"+str(elapsedtime)
            )
        )

    def flush(self):
        """Function to return the prefix format."""

        pass


class JSONStreamBackend(Backend):
    """Class for Json stream backend."""

    def __init__(self, verbosity, filename):
        """Instantiate the Json stream backend for logging.

        Args:
            verbosity (bool): Verbosity flag to set for
            filename (str): Filename to save the logs.
        """

        super().__init__(verbosity=verbosity)
        self._filename = filename
        self.file = open(filename, "w")
        atexit.register(self.file.close)

    def metadata(self, timestamp, elapsedtime, metric, metadata):
        """Function to return the prefix format."""

        self.file.write(
            "DLLL {}\n".format(
                json.dumps(
                    dict(
                        timestamp=str(timestamp.timestamp()),
                        elapsedtime=str(elapsedtime),
                        datetime=str(timestamp),
                        type="METADATA",
                        metric=metric,
                        metadata=metadata,
                    )
                )
            )
        )

    def log(self, timestamp, elapsedtime, step, data):
        """Function to return the prefix format."""

        self.file.write(
            "DLLL {}\n".format(
                json.dumps(
                    dict(
                        timestamp=str(timestamp.timestamp()),
                        datetime=str(timestamp),
                        elapsedtime=str(elapsedtime),
                        type="LOG",
                        step=step,
                        data=data,
                    )
                )
            )
        )

    def flush(self):
        """Function to return the prefix format."""

        self.file.flush()
