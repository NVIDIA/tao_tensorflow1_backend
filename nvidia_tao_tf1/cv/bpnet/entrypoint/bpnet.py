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

"""TLT command line wrapper to invoke CLI scripts."""

import sys
import nvidia_tao_tf1.cv.bpnet.scripts
from nvidia_tao_tf1.cv.common.entrypoint.entrypoint import launch_job


def main():
    """Function to launch the job."""
    launch_job(nvidia_tao_tf1.cv.bpnet.scripts, "bpnet", sys.argv[1:])


if __name__ == "__main__":
    main()
