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
"""FasterRCNN multi-gpu wrapper for train script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import subprocess


def parse_args(args_in=None):
    """Argument parser."""
    parser = argparse.ArgumentParser(description=('Train or retrain a Faster-RCNN model' +
                                                  ' using one or more GPUs.'))
    parser.add_argument("-e",
                        "--experiment_spec",
                        type=str,
                        required=True,
                        help="Experiment spec file has all the training params.")
    parser.add_argument("-k",
                        "--enc_key",
                        type=str,
                        required=False,
                        help="TLT encoding key, can override the one in the spec file.")
    parser.add_argument("-g",
                        "--gpus",
                        type=int,
                        default=None,
                        help="Number of GPUs for multi-gpu training.")
    return parser.parse_known_args(args_in)[0]


def main():
    '''main function for training.'''
    args = parse_args()
    np = args.gpus or 1
    # key is optional, we pass it to subprocess only if it exists
    if args.enc_key is not None:
        key_arg = ['-k', args.enc_key]
    else:
        key_arg = []
    train_script = 'nvidia_tao_tf1/cv/faster_rcnn/scripts/train.py'
    if np > 1:
        # multi-gpu training
        ret = subprocess.run(['mpirun', '-np', str(np),
                               '--oversubscribe',
                               '--bind-to', 'none',
                               'python', train_script,
                               '-e', args.experiment_spec] + key_arg, shell=False).returncode
    elif np in [None, 1]:
        # fallback to single gpu training by default
        ret = subprocess.run(['python', train_script,
                               '-e', args.experiment_spec] + key_arg, shell=False).returncode
    else:
        raise(
            ValueError(
                (
                    'Invalid value of GPU number specified: {}, '
                    'should be non-negative.'.format(np)
                )
            )
        )

    assert ret == 0, 'Multi-gpu training failed.'


if __name__ == '__main__':
    main()
