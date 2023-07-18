#!/usr/bin/env bash
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
NUM_GPUS=0
PYTHON_ARGS=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -np)
    NUM_GPUS="$2"
    shift # Skip argument.
    shift # Skip value.
    ;;
    *)    # Unknown option, pass these to python.
    PYTHON_ARGS+=("$1")
    shift # Skip argument.
    ;;
esac
done

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Usage: -np <num GPUs> [train.py arguments]"
else
    # Note: need to execute bazel created train script instead of train.py.
    mpirun -np $NUM_GPUS -map-by slot --bind-to none \
    -x NCCL_DEBUG=VERSION \
    -x LD_LIBRARY_PATH \
    -x PATH \
    -mca pml ob1 -mca btl ^openib \
    --allow-run-as-root \
    nvidia_tao_tf1/cv/mask_rcnn/scripts/train ${PYTHON_ARGS[*]}
fi

