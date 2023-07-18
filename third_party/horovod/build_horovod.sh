#!/usr/bin/env bash

# This script gets run inside dazel docker and compiles a pip wheel for horovod:
# dazel run third_party/horovod:build_horovod

set -eo pipefail

TAO_TOOLKIT_TF_ROOT=$PWD

# # Create a tempdir to work with FW team's repos.
# TEMPDIR=`mktemp -d`
# echo 'Creating tempdir' $TEMPDIR
# cd $TEMPDIR

# Clone the horovod repo.
cd $TAO_TOOLKIT_TF_ROOT/third_party/horovod
git clone --recursive https://github.com/horovod/horovod.git
cd horovod

# Checkout the required commit from the horovod repo.
git checkout v0.24.0

# Apply patches if any
git apply $TAO_TOOLKIT_TF_ROOT/third_party/horovod/keras_allreduce_fix.patch

# Build the wheel inside the tao-toolkit-tf base container in the bazel environment.
HOROVOD_GPU_ALLREDUCE=NCCL \
HOROVOD_NCCL_LINK=SHARED \
HOROVOD_WITH_PYTORCH=0 \
HOROVOD_WITH_TENSORFLOW=1 \
HOROVOD_WITH_GLOO=1 \
python3 setup.py bdist_wheel

# Since we mounted the /tmp/.../horovod folder inside the docker, the final wheel gets generated
# inside /tmp/.../horovod/dist folder.
WHEEL_NAME="horovod-0.24.0-cp38-cp38-linux_x86_64.whl"
SUM="$(sha256sum dist/$WHEEL_NAME | awk '{print $1;}')"

python3 -m pip install dist/$WHEEL_NAME

# Remove the repository.
cd ..
rm -rf horovod
