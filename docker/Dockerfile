ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:23.02-tf1-py3
FROM ${FROM_IMAGE_NAME}

RUN apt-get update
COPY docker/requirements-apt.txt requirements-apt.txt
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y curl apt-transport-https software-properties-common \
    && (echo "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable" | tee /etc/apt/sources.list.d/docker.list) \
    && (curl https://download.docker.com/linux/ubuntu/gpg | apt-key add -) \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive xargs apt-get install -y < requirements-apt.txt \
    && rm requirements-apt.txt
# upgrade pip
RUN pip install --upgrade pip

# Copy requirement and install
WORKDIR /workspace/

ENV TF_XLA_FLAGS="--tf_xla_enable_lazy_compilation=false tf_xla_async_io_level=0"

ARG TRT_VERSION_MAJOR=8
ARG TRT_VERSION_MINOR=5
ARG TRT_VERSION_PATCH=3
ARG TRT_VERSION_BUILD=1
ARG TRT_VERSION_MAJOR_MINOR=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR
ARG TRT_VERSION_MAJOR_MINOR_PARTCH=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR.$TRT_VERSION_PATCH
ARG TRT_VERSION_FULL=$TRT_VERSION_MAJOR_MINOR_PARTCH.$TRT_VERSION_BUILD

ARG CUDA_VERSION_MAJOR=12
ARG CUDA_VERSION_MINOR=0
ARG CUDA_VERSION_PATCH=0
ARG CUDA_VERSION_BUILD=031
ARG CUDA_VERSION_MAJOR_MINOR=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
ARG CUDA_VERSION_FULL=$CUDA_VERSION_MAJOR_MINOR.$CUDA_VERSION_PATCH.$CUDA_VERSION_BUILD
ARG CUDNN_VERSION=8.7

ENV TRT_VERSION=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR.$TRT_VERSION_PATCH.$TRT_VERSION_BUILD+cuda$CUDA_VERSION_FULL
ENV TRT_TAG "release/8.6"
RUN mkdir trt_oss_src && \
    cd trt_oss_src && \
    echo "$PWD Building TRT OSS..." && \
    git clone -b $TRT_TAG https://github.com/nvidia/TensorRT TensorRT && \
    cd TensorRT && \
    git submodule update --init --recursive && \
    mkdir -p build && cd build  && \
    cmake .. -DGPU_ARCHS="52 53 60 61 70 75 80 86 90" -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu -DTRT_BIN_DIR=`pwd`/out -DCUDA_VERSION=$CUDA_VERSION_MAJOR_MINOR -DCUDNN_VERSION=$CUDNN_VERSION && \
    make -j32  && \
    cp libnvinfer_plugin.so.8.6.* /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.$TRT_VERSION_MAJOR_MINOR_PARTCH && \
    cp libnvinfer_plugin_static.a /usr/lib/x86_64-linux-gnu/libnvinfer_plugin_static.a && \
    #    cp libnvonnxparser.so.8.6.* /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.$TRT_VERSION_MAJOR_MINOR_PARTCH && \
    cp libnvcaffeparser.so.8.6.* /usr/lib/x86_64-linux-gnu/libnvcaffeparser.so.$TRT_VERSION_MAJOR_MINOR_PARTCH && \
    cp trtexec /usr/local/bin/ && \
    rm -rf trt_oss_src

RUN pip install --upgrade setuptools==58.3.0
RUN pip install nvidia-pyindex
COPY docker/requirements-pip.txt /workspace/
RUN pip install -r requirements-pip.txt --use-deprecated=legacy-resolver && \
    rm -rf requirements-pip.txt

COPY third_party/horovod/keras_allreduce_fix.patch ./keras_allreduce_fix.patch
RUN HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_NCCL_LINK=SHARED \
    HOROVOD_WITH_PYTORCH=0 \
    HOROVOD_WITH_TENSORFLOW=1 \
    HOROVOD_WITH_GLOO=1 \
    python -m pip install horovod==0.28.0

# Patch keras with the patches from /opt/third_party/keras.
COPY third_party/keras/monkey.patch ./monkey.patch
RUN patch -run \
    -d /usr/local/lib/python3.8/dist-packages/keras < ./monkey.patch && \
    rm -rf ./monkey.patch

# Create user that will run commands
ARG user_id=1000
ARG user_name=developer
ARG groups=developer:1000
ARG home=/home/developer
RUN echo "ALL   ALL = (ALL) NOPASSWD: ALL" > /etc/sudoers \
    && mkdir -p "$(dirname $home)"
