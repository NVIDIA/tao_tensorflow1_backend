CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python
WORKSPACE ?= /workspace/tao-tf1

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14
NVCC_FLAGS= -I${WORKSPACE}/third_party
NVCC_OPTS = --cicc-options --no-opt-byval
LDFLAGS = -shared ${TF_LFLAGS}

spatialtransform_srcs = ./spatialtransform.cc
spatialtransform_lib = ../op_spatialtransform.so
spatialtransform_gpu_lib = ./spatialtransform.cu.o

spatialtransform_op: $(spatialtransform_lib)
$(spatialtransform_lib): $(spatialtransform_srcs) $(spatialtransform_gpu_lib)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS} ${NVCC_FLAGS} -D GOOGLE_CUDA=1  -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart

spatialtransform_gpu_only: $(spatialtransform_gpu_lib)
$(spatialtransform_gpu_lib): $(wildcard ./*.cu.cc)
	$(NVCC) -std=c++11 -c -o $@ $^  $(TF_CFLAGS) ${NVCC_FLAGS} ${NVCC_OPTS} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr


clean:
	rm -rf ../op_spatialtransform.* ${spatialtransform_gpu_lib}
