CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python
WORKSPACE ?= /workspace/tao-tf1
CUDA_INC_PATH = /usr/local/cuda/targets/x86_64-linux/include
CUDA_LIB_PATH = /usr/local/cuda/targets/x86_64-linux/lib

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14
NVCC_FLAGS= -I${WORKSPACE}/third_party
NVCC_OPTS = --cicc-options --no-opt-byval
LDFLAGS = -shared ${TF_LFLAGS}

binary_to_distance_srcs = ./binary_to_distance.cc
binary_to_distance_lib = ../op_binary_to_distance.so
binary_to_distance_gpu_lib = ./binary_to_distance.cu.o

binary_to_distance_op: $(binary_to_distance_lib)
$(binary_to_distance_lib): $(binary_to_distance_srcs) $(binary_to_distance_gpu_lib)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS} ${NVCC_FLAGS} -D GOOGLE_CUDA=1  -I${CUDA_INC_PATH} -L${CUDA_LIB_PATH} -lcudart

binary_to_distance_gpu_only: $(binary_to_distance_gpu_lib)
$(binary_to_distance_gpu_lib): $(wildcard ./*.cu.cc)
	$(NVCC) -std=c++11 -c -o $@ $^  $(TF_CFLAGS) ${NVCC_FLAGS} ${NVCC_OPTS} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

clean:
	rm -rf ${binary_to_distance_lib} ${binary_to_distance_gpu_lib}