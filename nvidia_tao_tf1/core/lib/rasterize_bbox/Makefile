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

rasterize_bbox_srcs = ./rasterize_bbox.cc
rasterize_bbox_lib = ../op_rasterize_bbox.so
rasterize_bbox_gpu_lib = ./rasterize_bbox.cu.o

rasterize_bbox_op: $(rasterize_bbox_lib)
$(rasterize_bbox_lib): $(rasterize_bbox_srcs) $(rasterize_bbox_gpu_lib)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS} ${NVCC_FLAGS} -D GOOGLE_CUDA=1  -I${CUDA_INC_PATH} -L${CUDA_LIB_PATH} -lcudart

rasterize_bbox_gpu_only: $(rasterize_bbox_gpu_lib)
$(rasterize_bbox_gpu_lib): $(wildcard ./*.cu.cc)
	$(NVCC) -std=c++11 -c -o $@ $^  $(TF_CFLAGS) ${NVCC_FLAGS} ${NVCC_OPTS} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

clean:
	rm -rf ${rasterize_bbox_lib} ${rasterize_bbox_gpu_lib}
